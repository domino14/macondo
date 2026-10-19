// make_all_wmps builds a .wmp word map for every lexicon a game can be
// created with on woogles, ready to be uploaded to the data volume.
//
// A lexicon's word graph is fetched if it isn't already on disk, and its
// letter distribution is derived from its name the same way the server derives
// it, so a word map is never built against the wrong alphabet.
//
// Two lexica cannot have a word map at all: NSF25 and OSPS52 have 33-letter
// alphabets, and a BitRack holds 32. Those are reported and skipped rather
// than failing the run.
//
// Usage:
//
//	go run ./cmd/make_all_wmps                     # into $MACONDO_DATA_PATH/lexica
//	go run ./cmd/make_all_wmps -out /tmp/wmps      # somewhere else
//	go run ./cmd/make_all_wmps -lexica CSW24,NWL23 # just these
//	go run ./cmd/make_all_wmps -force              # rebuild ones already present
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	wglconfig "github.com/domino14/word-golib/config"
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/lexicon"
	"github.com/domino14/macondo/wmp"
)

// defaultLexica is liwords' AllowedNewGameLexica, from pkg/entity/lexica.go in
// that repo. It has to be kept in step with it by hand: a lexicon that can be
// used to start a game and has no word map on the volume just falls back to
// the slower KWG move generator, so drift here is quiet rather than loud.
var defaultLexica = []string{
	"CSW24X",
	"CSW24",
	"ECWL",
	"FILE2017",
	"FRA24",
	"NSF25",
	"NSWL23",
	"NWL23",
	"RD29",
	"DISC2",
	"OSPS52",
	"SLV26",
}

// WMP only supports 15x15 boards.
const boardDim = 15

type result struct {
	lexicon string
	status  string // built, present, unsupported, failed
	detail  string
	bytes   int64
	elapsed time.Duration
}

func main() {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)

	lexicaFlag := flag.String("lexica", strings.Join(defaultLexica, ","),
		"comma-separated lexica to build word maps for")
	outDir := flag.String("out", "",
		"directory to write .wmp files into (defaults to $MACONDO_DATA_PATH/lexica, where macondo looks for them)")
	threads := flag.Int("threads", runtime.NumCPU(), "builder threads per word map")
	force := flag.Bool("force", false, "rebuild word maps that are already present")
	flag.Parse()

	cfg := config.DefaultConfig()
	if err := cfg.Load(nil); err != nil {
		log.Fatal().Err(err).Msg("could not load config")
	}
	wglCfg := cfg.WGLConfig()

	dir := *outDir
	if dir == "" {
		dir = filepath.Join(wglCfg.DataPath, "lexica")
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Fatal().Err(err).Str("dir", dir).Msg("could not create output directory")
	}

	var results []result
	for _, lex := range strings.Split(*lexicaFlag, ",") {
		lex = strings.TrimSpace(lex)
		if lex == "" {
			continue
		}
		results = append(results, buildOne(wglCfg, dir, lex, *threads, *force))
	}

	report(results, dir)
}

func buildOne(wglCfg *wglconfig.Config, dir, lex string, threads int, force bool) result {
	out := filepath.Join(dir, lex+".wmp")
	if !force {
		if fi, err := os.Stat(out); err == nil {
			return result{lexicon: lex, status: "present", bytes: fi.Size(),
				detail: "already on disk; -force to rebuild"}
		}
	}

	// Derive the letter distribution from the lexicon, the way liwords'
	// LetterDistributionForLexicon does. Building against the wrong alphabet
	// produces a word map that indexes tiles that don't exist.
	ldName, err := tilemapping.ProbableLetterDistributionName(lex)
	if err != nil {
		return result{lexicon: lex, status: "failed", detail: "no known letter distribution"}
	}
	ld, err := tilemapping.NamedLetterDistribution(wglCfg, ldName)
	if err != nil {
		return result{lexicon: lex, status: "failed", detail: err.Error()}
	}
	if err := wmp.CheckCompatible(ld, boardDim); err != nil {
		return result{lexicon: lex, status: "unsupported", detail: err.Error()}
	}

	if err := lexicon.EnsureLexiconFile(lex, ".kwg", wglCfg); err != nil {
		return result{lexicon: lex, status: "failed", detail: "word graph unavailable: " + err.Error()}
	}
	gd, err := kwg.GetKWG(wglCfg, lex, kwg.WithDistribution(ldName))
	if err != nil {
		return result{lexicon: lex, status: "failed", detail: err.Error()}
	}

	log.Info().Str("lexicon", lex).Str("ld", ldName).Int("threads", threads).
		Msg("building word map...")
	start := time.Now()
	w, err := wmp.MakeFromKWG(gd, ld, boardDim, threads)
	if err != nil {
		return result{lexicon: lex, status: "failed", detail: err.Error()}
	}
	if err := w.WriteToFile(out); err != nil {
		return result{lexicon: lex, status: "failed", detail: err.Error()}
	}
	elapsed := time.Since(start)
	var size int64
	if fi, err := os.Stat(out); err == nil {
		size = fi.Size()
	}
	return result{lexicon: lex, status: "built", bytes: size, elapsed: elapsed}
}

func report(results []result, dir string) {
	fmt.Fprintf(os.Stderr, "\nword maps in %s\n\n", dir)
	failed := 0
	for _, r := range results {
		switch r.status {
		case "built":
			fmt.Fprintf(os.Stderr, "  built        %-10s %8.1f MB  %s\n",
				r.lexicon, float64(r.bytes)/(1<<20), r.elapsed.Round(time.Millisecond))
		case "present":
			fmt.Fprintf(os.Stderr, "  present      %-10s %8.1f MB  %s\n",
				r.lexicon, float64(r.bytes)/(1<<20), r.detail)
		case "unsupported":
			fmt.Fprintf(os.Stderr, "  unsupported  %-10s %s\n", r.lexicon, r.detail)
		default:
			failed++
			fmt.Fprintf(os.Stderr, "  FAILED       %-10s %s\n", r.lexicon, r.detail)
		}
	}
	fmt.Fprintln(os.Stderr)
	if failed > 0 {
		log.Fatal().Int("failed", failed).Msg("some word maps could not be built")
	}
}
