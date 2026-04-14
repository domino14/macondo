// make_wmp builds a .wmp file from a .kwg file. Used by CI to generate
// the WMP needed for integration tests.
//
// Usage: go run ./cmd/make_wmp -kwg /path/to/CSW24.kwg -out /path/to/CSW24.wmp
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/wmp"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)

	kwgPath := flag.String("kwg", "", "path to input .kwg file (required)")
	outPath := flag.String("out", "", "path to output .wmp file (required)")
	lexicon := flag.String("lex", "CSW24", "lexicon name for cache key")
	threads := flag.Int("threads", runtime.NumCPU(), "number of builder threads")
	flag.Parse()

	if *kwgPath == "" || *outPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	cfg := config.DefaultConfig()
	ld, err := tilemapping.EnglishLetterDistribution(cfg.WGLConfig())
	if err != nil {
		fmt.Fprintf(os.Stderr, "letter distribution: %v\n", err)
		os.Exit(1)
	}

	_ = *lexicon
	gd, err := kwg.LoadKWG(cfg.WGLConfig(), *kwgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load kwg: %v\n", err)
		os.Exit(1)
	}

	// boardDim=15 for standard crossword boards. WMP only supports 15×15 boards.
	const boardDim = 15
	fmt.Fprintf(os.Stderr, "Building WMP from %s (%d threads, boardDim=%d)...\n", *kwgPath, *threads, boardDim)
	w, err := wmp.MakeFromKWG(gd, ld, boardDim, *threads)
	if err != nil {
		fmt.Fprintf(os.Stderr, "make wmp: %v\n", err)
		os.Exit(1)
	}

	if err := w.WriteToFile(*outPath); err != nil {
		fmt.Fprintf(os.Stderr, "write wmp: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Wrote %s\n", *outPath)
}
