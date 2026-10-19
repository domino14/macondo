// fetch_lexica downloads the lexicon data files the test suite needs into a
// data directory, using the same code path (and the same source) as the
// shell's `set lexicon`. Files already present are left alone, so a restored
// CI cache makes this a no-op.
//
// Usage: go run ./cmd/fetch_lexica [-data-path DIR] [-list FILE]
package main

import (
	"bufio"
	"flag"
	"os"
	"path/filepath"
	"strings"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	wglconfig "github.com/domino14/word-golib/config"

	"github.com/domino14/macondo/lexicon"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)

	// The destination is taken straight from the flag or the environment
	// rather than through config.Load, which silently rewrites a data path
	// that doesn't exist yet -- exactly the case on a cold CI cache.
	dataPath := flag.String("data-path", os.Getenv("MACONDO_DATA_PATH"),
		"data directory to fetch into (defaults to $MACONDO_DATA_PATH)")
	prefix := flag.String("kwg-path-prefix", os.Getenv("MACONDO_KWG_PATH_PREFIX"),
		"subdirectory of lexica/gaddag to fetch into (defaults to $MACONDO_KWG_PATH_PREFIX)")
	listPath := flag.String("list", "cmd/fetch_lexica/lexica.txt",
		"file listing the lexicon data files to fetch, one <name><ext> per line")
	flag.Parse()

	if *dataPath == "" {
		log.Fatal().Msg("no data path: pass -data-path or set MACONDO_DATA_PATH")
	}

	entries, err := readList(*listPath)
	if err != nil {
		log.Fatal().Err(err).Str("list", *listPath).Msg("could not read list")
	}

	cfg := &wglconfig.Config{DataPath: *dataPath, KWGPathPrefix: *prefix}
	dir := filepath.Join(cfg.DataPath, "lexica", "gaddag", cfg.KWGPathPrefix)
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Fatal().Err(err).Str("dir", dir).Msg("could not create lexicon data directory")
	}

	var failed []string
	for _, entry := range entries {
		ext := filepath.Ext(entry)
		if ext == "" {
			log.Error().Str("entry", entry).Msg("list entry has no extension")
			failed = append(failed, entry)
			continue
		}
		name := strings.TrimSuffix(entry, ext)
		if err := lexicon.EnsureLexiconFile(name, ext, cfg); err != nil {
			log.Error().Err(err).Str("entry", entry).Msg("could not fetch")
			failed = append(failed, entry)
		}
	}
	if len(failed) > 0 {
		log.Fatal().Strs("files", failed).Msg("could not fetch every lexicon data file")
	}
	log.Info().Int("files", len(entries)).Str("dir", dir).Msg("lexicon data ready")
}

func readList(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var entries []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		entries = append(entries, line)
	}
	return entries, scanner.Err()
}
