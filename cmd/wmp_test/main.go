// wmp_test is a command-line tool that runs WMP agreement tests.
// It plays a configurable number of deterministic games comparing the
// top play produced by the movegen with WMP off vs WMP on, verifying
// 100% agreement turn-by-turn.
//
// Usage:
//
//	go run ./cmd/wmp_test -games 100000
//	go run ./cmd/wmp_test -games 1000 -wmpfile /path/to/CSW24.wmp
//
// The WMP file is located via (in order):
//  1. -wmpfile flag, if set
//  2. $MACONDO_WMP_FILE, if set
//  3. $MACONDO_DATA_PATH/lexica/<lexicon>.wmp (or the default data path
//     under the macondo binary location)
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/wmp"
)

func main() {
	numGames := flag.Int("games", 1000, "number of games to play")
	wmpFlag := flag.String("wmpfile", "", "path to WMP file (overrides $MACONDO_WMP_FILE and data-path lookup)")
	flag.Parse()

	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	zerolog.SetGlobalLevel(zerolog.InfoLevel)

	ex, err := os.Executable()
	if err != nil {
		log.Fatal().Err(err).Msg("getting executable path")
	}

	cfg := config.DefaultConfig()
	cfg.AdjustRelativePaths(filepath.Dir(ex))

	lexicon := cfg.GetString(config.ConfigDefaultLexicon)

	wmpPath := resolveWMPPath(*wmpFlag, cfg, lexicon)
	if wmpPath == "" {
		log.Fatal().Msg("could not find a WMP file; pass -wmpfile, set $MACONDO_WMP_FILE, " +
			"or ensure $MACONDO_DATA_PATH/lexica/<lexicon>.wmp exists")
	}
	if _, err := os.Stat(wmpPath); err != nil {
		log.Fatal().Err(err).Str("path", wmpPath).Msg("WMP file not found")
	}

	fmt.Printf("Loading WMP from %s...\n", wmpPath)
	w, err := wmp.LoadFromFile(lexicon, wmpPath)
	if err != nil {
		log.Fatal().Err(err).Str("path", wmpPath).Msg("loading WMP")
	}

	fmt.Printf("Running WMP agreement test with %d games (lexicon=%s)...\n", *numGames, lexicon)

	result, err := automatic.RunWMPAgreementTest(cfg, w, *numGames)
	if err != nil {
		log.Fatal().Err(err).Msg("WMP agreement test failed")
	}

	fmt.Printf("\nResults:\n")
	fmt.Printf("  Games played:   %d\n", result.GamesPlayed)
	fmt.Printf("  Turns played:   %d\n", result.TurnsPlayed)
	fmt.Printf("  Disagreements:  %d\n", result.Disagreements)

	if result.Disagreements > 0 {
		fmt.Printf("\nDisagreement details:\n")
		for _, d := range result.Details {
			fmt.Printf("  %s\n", d)
		}
		os.Exit(1)
	}

	fmt.Printf("\n100.0000%% agreement across %d turns!\n", result.TurnsPlayed)
}

// resolveWMPPath mirrors the resolution used by the WMP tests/benches:
// flag, then $MACONDO_WMP_FILE, then <data-path>/lexica/<lexicon>.wmp.
func resolveWMPPath(flagVal string, cfg *config.Config, lexicon string) string {
	if flagVal != "" {
		return flagVal
	}
	if p := os.Getenv("MACONDO_WMP_FILE"); p != "" {
		return p
	}
	dataPath := cfg.GetString(config.ConfigDataPath)
	if dataPath == "" {
		return ""
	}
	return filepath.Join(dataPath, "lexica", lexicon+".wmp")
}
