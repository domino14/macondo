// shadow_test is a command-line tool that runs shadow agreement tests.
// It plays a configurable number of games comparing move generation with
// and without shadow, verifying 100% agreement.
//
// Usage:
//
//	go run ./cmd/shadow_test -games 1000
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
)

func main() {
	numGames := flag.Int("games", 100, "number of games to play")
	flag.Parse()

	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	zerolog.SetGlobalLevel(zerolog.InfoLevel)

	ex, err := os.Executable()
	if err != nil {
		log.Fatal().Err(err).Msg("getting executable path")
	}

	cfg := config.DefaultConfig()
	cfg.AdjustRelativePaths(filepath.Dir(ex))

	fmt.Printf("Running shadow agreement test with %d games...\n", *numGames)

	result, err := automatic.RunShadowAgreementTest(cfg, *numGames)
	if err != nil {
		log.Fatal().Err(err).Msg("shadow agreement test failed")
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

	fmt.Printf("\n100.0000%% agreement!\n")
}
