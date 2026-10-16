// playability estimates how much knowing each word is worth.
//
//	playability gen    HastyBot self-play; per-word regrets (estimator A) and a
//	                   reservoir of positions, written in resumable shards
//	playability synth  conditioned re-deals of saved positions for long words
//	                   (estimator B), written in resumable chunks
//	playability score  merges everything into per-word and per-alphagram tables
//
// See playability/regret.go for the definition.
package main

import (
	"fmt"
	"os"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: "15:04:05"})
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: playability gen|synth|score [flags]")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "gen":
		err = runGen(os.Args[2:])
	case "synth":
		err = runSynth(os.Args[2:])
	case "score":
		err = runScore(os.Args[2:])
	default:
		err = fmt.Errorf("unknown subcommand %q", os.Args[1])
	}
	if err != nil {
		log.Fatal().Err(err).Msg("playability")
	}
}
