package main

import (
	"os"
	"path/filepath"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/analyzer"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/runner"
)

func main() {
	// Determine the directory of the executable. We will use this
	// directory to find the data files if an absolute path is not
	// provided for these!
	ex, err := os.Executable()
	if err != nil {
		panic(err)
	}
	exPath := filepath.Dir(ex)

	cfg := &config.Config{}
	cfg.Load([]string{})
	log.Info().Msgf("Loaded config: %v", cfg)
	cfg.AdjustRelativePaths(exPath)
	cfg.Debug = false

	if cfg.Debug {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}

	opts := &runner.GameOptions{}
	an := analyzer.NewAnalyzer(cfg, opts)
	//analyzer.AnalyzeGCG(cfg, os.Args[1])
	an.RunTest()
}
