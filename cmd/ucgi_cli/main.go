package main

import (
	"os"
	"path/filepath"

	"github.com/rs/zerolog"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/shell"
)

var (
	GitVersion string
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
	args := os.Args[1:]
	err = cfg.Load(args)
	if err != nil {
		panic(err)
	}

	var logger zerolog.Logger
	ll := cfg.GetString("log-level")
	switch ll {
	case "debug":
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
		logger = zerolog.New(os.Stderr).Level(zerolog.DebugLevel)
	case "info":
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
		logger = zerolog.New(os.Stderr).Level(zerolog.InfoLevel)
	case "disabled":
		zerolog.SetGlobalLevel(zerolog.Disabled)
		logger = zerolog.New(os.Stderr).Level(zerolog.Disabled)
	}
	logger.Debug().Msg("Debug logging is on")
	logger.Info().Msg("Info logging is on")

	cfg.AdjustRelativePaths(exPath)

	logger.Info().Msgf("Loaded config: %v", cfg.AllSettings())

	zerolog.DefaultContextLogger = &logger

	shell.UCGILoop(cfg)
	logger.Info().Msg("bye")
}
