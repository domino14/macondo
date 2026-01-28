package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/worker"
)

func main() {
	// Parse command-line flags
	debug := flag.Bool("debug", false, "Enable debug logging")
	flag.Parse()

	// Set up logging
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	if *debug {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Create Macondo config
	macondoConfig := config.DefaultConfig()
	if err := macondoConfig.Load(os.Args[1:]); err != nil {
		log.Fatal().Err(err).Msg("failed to load config")
	}

	// Adjust relative paths for data
	exePath, err := os.Executable()
	if err != nil {
		log.Fatal().Err(err).Msg("failed to get executable path")
	}
	macondoConfig.AdjustRelativePaths(exePath)

	log.Info().Interface("config", macondoConfig.SanitizedSettings()).Msg("loaded macondo config")

	// Create worker config
	workerConfig := worker.DefaultWorkerConfig()
	workerConfig.MacondoConfig = macondoConfig

	log.Info().
		Str("woogles-url", workerConfig.WooglesBaseURL).
		Msg("starting analyzer worker")

	// Create worker
	w := worker.NewAnalysisWorker(workerConfig)

	// Set up signal handling for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigCh
		log.Info().Str("signal", sig.String()).Msg("received shutdown signal")
		cancel()
	}()

	// Run the worker
	if err := w.Run(ctx); err != nil && err != context.Canceled {
		log.Fatal().Err(err).Msg("worker failed")
	}

	log.Info().Msg("analyzer worker stopped")
}
