package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/bot"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/turnplayer"
)

const (
	// GracefulShutdownTimeout bounds the drain after SIGTERM. ECS kills the
	// container stopTimeout after SIGTERM (30s unless the task definition
	// sets it), so this, plus a 2s flush, stays under that.
	GracefulShutdownTimeout = 25 * time.Second
)

var profilePath = flag.String("profilepath", "", "path for profile")

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
	cfg.Load(os.Args[1:])
	log.Info().Msgf("Loaded config: %v, exPath: %v", cfg.SanitizedSettings(), exPath)
	cfg.AdjustRelativePaths(exPath)
	log.Info().Msg("adjusted paths")

	if cfg.GetBool("debug") {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
	// On SIGTERM, stop taking requests and finish the ones already taken
	// (see bot.Main).
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		log.Info().Msg("got quit signal...")
	}()

	opts := &turnplayer.GameOptions{}
	b := bot.NewBot(cfg, opts)
	bot.Main(ctx, "bot.commands", b, GracefulShutdownTimeout)
	log.Info().Msg("server gracefully shutting down")
}
