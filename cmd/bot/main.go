package main

import (
	"flag"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/bot"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/runner"
)

const (
	GracefulShutdownTimeout = 20 * time.Second
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
	log.Info().Msgf("Loaded config: %v, exPath: %v", cfg, exPath)
	cfg.AdjustRelativePaths(exPath)
	log.Info().Msg("adjusted paths")

	if strings.HasPrefix(cfg.LexiconPath, "./") {
		cfg.LexiconPath = filepath.Join(exPath, cfg.LexiconPath)
		log.Info().Str("path", cfg.LexiconPath).Msgf("new lexicon path")
	}
	if strings.HasPrefix(cfg.StrategyParamsPath, "./") {
		cfg.StrategyParamsPath = filepath.Join(exPath, cfg.StrategyParamsPath)
		log.Info().Str("sppath", cfg.StrategyParamsPath).Msgf("new strat params path")
	}

	if cfg.Debug {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
	idleConnsClosed := make(chan struct{})
	sig := make(chan os.Signal, 1)
	go func() {
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		// We received an interrupt signal, shut down.
		log.Info().Msg("got quit signal...")
		close(idleConnsClosed)
	}()

	opts := &runner.GameOptions{}
	b := bot.NewBot(cfg, opts)
	go bot.Main("macondo.bot", b)

	<-idleConnsClosed
	log.Info().Msg("server gracefully shutting down")
}
