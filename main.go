package main

import (
	"flag"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/shell"
)

const (
	GracefulShutdownTimeout = 20 * time.Second
)

var profilePath = flag.String("profilepath", "", "path for profile")

func main() {
	cfg := &config.Config{}
	cfg.Load(os.Args[1:])
	log.Info().Msgf("Loaded config: %v", cfg)

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

	sc := shell.NewShellController(cfg)
	go sc.Loop(sig)

	<-idleConnsClosed
	log.Info().Msg("server gracefully shutting down")
}
