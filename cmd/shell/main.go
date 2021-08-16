package main

import (
	"flag"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
	"strings"

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
	cfg.Load(args)
	log.Info().Msgf("Loaded config: %v", cfg)
	cfg.AdjustRelativePaths(exPath)

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

	argsLine := strings.Join(args, " ")
	argsLineTrimmed := strings.TrimSpace(argsLine)

	sc := shell.NewShellController(cfg, exPath)
	if argsLineTrimmed == "" {
		go sc.Loop(sig)
	} else {
		sc.Execute(sig, argsLineTrimmed)
		sig <- syscall.SIGINT
	}
	<-idleConnsClosed
	log.Info().Msg("server gracefully shutting down")
}
