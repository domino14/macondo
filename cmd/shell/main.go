package main

import (
	"os"
	"os/signal"
	"path/filepath"
	"runtime/pprof"
	"strings"
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

func main() {

	// Determine the directory of the executable. We will use this
	// directory to find the data files if an absolute path is not
	// provided for these!
	ex, err := os.Executable()
	if err != nil {
		panic(err)
	}
	exPath := filepath.Dir(ex)
	log.Info().Msgf("executable path: %v", exPath)

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

	if cfg.CPUProfile != "" {
		f, err := os.Create(cfg.CPUProfile)
		if err != nil {
			panic("could not create CPU profile: " + err.Error())
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			panic("could not start CPU profile: " + err.Error())
		}
		defer pprof.StopCPUProfile()
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

	if cfg.MemProfile != "" {
		f, err := os.Create(cfg.MemProfile)
		if err != nil {
			panic("could not create memory profile: " + err.Error())
		}
		defer f.Close() // error handling omitted for example
		// runtime.GC()    // get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			panic("could not write memory profile: " + err.Error())
		}
		log.Info().Msg("wrote memory profile")
	}

	log.Info().Msg("server gracefully shutting down")
}
