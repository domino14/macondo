package main

import (
	_ "embed"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/shell"
)

var (
	GitVersion string
)

const (
	GracefulShutdownTimeout = 20 * time.Second
)

//go:embed macondo.txt
var macondobanner string

func main() {

	// Determine the directory of the executable. We will use this
	// directory to find the data files if an absolute path is not
	// provided for these!
	ex, err := os.Executable()
	if err != nil {
		panic(err)
	}
	exPath := filepath.Dir(ex)
	fmt.Println(macondobanner)
	fmt.Println(GitVersion)

	log.Info().Msgf("executable path: %v", exPath)

	cfg := &config.Config{}
	args := os.Args[1:]
	cfg.Load(args)
	log.Info().Msgf("Loaded config: %v", cfg.SanitizedSettings())
	cfg.AdjustRelativePaths(exPath)

	output := zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}
	output.FormatLevel = func(i interface{}) string {
		return strings.ToUpper(fmt.Sprintf("| %-6s|", i))
	}
	output.FormatMessage = func(i interface{}) string {
		return fmt.Sprintf("%s", i)
	}
	output.FormatFieldName = func(i interface{}) string {
		return fmt.Sprintf("%s:", i)
	}

	var logger zerolog.Logger
	if cfg.GetBool("debug") {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
		logger = zerolog.New(output).Level(zerolog.DebugLevel).With().Timestamp().Logger()
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
		logger = zerolog.New(output).Level(zerolog.InfoLevel).With().Timestamp().Logger()
	}
	zerolog.DefaultContextLogger = &logger
	log.Logger = logger
	logger.Debug().Msg("Debug logging is on")
	if cfg.GetString("cpu-profile") != "" {
		f, err := os.Create(cfg.GetString("cpu-profile"))
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

	sc := shell.NewShellController(cfg, exPath, GitVersion)
	if argsLineTrimmed == "" {
		go sc.Loop(sig)
	} else {
		sc.Execute(sig, argsLineTrimmed)
		sig <- syscall.SIGINT
	}

	log.Info().Msg("started loop")

	<-idleConnsClosed

	if cfg.GetString("mem-profile") != "" {
		f, err := os.Create(cfg.GetString("mem-profile"))
		if err != nil {
			panic("could not create memory profile: " + err.Error())
		}
		defer f.Close() // error handling omitted for example
		memstats := &runtime.MemStats{}
		runtime.ReadMemStats(memstats)
		log.Info().Interface("memstats", memstats).Msg("memory-stats")

		// runtime.GC()    // get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			panic("could not write memory profile: " + err.Error())
		}
		log.Info().Msg("wrote memory profile")
	}

	sc.Cleanup()
	log.Info().Msg("server gracefully shutting down")
}
