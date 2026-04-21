package main

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/notebook"
)

var GitVersion string

const defaultAddr = ":8888"

func main() {
	ex, err := os.Executable()
	if err != nil {
		panic(err)
	}
	exPath := filepath.Dir(ex)

	cfg := &config.Config{}
	args := os.Args[1:]
	cfg.Load(args)
	cfg.AdjustRelativePaths(exPath)

	output := zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}
	output.FormatLevel = func(i interface{}) string {
		return strings.ToUpper(fmt.Sprintf("| %-6s|", i))
	}
	logger := zerolog.New(output).Level(zerolog.InfoLevel).With().Timestamp().Logger()
	if cfg.GetBool("debug") {
		logger = logger.Level(zerolog.DebugLevel)
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	}
	zerolog.DefaultContextLogger = &logger
	log.Logger = logger

	addr := os.Getenv("MACONDO_NOTEBOOK_ADDR")
	if addr == "" {
		addr = defaultAddr
	}

	srv := notebook.NewServer(cfg, exPath, GitVersion)

	idleConnsClosed := make(chan struct{})
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		log.Info().Msg("shutting down notebook server...")
		close(idleConnsClosed)
	}()

	log.Info().Msgf("Macondo Notebook %s starting at http://localhost%s", GitVersion, addr)
	go func() {
		if err := http.ListenAndServe(addr, srv.Handler()); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("notebook server failed")
		}
	}()

	// Open browser after a short delay
	go openBrowser("http://localhost" + addr)

	<-idleConnsClosed
	log.Info().Msg("notebook server stopped")
}

func openBrowser(url string) {
	time.Sleep(500 * time.Millisecond)
	log.Info().Msgf("Notebook available at: %s", url)

	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", url)
	default: // linux and others
		cmd = exec.Command("xdg-open", url)
	}
	if err := cmd.Start(); err != nil {
		log.Debug().Err(err).Msg("could not open browser automatically")
	}
}
