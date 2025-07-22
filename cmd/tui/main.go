package main

import (
	_ "embed"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"syscall"
	"time"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/tui"
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
	// Determine the directory of the executable
	ex, err := os.Executable()
	if err != nil {
		panic(err)
	}
	exPath := filepath.Dir(ex)
	
	// In TUI mode, let the TUI app handle logging setup
	// Don't set global level here - let TUI configure it

	cfg := &config.Config{}
	args := os.Args[1:]
	cfg.Load(args)
	cfg.AdjustRelativePaths(exPath)

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
		close(idleConnsClosed)
	}()

	// Create and run the TUI
	tuiApp := tui.NewTUIApp(cfg, exPath, GitVersion)
	
	// Run the TUI - this will block until the app exits
	if err := tuiApp.Run(); err != nil {
		fmt.Printf("TUI application error: %v\n", err)
	}

	if cfg.GetString("mem-profile") != "" {
		f, err := os.Create(cfg.GetString("mem-profile"))
		if err != nil {
			panic("could not create memory profile: " + err.Error())
		}
		defer f.Close()
		memstats := &runtime.MemStats{}
		runtime.ReadMemStats(memstats)

		if err := pprof.WriteHeapProfile(f); err != nil {
			panic("could not write memory profile: " + err.Error())
		}
	}

	tuiApp.Cleanup()
}