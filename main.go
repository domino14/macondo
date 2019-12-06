package main

import (
	"context"
	"flag"
	"net/http"
	"os"
	"os/signal"
	"runtime/pprof"
	"syscall"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/rpc/autoplayer"
)

const (
	GracefulShutdownTimeout = 20 * time.Second
)

var profilePath = flag.String("profilepath", "", "path for profile")

func main() {
	flag.Parse()

	if *profilePath != "" {
		f, err := os.Create(*profilePath)
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	autoplayerServer := &automatic.Server{}
	handler := autoplayer.NewAutoPlayerServer(autoplayerServer, nil)

	srv := &http.Server{Addr: ":8088", Handler: handler}

	idleConnsClosed := make(chan struct{})
	sig := make(chan os.Signal, 1)
	go func() {
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		// We received an interrupt signal, shut down.
		log.Info().Msg("got quit signal...")
		ctx, cancel := context.WithTimeout(context.Background(), GracefulShutdownTimeout)

		if err := srv.Shutdown(ctx); err != nil {
			// Error from closing listeners, or context timeout:
			log.Error().Msgf("HTTP server Shutdown: %v", err)
		}
		cancel()
		close(idleConnsClosed)
	}()

	go shellLoop(sig)

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal().Err(err).Msg("")
	}
	<-idleConnsClosed
	log.Info().Msg("server gracefully shutting down")
}
