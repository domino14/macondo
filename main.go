package main

import (
	"context"
	"flag"
	"io"
	"net/http"
	"os"
	"os/signal"
	"runtime/pprof"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/chzyer/readline"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/rpc/autoplayer"
)

const (
	GracefulShutdownTimeout = 20 * time.Second
)

var profilePath = flag.String("profilepath", "", "path for profile")

func filterInput(r rune) (rune, bool) {
	switch r {
	// block CtrlZ feature
	case readline.CharCtrlZ:
		return r, false
	}
	return r, true
}

func usage(w io.Writer) {
	io.WriteString(w, "commands:\n")
	io.WriteString(w, "load <path/to/gcg> - load a .gcg file\n")
	io.WriteString(w, "n - next play\n")
	io.WriteString(w, "b - previous play\n")
	io.WriteString(w, "turn <n> - go to turn <n>\n")
	io.WriteString(w, "gen [n] - generate n plays and sort by equity; n defaults to 15\n")
	io.WriteString(w, "add <play>\n - add a play, that looks like coords play (e.g. 10J FOO)\n")
	io.WriteString(w, "sim [plies] - start simulation, default to two-ply\n")
	io.WriteString(w, "endgame [p<maxplies>] [idoff] [sev] - run endgame, search to maxplies (4 is default)\n")
	io.WriteString(w, "    idoff - turn off iterative deepening,\n")
	io.WriteString(w, "    sev - use simple eval func (faster but less accurate)\n")
	io.WriteString(w, "solvepeg [params...] - solve pre-endgame only usable if there are 1 or 2 tiles in the bag\n")
	io.WriteString(w, "    takes in same parameters as endgame\n")
}

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

	l, err := readline.NewEx(&readline.Config{
		Prompt:      "\033[31mmacondo>\033[0m ",
		HistoryFile: "/tmp/readline.tmp",
		EOFPrompt:   "exit",

		HistorySearchFold:   true,
		FuncFilterInputRune: filterInput,
	})

	if err != nil {
		panic(err)
	}
	defer l.Close()

	go func() {
	readlineLoop:
		for {
			line, err := l.Readline()
			if err == readline.ErrInterrupt {
				if len(line) == 0 {
					break
				} else {
					continue
				}
			} else if err == io.EOF {
				break
			}
			line = strings.TrimSpace(line)
			switch {
			case line == "bye" || line == "exit":
				sig <- syscall.SIGINT
				break readlineLoop
			case line == "help":
				usage(l.Stderr())
			case line == "":
			default:
				log.Debug().Msgf("you said: %v", strconv.Quote(line))
			}
		}
	}()

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal().Err(err).Msg("")
	}
	<-idleConnsClosed
	log.Info().Msg("server gracefully shutting down")
}
