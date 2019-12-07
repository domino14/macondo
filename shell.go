package main

import (
	"io"
	"os"
	"strconv"
	"strings"
	"syscall"

	"github.com/chzyer/readline"
	"github.com/domino14/macondo/gcgio"
	"github.com/rs/zerolog/log"
)

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
	io.WriteString(w, "\tload <path/to/gcg> - load a .gcg file\n")
	io.WriteString(w, "\tsetlex <lexicon> - set a lexicon (NWL18, CSW19, and maybe others)\n")
	io.WriteString(w, "\tn - next play\n")
	io.WriteString(w, "\tb - previous play\n")
	io.WriteString(w, "\tturn <n> - go to turn <n>\n")
	io.WriteString(w, "\tgen [n] - generate n plays and sort by equity; n defaults to 15\n")
	io.WriteString(w, "\tadd <play> - add a play, that looks like coords play (e.g. 10J FOO)\n")
	io.WriteString(w, "\tsim [plies] - start simulation, default to two-ply\n")
	io.WriteString(w, "\tendgame [p<maxplies>] [idoff] [sev] - run endgame, search to maxplies (4 is default)\n")
	io.WriteString(w, "\t  idoff - turn off iterative deepening,\n")
	io.WriteString(w, "\t  sev - use simple eval func (faster but less accurate)\n")
	io.WriteString(w, "\tsolvepeg [params...] - solve pre-endgame only usable if there are 1 or 2 tiles in the bag\n")
	io.WriteString(w, "\t  takes in same parameters as endgame\n")
}

func showMessage(msg string, w io.Writer) {
	io.WriteString(w, msg)
	io.WriteString(w, "\n")
}

func shellLoop(sig chan os.Signal) {
	l, err := readline.NewEx(&readline.Config{
		Prompt:          "\033[31mmacondo>\033[0m ",
		HistoryFile:     "/tmp/readline.tmp",
		EOFPrompt:       "exit",
		InterruptPrompt: "^C",

		HistorySearchFold:   true,
		FuncFilterInputRune: filterInput,
	})

	if err != nil {
		panic(err)
	}
	defer l.Close()

readlineLoop:
	for {
		line, err := l.Readline()
		if err == readline.ErrInterrupt {
			if len(line) == 0 {
				sig <- syscall.SIGINT
				break
			} else {
				continue
			}
		} else if err == io.EOF {
			sig <- syscall.SIGINT
			break
		}
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "load "):
			filepath := line[5:]
			game, err := gcgio.ParseGCG(filepath)
			if err != nil {
				showMessage(err.Error(), l.Stderr())
				break
			}
			log.Debug().Msgf("Loaded game repr; players: %v", game.Players)

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
	log.Debug().Msgf("Exiting readline loop...")
}
