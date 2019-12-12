package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"syscall"

	"github.com/chzyer/readline"
	"github.com/domino14/macondo/gcgio"
	"github.com/domino14/macondo/mechanics"
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

func showMessage(msg string, w io.Writer) {
	io.WriteString(w, msg)
	io.WriteString(w, "\n")
}

func loadGCG(filepath string) (*mechanics.XWordGame, *mechanics.GameRepr, error) {
	var curGameRepr *mechanics.GameRepr
	var err error
	// Try to parse filepath as a network path.
	if strings.HasPrefix(filepath, "xt ") {
		xtgcg := strings.Split(filepath, " ")

		if len(xtgcg) != 2 {
			return nil, nil, fmt.Errorf(
				"if using a cross-tables id must provide in the format xt <game_id>")
		}
		idstr := xtgcg[1]
		id, err := strconv.Atoi(idstr)
		if err != nil {
			return nil, nil, fmt.Errorf("badly formatted game ID")
		}
		prefix := strconv.Itoa(id / 100)
		xtpath := "https://www.cross-tables.com/annotated/selfgcg/" + prefix +
			"/anno" + idstr + ".gcg"
		resp, err := http.Get(xtpath)
		if err != nil {
			return nil, nil, err
		}
		defer resp.Body.Close()

		curGameRepr, err = gcgio.ParseGCGFromReader(resp.Body)

	} else {
		curGameRepr, err = gcgio.ParseGCG(filepath)
		if err != nil {
			return nil, nil, err
		}
	}
	log.Debug().Msgf("Loaded game repr; players: %v", curGameRepr.Players)
	curGameState := mechanics.StateFromRepr(curGameRepr, "NWL18", 0)
	return curGameState, curGameRepr, nil
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
	var curGameState *mechanics.XWordGame
	var curGameRepr *mechanics.GameRepr
	var curTurnNum int
	var delta int
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
			curGameState, curGameRepr, err = loadGCG(filepath)
			if err != nil {
				showMessage("Error: "+err.Error(), l.Stderr())
				break
			}
			curTurnNum = 0
			showMessage(curGameState.ToDisplayText(), l.Stderr())

		case line == "n" || line == "p":
			if curGameState == nil {
				showMessage("Please load a game first with the `load` command",
					l.Stderr())
				break
			}
			if line == "n" {
				delta = 1
			} else {
				delta = -1
			}
			err := curGameState.PlayGameToTurn(curGameRepr, curTurnNum+delta)
			if err != nil {
				showMessage(err.Error(), l.Stderr())
				break
			}
			curTurnNum += delta
			showMessage(curGameState.ToDisplayText(), l.Stderr())

		case line == "bye" || line == "exit":
			sig <- syscall.SIGINT
			break readlineLoop
		case strings.HasPrefix(line, "help"):
			if strings.TrimSpace(line) == "help" {
				usage(l.Stderr())
			} else {
				helptopic := strings.SplitN(line, " ", 2)
				usageTopic(l.Stderr(), helptopic[1])
			}
		case line == "":
		default:
			log.Debug().Msgf("you said: %v", strconv.Quote(line))
		}
	}
	log.Debug().Msgf("Exiting readline loop...")
}
