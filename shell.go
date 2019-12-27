package main

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"syscall"

	"github.com/chzyer/readline"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/endgame/alphabeta"
	"github.com/domino14/macondo/gcgio"
	"github.com/domino14/macondo/mechanics"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
)

const (
	LeaveFile = "leave_values_112719.idx.gz"
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

func loadGCG(filepath string) (*mechanics.XWordGame, *mechanics.GameRepr, movegen.MoveGenerator, error) {
	var curGameRepr *mechanics.GameRepr
	var err error
	// Try to parse filepath as a network path.
	if strings.HasPrefix(filepath, "xt ") {
		xtgcg := strings.Split(filepath, " ")

		if len(xtgcg) != 2 {
			return nil, nil, nil, fmt.Errorf(
				"if using a cross-tables id must provide in the format xt <game_id>")
		}
		idstr := xtgcg[1]
		id, err := strconv.Atoi(idstr)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("badly formatted game ID")
		}
		prefix := strconv.Itoa(id / 100)
		xtpath := "https://www.cross-tables.com/annotated/selfgcg/" + prefix +
			"/anno" + idstr + ".gcg"
		resp, err := http.Get(xtpath)
		if err != nil {
			return nil, nil, nil, err
		}
		defer resp.Body.Close()

		curGameRepr, err = gcgio.ParseGCGFromReader(resp.Body)

	} else {
		curGameRepr, err = gcgio.ParseGCG(filepath)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	log.Debug().Msgf("Loaded game repr; players: %v", curGameRepr.Players)
	curGameState := mechanics.StateFromRepr(curGameRepr, "NWL18", 0)

	strategy := strategy.NewExhaustiveLeaveStrategy(curGameState.Bag(),
		curGameState.Gaddag().LexiconName(),
		curGameState.Gaddag().GetAlphabet(), LeaveFile)

	generator := movegen.NewGordonGenerator(curGameState, strategy)

	return curGameState, curGameRepr, generator, nil
}

func setToTurn(turnnum int, curGameState *mechanics.XWordGame,
	curGameRepr *mechanics.GameRepr) error {

	if curGameState == nil {
		return errors.New("please load a game first with the `load` command")
	}
	err := curGameState.PlayGameToTurn(curGameRepr, turnnum)
	if err != nil {
		return err
	}
	return nil
}

func moveTableHeader() string {
	return "Move                Leave  Score Equity\n"
}

func MoveTableRow(m *move.Move, alph *alphabet.Alphabet) string {
	return fmt.Sprintf("%-20s%-7s%-6d%-6.2f\n",
		m.ShortDescription(), m.Leave().UserVisible(alph), m.Score(), m.Equity())
}

func genMovesAndDisplay(curGameState *mechanics.XWordGame, gen movegen.MoveGenerator,
	numPlays int, w io.Writer) {

	alph := curGameState.Alphabet()
	curRack := curGameState.RackFor(curGameState.PlayerOnTurn())
	gen.GenAll(curRack)
	var plays []*move.Move
	if len(gen.Plays()) > numPlays {
		plays = gen.Plays()[:numPlays]
	} else {
		plays = gen.Plays()
	}
	io.WriteString(w, moveTableHeader())
	for _, p := range plays {
		io.WriteString(w, MoveTableRow(p, alph))
	}
}

func endgameArgs(line string) (plies int, deepening bool, simple bool, err error) {
	deepening = true
	plies = 4
	simple = false

	cmd := strings.Fields(line)
	if len(cmd) == 1 {
		return
	}
	if len(cmd) > 1 {
		plies, err = strconv.Atoi(cmd[1])
		if err != nil {
			return
		}
	}
	if len(cmd) > 2 {
		var id int
		id, err = strconv.Atoi(cmd[2])
		if err != nil {
			return
		}
		deepening = id == 1
	}
	if len(cmd) > 3 {
		var sp int
		sp, err = strconv.Atoi(cmd[3])
		if err != nil {
			return
		}
		simple = sp == 1
	}
	if len(cmd) > 4 {
		err = errors.New("endgame only takes 4 arguments")
		return
	}
	return
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
	var gen movegen.MoveGenerator
	var endgameGen movegen.MoveGenerator

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
			curGameState, curGameRepr, gen, err = loadGCG(filepath)
			if err != nil {
				showMessage("Error: "+err.Error(), l.Stderr())
				break
			}
			curTurnNum = 0
			showMessage(curGameState.ToDisplayText(curGameRepr), l.Stderr())

		case line == "n" || line == "p":
			if line == "n" {
				delta = 1
			} else {
				delta = -1
			}
			err := setToTurn(curTurnNum+delta, curGameState, curGameRepr)
			if err != nil {
				showMessage("Error: "+err.Error(), l.Stderr())
				break
			}
			curTurnNum += delta
			showMessage(curGameState.ToDisplayText(curGameRepr), l.Stderr())

		case strings.HasPrefix(line, "turn "):
			turnnum := line[5:]
			t, err := strconv.Atoi(turnnum)
			if err != nil {
				showMessage(err.Error(), l.Stderr())
				break
			}
			err = setToTurn(t, curGameState, curGameRepr)
			if err != nil {
				showMessage("Error: "+err.Error(), l.Stderr())
				break
			}
			curTurnNum = t
			showMessage(curGameState.ToDisplayText(curGameRepr), l.Stderr())

		case strings.HasPrefix(line, "gen"):
			var numPlays int
			var err error
			if strings.TrimSpace(line) == "gen" {
				numPlays = 15
			} else {
				fc := strings.SplitN(line, " ", 2)
				numPlays, err = strconv.Atoi(fc[1])
				if err != nil {
					showMessage("Error: "+err.Error(), l.Stderr())
					break
				}
			}
			genMovesAndDisplay(curGameState, gen, numPlays, l.Stderr())

		case strings.HasPrefix(line, "endgame"):
			if curGameState == nil {
				showMessage("please load a game first with the `load` command", l.Stderr())
				break
			}
			plies, deepening, simpleEval, err := endgameArgs(line)
			if err != nil {
				showMessage("Error: "+err.Error(), l.Stderr())
				break
			}
			showMessage(fmt.Sprintf("plies %v, deepening %v, simpleEval %v",
				plies, deepening, simpleEval), l.Stderr())

			curGameState.SetStateStackLength(plies)
			endgameGen = movegen.NewGordonGenerator(
				curGameState, &strategy.NoLeaveStrategy{})

			s := new(alphabeta.Solver)
			s.Init(endgameGen, curGameState)
			s.SetIterativeDeepening(deepening)
			s.SetSimpleEvaluator(simpleEval)

			showMessage(curGameState.ToDisplayText(curGameRepr), l.Stderr())

			s.Solve(plies)

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
