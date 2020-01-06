package shell

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

type Mode int
type ShellController struct {
	l *readline.Instance

	curGameState *mechanics.XWordGame
	curGameRepr  *mechanics.GameRepr
	curTurnNum   int
	gen          movegen.MoveGenerator
	endgameGen   movegen.MoveGenerator
	curMode      Mode
}

const (
	StandardMode Mode = iota
	EndgameDebugMode
	InvalidMode
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

func NewShellController() *ShellController {
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
	return &ShellController{l: l}
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

func modeFromStr(mode string) (Mode, error) {
	mode = strings.TrimSpace(mode)
	switch mode {
	case "standard":
		return StandardMode, nil
	case "endgamedebug":
		return EndgameDebugMode, nil
	}
	return InvalidMode, errors.New("mode " + mode + " is not a valid choice")
}

func (sc *ShellController) modeSelector(line string) {
	mode := strings.SplitN(line, " ", 2)
	if len(mode) != 2 {
		showMessage("Error: please provide a valid mode", sc.l.Stderr())
		return
	}
	m, err := modeFromStr(mode[1])
	if err != nil {
		showMessage("Error: "+err.Error(), sc.l.Stderr())
		return
	}
	showMessage("Setting current mode to "+mode[1], sc.l.Stderr())
	sc.curMode = m
}

func (sc *ShellController) endgameDebugModeSwitch(line string, sig chan os.Signal) error {
	switch {
	case strings.HasPrefix(line, "mode"):
		sc.modeSelector(line)

	case strings.HasPrefix(line, "help"):
		if strings.TrimSpace(line) == "help" {
			usage(sc.l.Stderr(), "endgamedebug")
		} else {
			showMessage("No additional info is available for this mode", sc.l.Stderr())
		}

	case line == "l":
		// List the current level of nodes
		showMessage("Listing nodes", sc.l.Stderr())

	case line == "u":
		// List the current level of nodes
		showMessage("Go up a level", sc.l.Stderr())

	case line == "i":
		// List the current level of nodes
		showMessage("Info about the current node", sc.l.Stderr())

	case strings.HasPrefix(line, "s"):
		nodeID, err := strconv.Atoi(line[:1])
		if err != nil {
			showMessage("Error: "+err.Error(), sc.l.Stderr())
			return nil
		}
		showMessage(fmt.Sprintf("Stepping into node %d", nodeID), sc.l.Stderr())

	default:
		log.Debug().Msgf("you said: %v", strconv.Quote(line))
	}
	return nil
}

func (sc *ShellController) standardModeSwitch(line string, sig chan os.Signal) error {
	var delta int
	var err error

	switch {
	case strings.HasPrefix(line, "load "):
		filepath := line[5:]
		sc.curGameState, sc.curGameRepr, sc.gen, err = loadGCG(filepath)
		if err != nil {
			showMessage("Error: "+err.Error(), sc.l.Stderr())
			break
		}
		sc.curTurnNum = 0
		showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr), sc.l.Stderr())

	case line == "n" || line == "p":
		if line == "n" {
			delta = 1
		} else {
			delta = -1
		}
		err := setToTurn(sc.curTurnNum+delta, sc.curGameState, sc.curGameRepr)
		if err != nil {
			showMessage("Error: "+err.Error(), sc.l.Stderr())
			break
		}
		sc.curTurnNum += delta
		showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr), sc.l.Stderr())

	case strings.HasPrefix(line, "turn "):
		turnnum := line[5:]
		t, err := strconv.Atoi(turnnum)
		if err != nil {
			showMessage(err.Error(), sc.l.Stderr())
			break
		}
		err = setToTurn(t, sc.curGameState, sc.curGameRepr)
		if err != nil {
			showMessage("Error: "+err.Error(), sc.l.Stderr())
			break
		}
		sc.curTurnNum = t
		showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr), sc.l.Stderr())

	case strings.HasPrefix(line, "gen"):
		var numPlays int
		var err error
		if strings.TrimSpace(line) == "gen" {
			numPlays = 15
		} else {
			fc := strings.SplitN(line, " ", 2)
			numPlays, err = strconv.Atoi(fc[1])
			if err != nil {
				showMessage("Error: "+err.Error(), sc.l.Stderr())
				break
			}
		}
		if sc.curGameState != nil {
			genMovesAndDisplay(sc.curGameState, sc.gen, numPlays, sc.l.Stderr())
		}

	case strings.HasPrefix(line, "endgame"):
		if sc.curGameState == nil {
			showMessage("please load a game first with the `load` command", sc.l.Stderr())
			break
		}
		plies, deepening, simpleEval, err := endgameArgs(line)
		if err != nil {
			showMessage("Error: "+err.Error(), sc.l.Stderr())
			break
		}
		showMessage(fmt.Sprintf("plies %v, deepening %v, simpleEval %v",
			plies, deepening, simpleEval), sc.l.Stderr())

		sc.curGameState.SetStateStackLength(plies)
		sc.endgameGen = movegen.NewGordonGenerator(
			sc.curGameState, &strategy.NoLeaveStrategy{})

		s := new(alphabeta.Solver)
		s.Init(sc.endgameGen, sc.curGameState)
		s.SetIterativeDeepening(deepening)
		s.SetSimpleEvaluator(simpleEval)

		showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr), sc.l.Stderr())

		s.Solve(plies)

	case line == "bye" || line == "exit":
		sig <- syscall.SIGINT
		return errors.New("sending quit signal")
	case strings.HasPrefix(line, "help"):
		if strings.TrimSpace(line) == "help" {
			usage(sc.l.Stderr(), "standard")
		} else {
			helptopic := strings.SplitN(line, " ", 2)
			usageTopic(sc.l.Stderr(), helptopic[1])
		}
	case strings.HasPrefix(line, "mode"):
		sc.modeSelector(line)
	default:
		if strings.TrimSpace(line) != "" {
			log.Debug().Msgf("you said: %v", strconv.Quote(line))
		}
	}
	return nil
}

func (sc *ShellController) Loop(sig chan os.Signal) {

	defer sc.l.Close()

	for {
		line, err := sc.l.Readline()
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

		if sc.curMode == StandardMode {
			err := sc.standardModeSwitch(line, sig)
			if err != nil {
				log.Error().Err(err).Msg("")
				break
			}
		} else if sc.curMode == EndgameDebugMode {
			err := sc.endgameDebugModeSwitch(line, sig)
			if err != nil {
				log.Error().Err(err).Msg("")
				break
			}
		}

	}
	log.Debug().Msgf("Exiting readline loop...")
}
