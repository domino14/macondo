package shell

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/montecarlo"

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

	SimLog = "/tmp/simlog"
)

type ShellController struct {
	l *readline.Instance

	curGameState *mechanics.XWordGame
	curGameRepr  *mechanics.GameRepr
	aiplayer     player.AIPlayer

	simmer        *montecarlo.Simmer
	simCtx        context.Context
	simCancel     context.CancelFunc
	simTicker     *time.Ticker
	simTickerDone chan bool
	simLogFile    *os.File

	curTurnNum     int
	gen            movegen.MoveGenerator
	curMode        Mode
	endgameSolver  *alphabeta.Solver
	curEndgameNode *alphabeta.GameNode
	curGenPlays    []*move.Move
}

type Mode int

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

func (sc *ShellController) loadGCG(filepath string) error {
	var err error
	// Try to parse filepath as a network path.
	if strings.HasPrefix(filepath, "xt ") {
		xtgcg := strings.Split(filepath, " ")

		if len(xtgcg) != 2 {
			return fmt.Errorf(
				"if using a cross-tables id must provide in the format xt <game_id>")
		}
		idstr := xtgcg[1]
		id, err := strconv.Atoi(idstr)
		if err != nil {
			return fmt.Errorf("badly formatted game ID")
		}
		prefix := strconv.Itoa(id / 100)
		xtpath := "https://www.cross-tables.com/annotated/selfgcg/" + prefix +
			"/anno" + idstr + ".gcg"
		resp, err := http.Get(xtpath)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		sc.curGameRepr, err = gcgio.ParseGCGFromReader(resp.Body)

	} else {
		sc.curGameRepr, err = gcgio.ParseGCG(filepath)
		if err != nil {
			return err
		}
	}
	log.Debug().Msgf("Loaded game repr; players: %v", sc.curGameRepr.Players)
	sc.curGameState = mechanics.StateFromRepr(sc.curGameRepr, "NWL18", 0)

	strategy := strategy.NewExhaustiveLeaveStrategy(sc.curGameState.Bag(),
		sc.curGameState.Gaddag().LexiconName(),
		sc.curGameState.Gaddag().GetAlphabet(), LeaveFile)

	sc.aiplayer = player.NewRawEquityPlayer(strategy)
	sc.gen = movegen.NewGordonGenerator(sc.curGameState.Gaddag(),
		sc.curGameState.Board(), sc.curGameState.Bag().LetterDistribution())

	sc.simmer = &montecarlo.Simmer{}
	sc.simmer.Init(sc.gen, sc.curGameState, sc.aiplayer)

	return nil
}

func (sc *ShellController) setToTurn(turnnum int) error {

	if sc.curGameState == nil {
		return errors.New("please load a game first with the `load` command")
	}
	err := sc.curGameState.PlayGameToTurn(sc.curGameRepr, turnnum)
	if err != nil {
		return err
	}
	log.Debug().Msgf("Set to turn %v", turnnum)
	sc.curTurnNum = turnnum
	return nil
}

func moveTableHeader() string {
	return "     Move                Leave  Score Equity\n"
}

func MoveTableRow(idx int, m *move.Move, alph *alphabet.Alphabet) string {
	return fmt.Sprintf("%3d: %-20s%-7s%-6d%-6.2f", idx+1,
		m.ShortDescription(), m.Leave().UserVisible(alph), m.Score(), m.Equity())
}

func exchangeAllowed(board *board.GameBoard, ld *alphabet.LetterDistribution) bool {
	// Instead of checking if the bag has 7 or more tiles, we check that the
	// board has 80 tiles on it or more.
	// We do this because during an interactive setup like this one, it's
	// likely that racks are not assigned to each player every turn.

	desiredNumber := ld.NumTotalTiles() - 14 - 7
	if board.TilesPlayed() > desiredNumber {
		return false
	}
	return true
}

func (sc *ShellController) genMovesAndDisplay(numPlays int) {

	alph := sc.curGameState.Alphabet()
	curRack := sc.curGameState.RackFor(sc.curGameState.PlayerOnTurn())
	opp := (sc.curGameState.PlayerOnTurn() + 1) % sc.curGameState.NumPlayers()
	oppRack := sc.curGameState.RackFor(opp)
	if oppRack.NumTiles() == 0 && sc.curGameState.Bag().TilesRemaining() <= 7 {
		log.Debug().Msg("Assigning remainder of unseen tiles to opponent...")
		oppRack = alphabet.NewRack(sc.curGameState.Alphabet())
		oppRack.Set(sc.curGameState.Bag().Peek())
	}

	canExchange := exchangeAllowed(sc.curGameState.Board(),
		sc.curGameState.Bag().LetterDistribution())
	sc.gen.GenAll(curRack, canExchange)

	// Assign equity to plays, and only show the top ones.
	sc.aiplayer.AssignEquity(sc.gen.Plays(), sc.curGameState.Board(),
		sc.curGameState.Bag(), oppRack)
	sc.curGenPlays = sc.aiplayer.TopPlays(sc.gen.Plays(), numPlays)

	sc.showMessage(moveTableHeader())
	for i, p := range sc.curGenPlays {
		sc.showMessage(MoveTableRow(i, p, alph))
	}
}

func endgameArgs(line string) (plies int, deepening, simple, disablePruning bool, err error) {
	deepening = true
	plies = 4
	simple = false
	disablePruning = false

	cmd := strings.Fields(line)
	if len(cmd) == 1 {
		if line != "endgame" {
			err = errors.New("could not understand your endgame arguments")
		}
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
		var d int
		d, err = strconv.Atoi(cmd[4])
		if err != nil {
			return
		}
		disablePruning = d == 1
	}
	if len(cmd) > 5 {
		err = errors.New("endgame only takes 5 arguments")
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

func (sc *ShellController) showMessage(msg string) {
	showMessage(msg, sc.l.Stderr())
}

func (sc *ShellController) showError(err error) {
	sc.showMessage("Error: " + err.Error())
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

func (sc *ShellController) addRack(rack string) error {
	// Set current player on turn's rack.
	playerid := sc.curTurnNum % 2
	return sc.curGameState.SetRackFor(playerid, alphabet.RackFromString(rack, sc.curGameState.Alphabet()))
}

func (sc *ShellController) addPlay(line string) error {
	cmd := strings.Fields(line)
	var playerid int
	var nick string
	var appendPlay bool
	var m *move.Move
	var cumul int
	var err error
	// Figure out whose turn it is.

	playerid = sc.curTurnNum % 2
	if sc.curTurnNum < len(sc.curGameRepr.Turns) {
		nick = sc.curGameRepr.Turns[sc.curTurnNum][0].GetNickname()
	} else if sc.curTurnNum == len(sc.curGameRepr.Turns) {
		nick = sc.curGameRepr.Players[playerid].Nickname
		appendPlay = true
	} else {
		return errors.New("unexpected turn number")
	}

	if len(cmd) == 2 && strings.HasPrefix(cmd[1], "#") {
		// Add play that was generated.
		playID, err := strconv.Atoi(cmd[1][1:])
		if err != nil {
			return err
		}

		idx := playID - 1 // since playID starts from 1
		if idx < 0 || idx > len(sc.curGenPlays)-1 {
			return errors.New("play outside range")
		}
		cumul = sc.curGameState.PointsFor(playerid) + m.Score()
		m = sc.curGenPlays[idx]
	} else if len(cmd) == 3 {
		coords := cmd[1]
		word := cmd[2]

		rack := sc.curGameState.RackFor(playerid).String()
		m, err = sc.curGameState.CreateAndScorePlacementMove(coords, word, rack)
		if err != nil {
			return err
		}
		// Handle exchange/pass later.
	} else {
		return errors.New("unrecognized arguments to `add`")
	}

	// Play the actual move on the board, draw tiles, etc.
	// Modify the game repr.
	err = sc.curGameRepr.AddTurnFromPlay(sc.curTurnNum, m, nick, cumul, appendPlay)
	if err != nil {
		return err
	}
	log.Debug().Msgf("Added turn at turn num %v", sc.curTurnNum)
	sc.setToTurn(sc.curTurnNum + 1)
	sc.showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr))

	return nil

}

func (sc *ShellController) standardModeSwitch(line string, sig chan os.Signal) error {
	var delta int

	switch {
	case strings.HasPrefix(line, "load "):
		filepath := line[5:]
		err := sc.loadGCG(filepath)

		if err != nil {
			showMessage("Error: "+err.Error(), sc.l.Stderr())
			break
		}
		sc.curTurnNum = 0
		sc.showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr))

	case line == "n" || line == "p":
		if line == "n" {
			delta = 1
		} else {
			delta = -1
		}
		err := sc.setToTurn(sc.curTurnNum + delta)
		if err != nil {
			showMessage("Error: "+err.Error(), sc.l.Stderr())
			break
		}
		sc.showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr))

	case line == "s":
		sc.showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr))

	case strings.HasPrefix(line, "turn "):
		turnnum := line[5:]
		t, err := strconv.Atoi(turnnum)
		if err != nil {
			sc.showError(err)
			break
		}
		err = sc.setToTurn(t)
		if err != nil {
			sc.showError(err)
			break
		}
		sc.showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr))

	case strings.HasPrefix(line, "rack "):
		rack := line[5:]
		err := sc.addRack(rack)
		if err != nil {
			sc.showError(err)
			break
		}
		sc.showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr))

	case strings.HasPrefix(line, "gen"):
		var numPlays int
		var err error
		if strings.TrimSpace(line) == "gen" {
			numPlays = 15
		} else {
			fc := strings.SplitN(line, " ", 2)
			if len(fc) == 1 {
				sc.showError(errors.New("wrong format for `gen` command"))
				break
			}
			numPlays, err = strconv.Atoi(fc[1])
			if err != nil {
				sc.showError(err)
				break
			}
		}
		if sc.curGameState != nil {
			sc.genMovesAndDisplay(numPlays)
		}

	case strings.HasPrefix(line, "sim"):
		var plies int
		var err error
		if sc.simmer == nil {
			sc.showError(errors.New("simmer stop"))
			break
		}
		if strings.TrimSpace(line) == "sim" {
			plies = 2
		} else {
			fc := strings.SplitN(line, " ", 2)
			if len(fc) == 1 {
				sc.showError(errors.New("wrong format for `gen` command"))
			}
			if fc[1] == "log" {
				sc.simLogFile, err = os.Create(SimLog)
				if err != nil {
					sc.showError(err)
					break
				}
				sc.simmer.SetLogStream(sc.simLogFile)
				sc.showMessage("sim will log to " + SimLog)
				break
			} else if fc[1] == "stop" {
				if !sc.simmer.IsSimming() {
					sc.showError(errors.New("no running sim to stop"))
					break
				}
				sc.simTicker.Stop()
				sc.simTickerDone <- true
				sc.simCancel()
				if sc.simLogFile != nil {
					err := sc.simLogFile.Close()
					if err != nil {
						sc.showError(err)
						break
					}
				}
				break
			} else if fc[1] == "details" {
				sc.showMessage(sc.simmer.ScoreDetails())
				break
			} else if fc[1] == "show" {
				sc.showMessage(sc.simmer.EquityStats())
				break
			} else {
				plies, err = strconv.Atoi(fc[1])
				if err != nil {
					sc.showError(err)
					break
				}
			}
		}
		if len(sc.curGenPlays) == 0 {
			sc.showError(errors.New("please generate some plays first"))
			break
		}
		if sc.simmer.IsSimming() {
			sc.showError(errors.New("simming already, please do a `sim stop` first"))
			break
		}

		if sc.curGameState != nil {
			sc.simCtx, sc.simCancel = context.WithCancel(context.Background())
			sc.simTicker = time.NewTicker(15 * time.Second)
			sc.simTickerDone = make(chan bool)

			go sc.simmer.Simulate(sc.simCtx, sc.curGenPlays, plies)

			go func() {
				for {
					select {
					case <-sc.simTickerDone:
						return
					case <-sc.simTicker.C:
						log.Info().Msgf("Simmer is at %v iterations...",
							sc.simmer.Iterations())
					}
				}
			}()

			sc.showMessage("Simulation started. Please do `sim show` and `sim details` to see more info")
		}

	case strings.HasPrefix(line, "add "):
		err := sc.addPlay(line)
		if err != nil {
			sc.showError(err)
		}
	case strings.HasPrefix(line, "endgame"):
		if sc.curGameState == nil {
			showMessage("please load a game first with the `load` command", sc.l.Stderr())
			break
		}
		plies, deepening, simpleEval, disablePruning, err := endgameArgs(line)
		if err != nil {
			showMessage("Error: "+err.Error(), sc.l.Stderr())
			break
		}
		showMessage(fmt.Sprintf("plies %v, deepening %v, simpleEval %v, pruningDisabled %v",
			plies, deepening, simpleEval, disablePruning), sc.l.Stderr())

		sc.curGameState.SetStateStackLength(plies)

		// clear out the last value of this endgame node; gc should
		// delete the tree.
		sc.curEndgameNode = nil
		sc.endgameSolver = new(alphabeta.Solver)
		sc.endgameSolver.Init(sc.gen, sc.curGameState)
		sc.endgameSolver.SetIterativeDeepening(deepening)
		sc.endgameSolver.SetSimpleEvaluator(simpleEval)
		sc.endgameSolver.SetPruningDisabled(disablePruning)

		showMessage(sc.curGameState.ToDisplayText(sc.curGameRepr), sc.l.Stderr())

		sc.endgameSolver.Solve(plies)

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
