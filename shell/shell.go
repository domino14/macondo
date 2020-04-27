package shell

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/chzyer/readline"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/alphabeta"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	pb "github.com/domino14/macondo/rpc/api/proto"
	"github.com/domino14/macondo/strategy"
)

const (
	SimLog = "/tmp/simlog"
)

type ShellController struct {
	l      *readline.Instance
	config *config.Config

	game     *game.Game
	aiplayer player.AIPlayer

	simmer        *montecarlo.Simmer
	simCtx        context.Context
	simCancel     context.CancelFunc
	simTicker     *time.Ticker
	simTickerDone chan bool
	simLogFile    *os.File

	gameRunnerCtx     context.Context
	gameRunnerCancel  context.CancelFunc
	gameRunnerRunning bool
	gameRunnerTicker  *time.Ticker

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

func NewShellController(cfg *config.Config) *ShellController {
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
	return &ShellController{l: l, config: cfg}
}

func (sc *ShellController) initGameDataStructures() {
	strategy := strategy.NewExhaustiveLeaveStrategy(
		sc.game.Gaddag().LexiconName(),
		sc.game.Gaddag().GetAlphabet(),
		sc.config.StrategyParamsPath)

	sc.aiplayer = player.NewRawEquityPlayer(strategy)
	sc.gen = movegen.NewGordonGenerator(sc.game.Gaddag(),
		sc.game.Board(), sc.game.Bag().LetterDistribution())

	sc.simmer = &montecarlo.Simmer{}
	sc.simmer.Init(sc.game, sc.aiplayer)
}

func (sc *ShellController) loadGCG(filepath string) error {
	var err error
	var history *pb.GameHistory
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

		history, err = gcgio.ParseGCGFromReader(resp.Body)
		if err != nil {
			return err
		}

	} else {
		history, err = gcgio.ParseGCG(filepath)
		if err != nil {
			return err
		}
	}
	log.Debug().Msgf("Loaded game repr; players: %v", history.Players)
	rules, err := game.NewGameRules(sc.config, board.CrosswordGameBoard,
		history.Lexicon, sc.config.DefaultLetterDistribution)
	if err != nil {
		return err
	}
	sc.game, err = game.NewFromHistory(history, rules, 0)
	if err != nil {
		return err
	}
	sc.initGameDataStructures()

	return nil
}

func (sc *ShellController) setToTurn(turnnum int) error {

	if sc.game == nil {
		return errors.New("please load a game first with the `load` command")
	}
	err := sc.game.PlayToTurn(turnnum)
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

// XXX: THIS FUNCTION MIGHT NOT BE NEEDED ANYMORE
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

func (sc *ShellController) printEndgameSequence(moves []*move.Move) {
	sc.showMessage("Best sequence:")
	for idx, move := range moves {
		sc.showMessage(fmt.Sprintf("%d) %v", idx+1, move.ShortDescription()))
	}
}

func (sc *ShellController) genMovesAndDisplay(numPlays int) {

	alph := sc.game.Alphabet()
	curRack := sc.game.RackFor(sc.game.PlayerOnTurn())
	opp := (sc.game.PlayerOnTurn() + 1) % sc.game.NumPlayers()
	oppRack := sc.game.RackFor(opp)

	// XXX: MIGHT BE ABLE TO REMOVE THIS:
	if oppRack.NumTiles() == 0 && sc.game.Bag().TilesRemaining() <= 7 {
		log.Debug().Msg("Assigning remainder of unseen tiles to opponent...")
		oppRack = alphabet.NewRack(sc.game.Alphabet())
		oppRack.Set(sc.game.Bag().Peek())
	}

	canExchange := exchangeAllowed(sc.game.Board(),
		sc.game.Bag().LetterDistribution())
	sc.gen.GenAll(curRack, canExchange)

	// Assign equity to plays, and only show the top ones.
	sc.aiplayer.AssignEquity(sc.gen.Plays(), sc.game.Board(),
		sc.game.Bag(), oppRack)
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
	if sc.game == nil {
		return errors.New("please start a game first")
	}
	return sc.game.SetRackFor(playerid, alphabet.RackFromString(rack, sc.game.Alphabet()))
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

	if len(cmd) == 2 {
		if strings.HasPrefix(cmd[1], "#") {
			// Add play that was generated.
			playID, err := strconv.Atoi(cmd[1][1:])
			if err != nil {
				return err
			}

			idx := playID - 1 // since playID starts from 1
			if idx < 0 || idx > len(sc.curGenPlays)-1 {
				return errors.New("play outside range")
			}
			m = sc.curGenPlays[idx]
			cumul = sc.game.PointsFor(playerid) + m.Score()
		} else if cmd[1] == "pass" {
			rack := sc.game.RackFor(playerid)
			m = move.NewPassMove(rack.TilesOn(), sc.game.Alphabet())
			cumul = sc.game.PointsFor(playerid)
		} else {
			return errors.New("unrecognized arguments to `add`")
		}

	} else if len(cmd) == 3 {
		coords := cmd[1]
		word := cmd[2]

		if coords == "exchange" {

			rack := sc.game.RackFor(playerid)
			tiles, err := alphabet.ToMachineWord(word, sc.game.Alphabet())
			if err != nil {
				return err
			}
			leaveMW, err := game.Leave(rack.TilesOn(), tiles)
			if err != nil {
				return err
			}

			m = move.NewExchangeMove(tiles, leaveMW, sc.game.Alphabet())
			cumul = sc.game.PointsFor(playerid)
		} else {

			rack := sc.game.RackFor(playerid).String()
			m, err = sc.game.CreateAndScorePlacementMove(coords, word, rack)
			if err != nil {
				return err
			}
		}

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
	sc.showMessage(sc.game.ToDisplayText(sc.curGameRepr))

	return nil

}

func (sc *ShellController) standardModeSwitch(line string, sig chan os.Signal) error {
	var delta int

	switch {
	case line == "new":

		rules, err := game.NewGameRules(sc.config, board.CrosswordGameBoard,
			sc.config.DefaultLexicon, sc.config.DefaultLetterDistribution)
		if err != nil {
			sc.showError(err)
			break
		}

		players := []*pb.PlayerInfo{
			&pb.PlayerInfo{Nickname: "player1", RealName: "Player 1", Number: 1},
			&pb.PlayerInfo{Nickname: "player2", RealName: "Player 2", Number: 2},
		}

		sc.game, err = game.NewGame(rules, players)
		if err != nil {
			sc.showError(err)
			break
		}
		sc.game.StartGame()
		sc.initGameDataStructures()
		sc.showMessage(sc.game.ToDisplayText())

	case strings.HasPrefix(line, "load "):
		filepath := line[5:]
		err := sc.loadGCG(filepath)

		if err != nil {
			sc.showError(err)
			break
		}
		sc.curTurnNum = 0
		sc.showMessage(sc.game.ToDisplayText())

	case line == "n" || line == "p":
		if line == "n" {
			delta = 1
		} else {
			delta = -1
		}
		err := sc.setToTurn(sc.curTurnNum + delta)
		if err != nil {
			sc.showError(err)
			break
		}
		sc.showMessage(sc.game.ToDisplayText())

	case line == "s":
		sc.showMessage(sc.game.ToDisplayText())

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
		sc.showMessage(sc.game.ToDisplayText())

	case strings.HasPrefix(line, "rack "):
		rack := line[5:]
		err := sc.addRack(strings.ToUpper(rack))
		if err != nil {
			sc.showError(err)
			break
		}
		sc.showMessage(sc.game.ToDisplayText())

		// XXX: TEST AND FIX ME:

	// case strings.HasPrefix(line, "setlex "):
	// 	lex := line[7:]
	// 	gdFilename := filepath.Join(
	// 		sc.config.LexiconPath, "gaddag", lex+".gaddag")
	// 	gd, err := gaddag.LoadGaddag(gdFilename)
	// 	if err != nil {
	// 		sc.showError(err)
	// 		break
	// 	}
	// 	sc.game.SetGaddag(gd)
	// 	sc.showMessage("Lexicon set to " + lex)

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
		if sc.game != nil {
			sc.genMovesAndDisplay(numPlays)
		}

	case strings.HasPrefix(line, "autoplay"):
		var logfile string
		if strings.TrimSpace(line) == "autoplay" {
			logfile = "/tmp/autoplay.txt"
		} else {
			fields := strings.Split(line, " ")
			if len(fields) == 1 {
				sc.showError(errors.New("wrong format for `autoplay` command"))
				break
			}
			if fields[1] == "stop" {
				if !sc.gameRunnerRunning {
					sc.showError(errors.New("automatic game runner is not running"))
					break
				} else {
					sc.gameRunnerCancel()
					sc.gameRunnerRunning = false
					break
				}
			} else {
				// It's a filename
				logfile = fields[1]
			}
		}
		if sc.gameRunnerRunning {
			sc.showError(errors.New("please stop automatic game runner before running another one"))
			break
		}
		// XXX Refactor this

		gdFilename := filepath.Join(sc.config.LexiconPath, "gaddag", sc.config.DefaultLexicon+".gaddag")

		gd, err := gaddag.LoadGaddag(gdFilename)
		if err != nil {
			sc.showError(err)
			break
		}
		sc.showMessage("automatic game runner will log to " + logfile)
		sc.gameRunnerCtx, sc.gameRunnerCancel = context.WithCancel(context.Background())
		automatic.StartCompVCompStaticGames(sc.gameRunnerCtx, sc.config, gd, 1e9, runtime.NumCPU(), logfile)
		sc.gameRunnerRunning = true
		sc.showMessage("Started automatic game runner...")

	case strings.HasPrefix(line, "sim"):
		var plies, threads int
		var err error
		if sc.simmer == nil {
			sc.showError(errors.New("load a game or something"))
			break
		}
		if strings.TrimSpace(line) == "sim" {
			plies = 2
		} else {
			fc := strings.Split(line, " ")
			if len(fc) == 1 {
				sc.showError(errors.New("wrong format for `sim` command"))
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
			} else if len(fc) == 2 || len(fc) == 3 {
				if len(fc) == 2 {
					plies, err = strconv.Atoi(fc[1])
					if err != nil {
						sc.showError(err)
						break
					}
				}
				if len(fc) == 3 {
					// The second number is the number of sim threads
					threads, err = strconv.Atoi(fc[2])
					if err != nil {
						sc.showError(err)
						break
					}
					sc.simmer.SetThreads(threads)
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

		if sc.game != nil {
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
		if sc.game == nil {
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

		sc.game.SetStateStackLength(plies)

		// clear out the last value of this endgame node; gc should
		// delete the tree.
		sc.curEndgameNode = nil
		sc.endgameSolver = new(alphabeta.Solver)
		err = sc.endgameSolver.Init(sc.gen, sc.game)
		if err != nil {
			sc.showError(err)
			break
		}
		sc.endgameSolver.SetIterativeDeepening(deepening)
		sc.endgameSolver.SetSimpleEvaluator(simpleEval)
		sc.endgameSolver.SetPruningDisabled(disablePruning)

		showMessage(sc.game.ToDisplayText(), sc.l.Stderr())

		val, seq := sc.endgameSolver.Solve(plies)

		sc.showMessage(fmt.Sprintf("Best sequence has a spread difference of %v", val))
		sc.printEndgameSequence(seq)

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
