package shell

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"sort"
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
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
)

const (
	SimLog = "/tmp/simlog"
)

var (
	errNoData            = errors.New("no data in this line")
	errWrongOptionSyntax = errors.New("wrong format; all options need arguments")
)

type ShellController struct {
	l        *readline.Instance
	config   *config.Config
	execPath string

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
	curPlayList    []*move.Move
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

func NewShellController(cfg *config.Config, execPath string) *ShellController {
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
	return &ShellController{l: l, config: cfg, execPath: execPath}
}

func (sc *ShellController) initGameDataStructures() error {
	strategy, err := strategy.NewExhaustiveLeaveStrategy(
		sc.game.Gaddag().LexiconName(),
		sc.game.Gaddag().GetAlphabet(),
		sc.config.StrategyParamsPath, strategy.LeaveFilename)
	if err != nil {
		return err
	}

	sc.aiplayer = player.NewRawEquityPlayer(strategy)
	sc.gen = movegen.NewGordonGenerator(sc.game.Gaddag(),
		sc.game.Board(), sc.game.Bag().LetterDistribution())

	sc.simmer = &montecarlo.Simmer{}
	sc.simmer.Init(sc.game, sc.aiplayer)
	return nil
}

func (sc *ShellController) loadGCG(args []string) error {
	var err error
	var history *pb.GameHistory
	// Try to parse filepath as a network path.
	if args[0] == "xt" {
		if len(args) < 2 {
			return errors.New("need to provide a cross-tables game id")
		}
		idstr := args[1]
		id, err := strconv.Atoi(idstr)
		if err != nil {
			return errors.New("badly formatted game ID")
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
		history, err = gcgio.ParseGCG(args[0])
		if err != nil {
			return err
		}
	}
	log.Debug().Msgf("Loaded game repr; players: %v", history.Players)
	lexicon := history.Lexicon
	if lexicon == "" {
		lexicon = sc.config.DefaultLexicon
		log.Info().Msgf("gcg file had no lexicon, so using default lexicon %v",
			lexicon)
	}
	rules, err := game.NewGameRules(sc.config, board.CrosswordGameBoard,
		lexicon, sc.config.DefaultLetterDistribution)
	if err != nil {
		return err
	}
	sc.game, err = game.NewFromHistory(history, rules, 0)
	if err != nil {
		return err
	}
	sc.game.SetBackupMode(game.InteractiveGameplayMode)
	// Set challenge rule to double by default. This can be overridden.
	sc.game.SetChallengeRule(pb.ChallengeRule_DOUBLE)

	return sc.initGameDataStructures()

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
	sc.curPlayList = nil
	sc.simmer.Reset()
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

func (sc *ShellController) printEndgameSequence(moves []*move.Move) {
	sc.showMessage("Best sequence:")
	for idx, move := range moves {
		sc.showMessage(fmt.Sprintf("%d) %v", idx+1, move.ShortDescription()))
	}
}

func (sc *ShellController) genMovesAndDisplay(numPlays int) {

	curRack := sc.game.RackFor(sc.game.PlayerOnTurn())
	opp := (sc.game.PlayerOnTurn() + 1) % sc.game.NumPlayers()
	oppRack := sc.game.RackFor(opp)

	sc.gen.GenAll(curRack, sc.game.Bag().TilesRemaining() >= 7)

	// Assign equity to plays, and only show the top ones.
	sc.aiplayer.AssignEquity(sc.gen.Plays(), sc.game.Board(),
		sc.game.Bag(), oppRack)
	sc.curPlayList = sc.aiplayer.TopPlays(sc.gen.Plays(), numPlays)
	sc.displayMoveList()
}

func (sc *ShellController) displayMoveList() {
	sc.showMessage(moveTableHeader())
	for i, p := range sc.curPlayList {
		sc.showMessage(MoveTableRow(i, p, sc.game.Alphabet()))
	}
}

func endgameArgs(args []string) (plies int, deepening, simple, disablePruning bool, err error) {
	deepening = true
	plies = 4
	simple = false
	disablePruning = false

	if len(args) > 0 {
		plies, err = strconv.Atoi(args[0])
		if err != nil {
			return
		}
	}
	if len(args) > 1 {
		var id int
		id, err = strconv.Atoi(args[1])
		if err != nil {
			return
		}
		deepening = id == 1
	}
	if len(args) > 2 {
		var sp int
		sp, err = strconv.Atoi(args[2])
		if err != nil {
			return
		}
		simple = sp == 1
	}
	if len(args) > 3 {
		var d int
		d, err = strconv.Atoi(args[3])
		if err != nil {
			return
		}
		disablePruning = d == 1
	}
	if len(args) > 4 {
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

func (sc *ShellController) modeSelector(mode string) {
	m, err := modeFromStr(mode)
	if err != nil {
		sc.showError(err)
		return
	}
	sc.showMessage("Setting current mode to " + mode)
	sc.curMode = m
}

func (sc *ShellController) addRack(rack string) error {
	// Set current player on turn's rack.
	if sc.game == nil {
		return errors.New("please start a game first")
	}
	return sc.game.SetRackFor(sc.game.PlayerOnTurn(), alphabet.RackFromString(rack, sc.game.Alphabet()))
}

func (sc *ShellController) challenge(fields []string) error {
	if len(fields) > 0 {
		addlBonus, err := strconv.Atoi(fields[0])
		if err != nil {
			return err
		}
		// Set it to single to have a base bonus of 0, and add passed-in bonus.
		sc.game.SetChallengeRule(pb.ChallengeRule_SINGLE)
		sc.game.ChallengeEvent(addlBonus, 0)
		sc.game.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	} else {
		// Do double-challenge.
		sc.game.ChallengeEvent(0, 0)
	}
	sc.curTurnNum++
	sc.showMessage(sc.game.ToDisplayText())
	return nil
}

func (sc *ShellController) addPlay(fields []string, commit bool) error {
	var playerid int
	var m *move.Move
	var err error
	// Figure out whose turn it is.

	playerid = sc.game.PlayerOnTurn()

	if len(fields) == 1 {
		if strings.HasPrefix(fields[0], "#") {
			if !commit {
				// This option makes no sense with just an `add` command
				return errors.New("cannot use this option with the `add` command, " +
					"you may have wanted to use the `commit` command instead")
			}
			// Add play that was generated.
			playID, err := strconv.Atoi(fields[0][1:])
			if err != nil {
				return err
			}

			idx := playID - 1 // since playID starts from 1
			if idx < 0 || idx > len(sc.curPlayList)-1 {
				return errors.New("play outside range")
			}
			m = sc.curPlayList[idx]
		} else if fields[0] == "pass" {
			rack := sc.game.RackFor(playerid)
			m = move.NewPassMove(rack.TilesOn(), sc.game.Alphabet())
		} else {
			return errors.New("unrecognized arguments to `add`")
		}

	} else if len(fields) == 2 {
		coords, word := fields[0], fields[1]

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
	if !commit {
		opp := (sc.game.PlayerOnTurn() + 1) % sc.game.NumPlayers()
		oppRack := sc.game.RackFor(opp)
		sc.aiplayer.AssignEquity([]*move.Move{m}, sc.game.Board(), sc.game.Bag(),
			oppRack)
		sc.curPlayList = append(sc.curPlayList, m)
		sort.Slice(sc.curPlayList, func(i, j int) bool {
			return sc.curPlayList[j].Equity() < sc.curPlayList[i].Equity()
		})
		sc.displayMoveList()
	} else {

		// Play the actual move on the board, draw tiles, etc.
		err = sc.game.PlayMove(m, true, 0)
		if err != nil {
			return err
		}
		log.Debug().Msgf("Added turn at turn num %v", sc.curTurnNum)
		sc.curTurnNum++
		sc.curPlayList = nil
		sc.simmer.Reset()
		sc.showMessage(sc.game.ToDisplayText())

	}
	return nil
}

func (sc *ShellController) handleAutoplay(args []string, options map[string]string) error {
	var logfile, lexicon, leavefile1, leavefile2, pegfile1, pegfile2 string
	if options["logfile"] == "" {
		logfile = "/tmp/autoplay.txt"
	} else {
		logfile = options["logfile"]
	}
	if options["lexicon"] == "" {
		lexicon = sc.config.DefaultLexicon
	} else {
		lexicon = options["lexicon"]
	}
	if options["leavefile1"] == "" {
		leavefile1 = ""
	} else {
		leavefile1 = options["leavefile1"]
	}
	if options["leavefile2"] == "" {
		leavefile2 = ""
	} else {
		leavefile2 = options["leavefile2"]
	}
	if options["pegfile1"] == "" {
		pegfile1 = ""
	} else {
		pegfile1 = options["pegfile1"]
	}
	if options["pegfile2"] == "" {
		pegfile2 = ""
	} else {
		pegfile2 = options["pegfile2"]
	}

	player1 := "exhaustiveleave"
	player2 := player1
	if len(args) == 1 {
		if args[0] == "stop" {
			if !sc.gameRunnerRunning {
				return errors.New("automatic game runner is not running")
			}
			sc.gameRunnerCancel()
			sc.gameRunnerRunning = false
			return nil

		}
	} else if len(args) == 2 {
		// It's player names
		player1 = args[0]
		player2 = args[1]
	}
	if sc.gameRunnerRunning {
		return errors.New("please stop automatic game runner before running another one")
	}

	sc.showMessage("automatic game runner will log to " + logfile)
	sc.gameRunnerCtx, sc.gameRunnerCancel = context.WithCancel(context.Background())
	err := automatic.StartCompVCompStaticGames(sc.gameRunnerCtx, sc.config, 1e9, runtime.NumCPU(),
		logfile, player1, player2, lexicon, leavefile1, leavefile2, pegfile1, pegfile2)
	if err != nil {
		return err
	}
	sc.gameRunnerRunning = true
	sc.showMessage("Started automatic game runner...")
	return nil
}

type shellcmd struct {
	cmd     string
	args    []string
	options map[string]string
}

func extractFields(line string) (*shellcmd, error) {
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return nil, errNoData
	}
	cmd := fields[0]
	var args []string
	options := map[string]string{}
	// handle options

	lastWasOption := false
	lastOption := ""
	for idx := 1; idx < len(fields); idx++ {
		if strings.HasPrefix(fields[idx], "-") {
			// option
			lastWasOption = true
			lastOption = fields[idx][1:]
			continue
		}
		if lastWasOption {
			lastWasOption = false
			options[lastOption] = fields[idx]
		} else {
			args = append(args, fields[idx])
		}
	}

	if lastWasOption {
		// all options are non-boolean, cannot have a naked option.
		return nil, errWrongOptionSyntax
	}

	return &shellcmd{
		cmd:     cmd,
		args:    args,
		options: options,
	}, nil
}

func (sc *ShellController) standardModeSwitch(line string, sig chan os.Signal) error {
	cmd, err := extractFields(line)
	if err != nil {
		return err
	}
	switch cmd.cmd {
	case "new":

		rules, err := game.NewGameRules(sc.config, board.CrosswordGameBoard,
			sc.config.DefaultLexicon, sc.config.DefaultLetterDistribution)
		if err != nil {
			sc.showError(err)
			break
		}

		players := []*pb.PlayerInfo{
			{Nickname: "arcadio", RealName: "José Arcadio Buendía"},
			{Nickname: "úrsula", RealName: "Úrsula Iguarán Buendía"},
		}

		sc.game, err = game.NewGame(rules, players)
		if err != nil {
			sc.showError(err)
			break
		}
		sc.game.StartGame()
		sc.game.SetBackupMode(game.InteractiveGameplayMode)
		// Set challenge rule to double by default. This can be overridden.
		sc.game.SetChallengeRule(pb.ChallengeRule_DOUBLE)
		err = sc.initGameDataStructures()
		if err != nil {
			sc.showError(err)
			break
		}
		sc.curTurnNum = 0
		sc.showMessage(sc.game.ToDisplayText())

	case "load":
		if cmd.args == nil {
			sc.showError(errors.New("need arguments for load"))
			break
		}
		err := sc.loadGCG(cmd.args)

		if err != nil {
			sc.showError(err)
			break
		}
		sc.curTurnNum = 0
		sc.showMessage(sc.game.ToDisplayText())

	case "n":
		err := sc.setToTurn(sc.curTurnNum + 1)
		if err != nil {
			sc.showError(err)
			break
		}
		sc.showMessage(sc.game.ToDisplayText())
	case "p":
		err := sc.setToTurn(sc.curTurnNum - 1)
		if err != nil {
			sc.showError(err)
			break
		}
		sc.showMessage(sc.game.ToDisplayText())

	case "s":
		sc.showMessage(sc.game.ToDisplayText())

	case "turn":
		if cmd.args == nil {
			sc.showError(errors.New("need argument for turn"))
			break
		}
		t, err := strconv.Atoi(cmd.args[0])
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

	case "rack":
		if cmd.args == nil {
			sc.showError(errors.New("need argument for rack"))
			break
		}

		rack := cmd.args[0]

		err := sc.addRack(strings.ToUpper(rack))
		if err != nil {
			sc.showError(err)
			break
		}
		sc.showMessage(sc.game.ToDisplayText())

	case "setlex":
		if cmd.args == nil {
			sc.showError(errors.New("must set a lexicon"))
			break
		}
		if sc.game == nil {
			sc.showError(errors.New("please load or create a game first"))
			break
		}
		letdist := "english"
		if len(cmd.args) == 2 {
			letdist = cmd.args[1]
		}
		lexname := cmd.args[0]
		rules, err := game.NewGameRules(sc.config, board.CrosswordGameBoard,
			lexname, letdist)
		if err != nil {
			sc.showError(err)
			break
		}
		err = sc.game.SetNewRules(rules)
		if err != nil {
			sc.showError(err)
			break
		}
		err = sc.initGameDataStructures()
		if err != nil {
			sc.showError(err)
			break
		}

	case "gen":

		var numPlays int
		var err error

		if cmd.args == nil {
			numPlays = 15
		} else {
			numPlays, err = strconv.Atoi(cmd.args[0])
			if err != nil {
				sc.showError(err)
				break
			}
		}
		if sc.game != nil {
			sc.genMovesAndDisplay(numPlays)
		}

	case "autoplay":
		err := sc.handleAutoplay(cmd.args, cmd.options)
		if err != nil {
			sc.showError(err)
		}

	case "sim":
		err := sc.handleSim(cmd.args)
		if err != nil {
			sc.showError(err)
		}

	case "add":
		err := sc.addPlay(cmd.args, false)
		if err != nil {
			sc.showError(err)
		}

	case "challenge":
		err := sc.challenge(cmd.args)
		if err != nil {
			sc.showError(err)
		}

	case "commit":
		err := sc.addPlay(cmd.args, true)
		if err != nil {
			sc.showError(err)
		}

	case "list":
		sc.displayMoveList()

	case "endgame":
		if sc.game == nil {
			showMessage("please load a game first with the `load` command", sc.l.Stderr())
			break
		}
		plies, deepening, simpleEval, disablePruning, err := endgameArgs(cmd.args)
		if err != nil {
			showMessage("Error: "+err.Error(), sc.l.Stderr())
			break
		}
		showMessage(fmt.Sprintf("plies %v, deepening %v, simpleEval %v, pruningDisabled %v",
			plies, deepening, simpleEval, disablePruning), sc.l.Stderr())

		sc.game.SetStateStackLength(plies)
		sc.game.SetBackupMode(game.SimulationMode)

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

		val, seq, err := sc.endgameSolver.Solve(plies)
		if err != nil {
			sc.showError(err)
			break
		}
		// And turn off simulation mode again.
		sc.game.SetBackupMode(game.InteractiveGameplayMode)
		sc.showMessage(fmt.Sprintf("Best sequence has a spread difference of %v", val))
		sc.printEndgameSequence(seq)

	case "exit":
		sig <- syscall.SIGINT
		return errors.New("sending quit signal")
	case "help":
		if cmd.args == nil {
			usage(sc.l.Stderr(), "standard", sc.execPath)
		} else {
			helptopic := cmd.args[0]
			usageTopic(sc.l.Stderr(), helptopic, sc.execPath)
		}
	case "mode":
		sc.modeSelector(cmd.args[0])

	case "export":
		if cmd.args == nil {
			sc.showError(errors.New("please provide a filename to save to"))
			break
		}
		filename := cmd.args[0]
		contents, err := gcgio.GameHistoryToGCG(sc.game.History(), true)
		if err != nil {
			sc.showError(err)
			break
		}
		f, err := os.Create(filename)
		if err != nil {
			sc.showError(err)
			break
		}
		log.Debug().Interface("game-history", sc.game.History()).Msg("converted game history to gcg")
		f.WriteString(contents)
		f.Close()
		sc.showMessage("gcg written to " + filename)

	case "autoanalyze":
		if cmd.args == nil {
			sc.showError(errors.New("please provide a filename to analyze"))
			break
		}
		filename := cmd.args[0]
		analysis, err := automatic.AnalyzeLogFile(filename)
		if err != nil {
			sc.showError(err)
			break
		}
		sc.showMessage(analysis)

	case "challengerule":
		if cmd.args == nil {
			sc.showError(errors.New("need rule"))
			break
		}
		var challRule pb.ChallengeRule
		switch cmd.args[0] {
		case "void":
			challRule = pb.ChallengeRule_VOID
		case "single":
			challRule = pb.ChallengeRule_SINGLE
		case "double":
			challRule = pb.ChallengeRule_DOUBLE
		case "5pt":
			challRule = pb.ChallengeRule_FIVE_POINT
		case "10pt":
			challRule = pb.ChallengeRule_TEN_POINT
		default:
			return errors.New("challenge rule nonexistent")
		}
		sc.game.SetChallengeRule(challRule)
		sc.showMessage("set challenge rule to " + cmd.args[0])

	default:

		log.Info().Msgf("command %v not found", strconv.Quote(cmd.cmd))

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
				sc.showError(err)
			}
		} else if sc.curMode == EndgameDebugMode {
			err := sc.endgameDebugModeSwitch(line, sig)
			if err != nil {
				sc.showError(err)
			}
		}

	}
	log.Debug().Msgf("Exiting readline loop...")
}
