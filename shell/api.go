package shell

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/preendgame"
)

const defaultEndgamePlies = 4

type Response struct {
	message string
}

type CmdOptions map[string][]string

func (c CmdOptions) String(key string) string {
	v := c[key]
	if len(v) > 0 {
		return v[0]
	}
	return ""
}

func (c CmdOptions) Int(key string) (int, error) {
	v := c[key]
	if len(v) == 0 {
		return 0, errors.New(key + " not found in options")
	}
	return strconv.Atoi(v[0])
}
func (c CmdOptions) IntDefault(key string, defaultI int) (int, error) {
	v := c[key]
	if len(v) == 0 {
		return defaultI, nil
	}
	return strconv.Atoi(v[0])
}

func (c CmdOptions) Bool(key string) bool {
	v := c[key]
	if len(v) == 0 {
		return false
	}
	return strings.ToLower(v[0]) == "true"
}

func (c CmdOptions) StringArray(key string) []string {
	return c[key]
}

func msg(message string) *Response {
	return &Response{message: message}
}

func (sc *ShellController) set(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return msg(sc.options.ToDisplayText()), nil
	}
	opt := cmd.args[0]
	if len(cmd.args) == 1 {
		_, val := sc.options.Show(opt)
		return msg(val), nil
	}
	values := cmd.args[1:]
	ret, err := sc.Set(opt, values)
	if err != nil {
		return nil, err
	}
	return msg("set " + opt + " to " + ret), nil
}

func (sc *ShellController) gid(cmd *shellcmd) (*Response, error) {
	if sc.game == nil {
		return nil, errors.New("no currently loaded game")
	}
	gid := sc.game.History().Uid
	if gid != "" {
		idauth := sc.game.History().IdAuth
		fullID := strings.TrimSpace(idauth + " " + gid)
		return msg(fullID), nil
	}
	return nil, errors.New("no ID set for this game")
}

func (sc *ShellController) newGame(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	players := []*pb.PlayerInfo{
		{Nickname: "arcadio", RealName: "José Arcadio Buendía"},
		{Nickname: "úrsula", RealName: "Úrsula Iguarán Buendía"},
	}
	if frand.Intn(2) == 1 {
		players[0], players[1] = players[1], players[0]
	}

	opts := sc.options.GameOptions
	leavesFile := ""
	if opts.BoardLayoutName == board.SuperCrosswordGameLayout {
		leavesFile = "super-leaves.klv2"
	}

	conf := &bot.BotConfig{Config: *sc.config, LeavesFile: leavesFile}

	g, err := bot.NewBotTurnPlayer(conf, &opts, players, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return nil, err
	}
	sc.game = g
	err = sc.initGameDataStructures()
	if err != nil {
		return nil, err
	}
	sc.curTurnNum = 0
	return msg(sc.game.ToDisplayText()), nil
}
func (sc *ShellController) solving() bool {
	return (sc.endgameSolver != nil && sc.endgameSolver.IsSolving()) ||
		(sc.preendgameSolver != nil && sc.preendgameSolver.IsSolving()) ||
		(sc.simmer != nil && sc.simmer.IsSimming()) ||
		(sc.rangefinder != nil && sc.rangefinder.IsBusy()) ||
		sc.botBusy
}

func (sc *ShellController) load(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("need arguments for load")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}

	if cmd.args[0] == "cgp" {
		if len(cmd.args) < 2 {
			return nil, errors.New("need to provide a cgp string")
		}
		cgpStr := strings.Join(cmd.args[1:], " ")
		err := sc.loadCGP(cgpStr)
		if err != nil {
			return nil, err
		}

	} else {
		err := sc.loadGCG(cmd.args)
		if err != nil {
			return nil, err
		}
	}
	sc.curTurnNum = 0
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) unload(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	sc.game = nil
	return msg("No active game."), nil
}

func (sc *ShellController) show(cmd *shellcmd) (*Response, error) {
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) list(cmd *shellcmd) (*Response, error) {
	res := sc.genDisplayMoveList()
	return msg(res), nil
}

func (sc *ShellController) next(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	err := sc.setToTurn(sc.curTurnNum + 1)
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) last(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	err := sc.setToTurn(len(sc.game.History().Events))
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) prev(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	err := sc.setToTurn(sc.curTurnNum - 1)
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) name(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) < 3 {
		return nil, errors.New("need 3 arguments for name")
	}
	if sc.game == nil {
		return nil, errors.New("no game is loaded")
	}
	p, err := strconv.Atoi(cmd.args[0])
	if err != nil {
		return nil, err
	}
	p -= 1
	if p < 0 || p >= len(sc.game.History().Players) {
		return nil, errors.New("player index not in range")
	}
	err = sc.game.RenamePlayer(p, &pb.PlayerInfo{
		Nickname: cmd.args[1],
		RealName: strings.Join(cmd.args[2:], " "),
	})
	if err != nil {
		return nil, err
	}

	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) note(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) < 1 {
		return nil, errors.New("need at least one argument for note")
	}
	if sc.game == nil {
		return nil, errors.New("no game is loaded")
	}
	if sc.game.Turn() == 0 {
		return nil, errors.New("there must be at least one turn that has been played")
	}
	note := strings.Join(cmd.args, " ")
	err := sc.game.AddNote(note)
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) turn(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("need argument for turn")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}
	t, err := strconv.Atoi(cmd.args[0])
	if err != nil {
		return nil, err
	}
	err = sc.setToTurn(t)
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) rack(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("need argument for rack")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}
	rack := cmd.args[0]
	err := sc.addRack(strings.ToUpper(rack))
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) generate(cmd *shellcmd) (*Response, error) {
	var numPlays int
	var err error

	if sc.game == nil {
		return nil, errors.New("please load or create a game first")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}

	if cmd.args == nil {
		numPlays = 15
	} else {
		numPlays, err = strconv.Atoi(cmd.args[0])
		if err != nil {
			return nil, err
		}
	}

	return msg(sc.genMovesAndDescription(numPlays)), nil
}

func (sc *ShellController) autoplay(cmd *shellcmd) (*Response, error) {
	return nil, sc.handleAutoplay(cmd.args, cmd.options)
}

func (sc *ShellController) sim(cmd *shellcmd) (*Response, error) {
	return nil, sc.handleSim(cmd.args, cmd.options)
}

func (sc *ShellController) add(cmd *shellcmd) (*Response, error) {
	return nil, sc.addPlay(cmd.args)
}

func (sc *ShellController) commit(cmd *shellcmd) (*Response, error) {
	return nil, sc.commitPlay(cmd.args)
}

func (sc *ShellController) eliteplay(cmd *shellcmd) (*Response, error) {
	return nil, sc.commitAIMove()
}

func (sc *ShellController) hastyplay(cmd *shellcmd) (*Response, error) {
	return nil, sc.commitHastyMove()
}

func (sc *ShellController) selftest(cmd *shellcmd) (*Response, error) {
	_, err := sc.newGame(cmd)
	if err != nil {
		return nil, err
	}
	for sc.IsPlaying() {
		err := sc.commitAIMove()
		if err != nil {
			return nil, err
		}
	}
	return nil, nil
}

func (sc *ShellController) challenge(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	fields := cmd.args
	if len(fields) > 0 {
		addlBonus, err := strconv.Atoi(fields[0])
		if err != nil {
			return nil, err
		}
		// Set it to single to have a base bonus of 0, and add passed-in bonus.
		sc.game.SetChallengeRule(pb.ChallengeRule_SINGLE)
		sc.game.ChallengeEvent(addlBonus, 0)
		sc.game.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	} else {
		// Do double-challenge.
		sc.game.ChallengeEvent(0, 0)
	}
	sc.curTurnNum = sc.game.Turn()
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) endgame(cmd *shellcmd) (*Response, error) {
	if sc.game == nil {
		return nil, errors.New("please load a game first with the `load` command")
	}

	if len(cmd.args) > 0 && cmd.args[0] == "stop" {
		if sc.endgameSolver.IsSolving() {
			sc.endgameCancel()
		} else {
			return nil, errors.New("no endgame to cancel")
		}
		return msg(""), nil
	}

	if sc.solving() {
		return nil, errMacondoSolving
	}

	var plies int
	var maxtime int
	var maxthreads = runtime.NumCPU() - 1
	if maxthreads == 0 {
		maxthreads = 1
	}
	if maxthreads > negamax.MaxLazySMPThreads {
		maxthreads = negamax.MaxLazySMPThreads
	}
	var disableID bool
	var disableTT bool
	var enableFW bool
	var err error

	if plies, err = cmd.options.IntDefault("plies", defaultEndgamePlies); err != nil {
		return nil, err
	}
	if maxtime, err = cmd.options.IntDefault("maxtime", 0); err != nil {
		return nil, err
	}
	if maxthreads, err = cmd.options.IntDefault("threads", maxthreads); err != nil {
		return nil, err
	}
	disableID = cmd.options.Bool("disable-id")
	disableTT = cmd.options.Bool("disable-tt")
	enableFW = cmd.options.Bool("first-win-optim")

	// clear out the last value of this endgame node; gc should
	// delete the tree.
	sc.endgameSolver = new(negamax.Solver)

	if cmd.options.Bool("log") {
		sc.endgameLogFile, err = os.Create(EndgameLog)
		if err != nil {
			return nil, err
		}
		sc.endgameSolver.SetLogStream(sc.endgameLogFile)
		sc.showMessage("endgame will log to " + EndgameLog)
	}

	sc.showMessage(fmt.Sprintf(
		"plies %v, maxtime %v, threads %v",
		plies, maxtime, maxthreads))

	sc.game.SetBackupMode(game.SimulationMode)

	sc.endgameCtx, sc.endgameCancel = context.WithCancel(context.Background())
	if maxtime > 0 {
		sc.endgameCtx, sc.endgameCancel = context.WithTimeout(sc.endgameCtx, time.Duration(maxtime)*time.Second)
	}

	err = sc.endgameSolver.Init(sc.gen, sc.game.Game)
	if err != nil {
		return nil, err
	}

	sc.endgameSolver.SetIterativeDeepening(!disableID)
	sc.endgameSolver.SetTranspositionTableOptim(!disableTT)
	sc.endgameSolver.SetThreads(maxthreads)
	sc.endgameSolver.SetFirstWinOptim(enableFW)

	sc.showMessage(sc.game.ToDisplayText())

	go func() {
		defer func() {
			sc.game.SetBackupMode(game.InteractiveGameplayMode)
			sc.game.SetStateStackLength(1)
		}()

		val, seq, err := sc.endgameSolver.Solve(sc.endgameCtx, plies)
		if err != nil {
			sc.showError(err)
			return
		}
		if !enableFW {
			sc.showMessage(fmt.Sprintf("Best sequence has a spread difference of %v", val))
		} else {
			if val+int16(sc.game.CurrentSpread()) > 0 {
				sc.showMessage("Win found!")
			} else {
				sc.showMessage("Win was not found.")
			}
			sc.showMessage(fmt.Sprintf("Spread diff: %v. Note: this sequence may not be correct. Turn off first-win-optim to search more accurately.", val))
		}
		sc.showMessage(fmt.Sprintf("Final spread after seq: %d", val+int16(sc.game.CurrentSpread())))
		sc.printEndgameSequence(seq)
	}()
	return msg(""), nil
}

func (sc *ShellController) preendgame(cmd *shellcmd) (*Response, error) {
	if sc.game == nil {
		return nil, errors.New("please load a game first with the `load` command")
	}
	endgamePlies := 4

	if len(cmd.args) > 0 && cmd.args[0] == "stop" {
		if sc.preendgameSolver.IsSolving() {
			sc.pegCancel()
		} else {
			return nil, errors.New("no pre-endgame to cancel")
		}
		return msg(""), nil
	}

	if sc.solving() {
		return nil, errMacondoSolving
	}

	var maxtime int
	var maxthreads = 0
	var maxsolutions = 30
	var err error
	var earlyCutoff bool
	var skipNonEmptying bool
	var skipLoss bool
	var skipTiebreaker bool
	var disableIterativeDeepening bool
	knownOppRack := cmd.options.String("opprack")

	if endgamePlies, err = cmd.options.IntDefault("endgameplies", defaultEndgamePlies); err != nil {
		return nil, err
	}
	if maxtime, err = cmd.options.IntDefault("maxtime", 0); err != nil {
		return nil, err
	}
	if maxthreads, err = cmd.options.IntDefault("threads", 0); err != nil {
		return nil, err
	}
	if maxsolutions, err = cmd.options.IntDefault("maxsolutions", maxsolutions); err != nil {
		return nil, err
	}
	skipNonEmptying = cmd.options.Bool("skip-non-emptying")
	skipLoss = cmd.options.Bool("skip-loss")
	earlyCutoff = cmd.options.Bool("early-cutoff")
	skipTiebreaker = cmd.options.Bool("skip-tiebreaker")
	disableIterativeDeepening = cmd.options.Bool("disable-id")
	movesToSolveStrs := cmd.options.StringArray("only-solve")
	movesToSolve := []*move.Move{}

	for _, ms := range movesToSolveStrs {
		m, err := sc.game.ParseMove(
			sc.game.PlayerOnTurn(),
			sc.options.lowercaseMoves,
			strings.Fields(ms))
		if err != nil {
			return nil, err
		}
		movesToSolve = append(movesToSolve, m)
	}
	sc.showMessage(fmt.Sprintf(
		"endgameplies %v, maxtime %v, threads %v",
		endgamePlies, maxtime, maxthreads))
	gd, err := kwg.Get(sc.game.Config().WGLConfig(), sc.game.LexiconName())
	if err != nil {
		return nil, err
	}
	sc.showMessage(sc.game.ToDisplayText())
	sc.preendgameSolver = new(preendgame.Solver)
	sc.preendgameSolver.Init(sc.game.Game, gd)

	if maxthreads != 0 {
		sc.preendgameSolver.SetThreads(maxthreads)
	}

	if cmd.options.Bool("log") {
		sc.pegLogFile, err = os.Create(PEGLog)
		if err != nil {
			return nil, err
		}
		sc.preendgameSolver.SetLogStream(sc.pegLogFile)
		sc.showMessage("peg will log to " + PEGLog)
	}

	if knownOppRack != "" {
		knownOppRack = strings.ToUpper(knownOppRack)
		r, err := tilemapping.ToMachineLetters(knownOppRack, sc.game.Alphabet())
		if err != nil {
			return nil, err
		}
		sc.preendgameSolver.SetKnownOppRack(r)
	}
	sc.preendgameSolver.SetEndgamePlies(endgamePlies)
	sc.preendgameSolver.SetEarlyCutoffOptim(earlyCutoff)
	sc.preendgameSolver.SetSkipNonEmptyingOptim(skipNonEmptying)
	sc.preendgameSolver.SetSkipTiebreaker(skipTiebreaker)
	sc.preendgameSolver.SetSkipLossOptim(skipLoss)
	sc.preendgameSolver.SetIterativeDeepening(!disableIterativeDeepening)
	sc.preendgameSolver.SetSolveOnly(movesToSolve)
	sc.pegCtx, sc.pegCancel = context.WithCancel(context.Background())
	if maxtime > 0 {
		sc.pegCtx, sc.pegCancel = context.WithTimeout(sc.pegCtx, time.Duration(maxtime)*time.Second)
	}
	go func() {
		moves, err := sc.preendgameSolver.Solve(sc.pegCtx)
		if err != nil {
			sc.showError(err)
			return
		}
		if len(moves) < maxsolutions {
			maxsolutions = len(moves)
		}
		if sc.pegLogFile != nil {
			err := sc.pegLogFile.Close()
			if err != nil {
				log.Err(err).Msg("closing-log-file")
			}
		}
		sc.showMessage(sc.preendgameSolver.SolutionStats(maxsolutions))
	}()
	return msg(""), nil
}

func (sc *ShellController) infer(cmd *shellcmd) (*Response, error) {
	if sc.game == nil {
		return nil, errors.New("please load a game first with the `load` command")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}

	var err error
	var threads, timesec int

	if len(cmd.args) > 0 {
		switch cmd.args[0] {
		case "log":
			sc.rangefinderFile, err = os.Create(InferLog)
			if err != nil {
				return nil, err
			}
			sc.rangefinder.SetLogStream(sc.rangefinderFile)
			sc.showMessage("inference engine will log to " + InferLog)

		case "details":
			sc.showMessage(sc.rangefinder.AnalyzeInferences(true))

		default:
			return nil, errors.New("don't recognize " + cmd.args[0])
		}

		return nil, nil
	}

	for opt := range cmd.options {
		switch opt {
		case "threads":
			threads, err = cmd.options.Int(opt)
			if err != nil {
				return nil, err
			}

		case "time":
			timesec, err = cmd.options.Int(opt)
			if err != nil {
				return nil, err
			}

		default:
			return nil, errors.New("option " + opt + " not recognized")

		}
	}
	if threads != 0 {
		sc.rangefinder.SetThreads(threads)
	}
	if timesec == 0 {
		timesec = 5
	}
	err = sc.rangefinder.PrepareFinder(sc.game.RackFor(sc.game.PlayerOnTurn()).TilesOn())
	if err != nil {
		return nil, err
	}
	timeout, _ := context.WithTimeout(
		context.Background(), time.Duration(timesec*int(time.Second)))

	sc.showMessage("Rangefinding started. Please wait until it is done.")

	go func() {
		err := sc.rangefinder.Infer(timeout)
		if err != nil {
			sc.showError(err)
		}
		sc.showMessage(sc.rangefinder.AnalyzeInferences(false))
		log.Debug().Msg("inference thread exiting...")
	}()

	return nil, nil

}

func (sc *ShellController) help(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return usage("standard")
	} else {
		helptopic := cmd.args[0]
		return usageTopic(helptopic)
	}
}

func (sc *ShellController) setMode(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return msg("Current mode: " + modeToStr(sc.curMode)), nil
	}
	mode := cmd.args[0]
	m, err := modeFromStr(mode)
	if err != nil {
		return nil, err
	}
	sc.curMode = m
	return msg("Setting current mode to " + mode), nil
}

func (sc *ShellController) export(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("please provide a filename to save to")
	}
	filename := cmd.args[0]
	contents, err := gcgio.GameHistoryToGCG(sc.game.History(), true)
	if err != nil {
		return nil, err
	}
	f, err := os.Create(filename)
	if err != nil {
		return nil, err
	}
	log.Debug().Interface("game-history", sc.game.History()).Msg("converted game history to gcg")
	f.WriteString(contents)
	f.Close()
	return msg("gcg written to " + filename), nil
}

func (sc *ShellController) autoAnalyze(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("please provide a filename to analyze")
	}
	filename := cmd.args[0]
	options := cmd.options
	if options.String("export") != "" {

		f, err := os.Create(options.String("export") + ".gcg")
		if err != nil {
			return nil, err
		}
		ld := options.String("letterdist")
		lex := options.String("lex")
		if ld == "" {
			ld = sc.config.GetString(config.ConfigDefaultLetterDistribution)
		}
		if lex == "" {
			lex = sc.config.GetString(config.ConfigDefaultLexicon)
		}

		err = automatic.ExportGCG(
			sc.config, filename, ld, lex,
			options.String("boardlayout"), options.String("export"), f)
		if err != nil {
			ferr := os.Remove(options.String("export") + ".gcg")
			if ferr != nil {
				log.Err(ferr).Msg("removing gcg output file")
			}
			return nil, err
		}
		err = f.Close()
		if err != nil {
			return nil, err
		}
		return msg("exported to " + options.String("export") + ".gcg"), nil
	}
	analysis, err := automatic.AnalyzeLogFile(filename)
	if err != nil {
		return nil, err
	}
	return msg(analysis), nil
}

func (sc *ShellController) leave(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) != 1 {
		return nil, errors.New("please provide a leave")
	}
	ldName := sc.config.GetString(config.ConfigDefaultLetterDistribution)
	dist, err := tilemapping.GetDistribution(sc.config.WGLConfig(), ldName)
	if err != nil {
		return nil, err
	}
	leaves := ""
	if strings.HasSuffix(ldName, "_super") {
		leaves = "super-leaves.klv2"
	}
	els, err := equity.NewExhaustiveLeaveCalculator(sc.config.GetString(config.ConfigDefaultLexicon),
		sc.config, leaves)
	if err != nil {
		return nil, err
	}
	leave, err := tilemapping.ToMachineWord(cmd.args[0], dist.TileMapping())
	if err != nil {
		return nil, err
	}
	res := els.LeaveValue(leave)
	return msg(strconv.FormatFloat(res, 'f', 3, 64)), nil
}

func (sc *ShellController) cgp(cmd *shellcmd) (*Response, error) {
	cgpstr := sc.game.ToCGP(false)
	return msg(cgpstr), nil
}

func (sc *ShellController) check(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) == 0 {
		return nil, errors.New("please provide a word or space-separated list of words to check")
	}
	dist, err := tilemapping.GetDistribution(sc.config.WGLConfig(),
		sc.config.GetString(config.ConfigDefaultLetterDistribution))
	if err != nil {
		return nil, err
	}
	k, err := kwg.Get(sc.config.WGLConfig(), sc.config.GetString(config.ConfigDefaultLexicon))
	if err != nil {
		return nil, err
	}
	lex := kwg.Lexicon{KWG: *k}

	playValid := true
	wordsFriendly := []string{}

	for _, w := range cmd.args {
		wordFriendly := strings.Trim(strings.ToUpper(w), ",")
		wordsFriendly = append(wordsFriendly, wordFriendly)

		word, err := tilemapping.ToMachineWord(wordFriendly, dist.TileMapping())
		if err != nil {
			return nil, err
		}
		valid := lex.HasWord(word)
		if !valid {
			playValid = false
		}
	}
	validStr := "VALID"
	if !playValid {
		validStr = "INVALID"
	}

	return msg(fmt.Sprintf("The play (%v) is %v in %v", strings.Join(wordsFriendly, ","), validStr, sc.config.GetString(config.ConfigDefaultLexicon))), nil

}
