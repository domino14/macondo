package shell

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"

	"github.com/domino14/macondo/endgame/negamax"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	"github.com/domino14/macondo/tilemapping"
)

type Response struct {
	message string
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
	players := []*pb.PlayerInfo{
		{Nickname: "arcadio", RealName: "José Arcadio Buendía"},
		{Nickname: "úrsula", RealName: "Úrsula Iguarán Buendía"},
	}
	if frand.Intn(2) == 1 {
		players[0], players[1] = players[1], players[0]
	}

	opts := sc.options.GameOptions
	conf := &bot.BotConfig{Config: *sc.config}

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

func (sc *ShellController) load(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("need arguments for load")
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

func (sc *ShellController) show(cmd *shellcmd) (*Response, error) {
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) list(cmd *shellcmd) (*Response, error) {
	res := sc.genDisplayMoveList()
	return msg(res), nil
}

func (sc *ShellController) next(cmd *shellcmd) (*Response, error) {
	err := sc.setToTurn(sc.curTurnNum + 1)
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) prev(cmd *shellcmd) (*Response, error) {
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

func (sc *ShellController) aiplay(cmd *shellcmd) (*Response, error) {
	return nil, sc.commitAIMove()
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
	plies := 4
	var maxtime int
	var maxthreads = 1
	var disableID bool
	var disableTT bool
	var enableFW bool
	var enableKillerPlayOptim bool
	var err error

	if cmd.options["plies"] != "" {
		plies, err = strconv.Atoi(cmd.options["plies"])
		if err != nil {
			return nil, err
		}
	}

	if cmd.options["maxtime"] != "" {
		maxtime, err = strconv.Atoi(cmd.options["maxtime"])
		if err != nil {
			return nil, err
		}
	}

	if cmd.options["disable-id"] == "true" {
		disableID = true
	}
	if cmd.options["disable-tt"] == "true" {
		disableTT = true
	}
	if cmd.options["killer-optim"] == "true" {
		enableKillerPlayOptim = true
	}
	if cmd.options["lazysmp-threads"] != "" {
		maxthreads, err = strconv.Atoi(cmd.options["lazysmp-threads"])
		if err != nil {
			return nil, err
		}
	}
	if cmd.options["first-win-optim"] == "true" {
		enableFW = true
	}
	sc.showMessage(fmt.Sprintf(
		"plies %v, maxtime %v, threads %v",
		plies, maxtime, maxthreads))

	sc.game.SetBackupMode(game.SimulationMode)

	defer func() {
		sc.game.SetBackupMode(game.InteractiveGameplayMode)
		sc.game.SetStateStackLength(1)
	}()
	var cancel context.CancelFunc
	ctx := context.Background()
	if maxtime > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(maxtime)*time.Second)
		defer cancel()
	}

	// clear out the last value of this endgame node; gc should
	// delete the tree.
	sc.endgameSolver = new(negamax.Solver)
	err = sc.endgameSolver.Init(sc.gen, sc.game.Game)
	if err != nil {
		return nil, err
	}

	sc.endgameSolver.SetIterativeDeepening(!disableID)
	sc.endgameSolver.SetKillerPlayOptim(enableKillerPlayOptim)
	sc.endgameSolver.SetTranspositionTableOptim(!disableTT)
	sc.endgameSolver.SetThreads(maxthreads)
	sc.endgameSolver.SetFirstWinOptim(enableFW)

	sc.showMessage(sc.game.ToDisplayText())

	val, seq, err := sc.endgameSolver.Solve(ctx, plies)
	if err != nil {
		return nil, err
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
	return msg("done"), nil
}

func (sc *ShellController) infer(cmd *shellcmd) (*Response, error) {
	if sc.game == nil {
		return nil, errors.New("please load a game first with the `load` command")
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

	for opt, val := range cmd.options {
		switch opt {
		case "threads":
			threads, err = strconv.Atoi(val)
			if err != nil {
				return nil, err
			}

		case "time":
			timesec, err = strconv.Atoi(val)
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
	if options["export"] != "" {

		f, err := os.Create(options["export"] + ".gcg")
		if err != nil {
			return nil, err
		}

		if options["letterdist"] == "" {
			options["letterdist"] = sc.config.DefaultLetterDistribution
		}
		if options["lexicon"] == "" {
			options["lexicon"] = sc.config.DefaultLexicon
		}

		err = automatic.ExportGCG(
			sc.config, filename, options["letterdist"], options["lexicon"],
			options["boardlayout"], options["export"], f)
		if err != nil {
			ferr := os.Remove(options["export"] + ".gcg")
			if ferr != nil {
				log.Err(ferr).Msg("removing gcg output file")
			}
			return nil, err
		}
		err = f.Close()
		if err != nil {
			return nil, err
		}
		return msg("exported to " + options["export"] + ".gcg"), nil
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
	dist, err := tilemapping.GetDistribution(sc.config, sc.config.DefaultLetterDistribution)
	if err != nil {
		return nil, err
	}
	els, err := equity.NewExhaustiveLeaveCalculator(sc.config.DefaultLexicon,
		sc.config, "")
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
