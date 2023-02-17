package shell

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/endgame/alphabeta"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
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
	g, err := aiturnplayer.NewBotTurnPlayer(sc.config, &opts, players, pb.BotRequest_HASTY_BOT)
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
	return nil, sc.handleSim(cmd.args)
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
	var maxnodes int
	var disablePruning bool
	var disableID bool
	var complexEstimator bool
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

	if cmd.options["maxnodes"] != "" {
		maxnodes, err = strconv.Atoi(cmd.options["maxnodes"])
		if err != nil {
			return nil, err
		}
	}

	if cmd.options["disable-pruning"] == "true" {
		disablePruning = true
	}
	if cmd.options["disable-id"] == "true" {
		disableID = true
	}
	if cmd.options["complex-estimator"] == "true" {
		complexEstimator = true
	}

	sc.showMessage(fmt.Sprintf(
		"plies %v, maxtime %v, maxnodes %v",
		plies, maxtime, maxnodes))

	sc.game.SetStateStackLength(plies)
	sc.game.SetBackupMode(game.SimulationMode)

	defer func() {
		sc.game.SetBackupMode(game.InteractiveGameplayMode)
		sc.game.SetStateStackLength(1)
	}()

	oldmaxtime := sc.config.AlphaBetaTimeLimit

	sc.config.AlphaBetaTimeLimit = maxtime

	defer func() {
		sc.config.AlphaBetaTimeLimit = oldmaxtime
	}()

	// clear out the last value of this endgame node; gc should
	// delete the tree.
	sc.curEndgameNode = nil
	sc.endgameSolver = new(alphabeta.Solver)
	err = sc.endgameSolver.Init(sc.gen, sc.backupgen, sc.game.Game, sc.config)
	if err != nil {
		return nil, err
	}

	sc.endgameSolver.SetIterativeDeepening(!disableID)
	sc.endgameSolver.SetComplexEvaluator(complexEstimator)
	sc.endgameSolver.SetPruningDisabled(disablePruning)

	sc.showMessage(sc.game.ToDisplayText())

	val, seq, err := sc.endgameSolver.Solve(plies)
	if err != nil {
		return nil, err
	}

	sc.showMessage(fmt.Sprintf("Best sequence has a spread difference of %v", val))
	sc.printEndgameSequence(seq)
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
	dist, err := alphabet.Get(sc.config, sc.config.DefaultLetterDistribution)
	if err != nil {
		return nil, err
	}
	els, err := equity.NewExhaustiveLeaveCalculator(sc.config.DefaultLexicon,
		sc.config, equity.LeaveFilename)
	if err != nil {
		return nil, err
	}
	leave, err := alphabet.ToMachineWord(cmd.args[0], dist.Alphabet())
	if err != nil {
		return nil, err
	}
	res := els.LeaveValue(leave)
	return msg(strconv.FormatFloat(res, 'f', 3, 64)), nil
}
