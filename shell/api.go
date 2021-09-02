package shell

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/rs/zerolog/log"

	airunner "github.com/domino14/macondo/ai/runner"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/endgame/alphabeta"
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

func (sc *ShellController) newGame(cmd *shellcmd) (*Response, error) {
	players := []*pb.PlayerInfo{
		{Nickname: "arcadio", RealName: "José Arcadio Buendía"},
		{Nickname: "úrsula", RealName: "Úrsula Iguarán Buendía"},
	}

	opts := sc.options.GameOptions
	g, err := airunner.NewAIGameRunner(sc.config, &opts, players, pb.BotRequest_HASTY_BOT)
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
	err := sc.loadGCG(cmd.args)
	if err != nil {
		return nil, err
	}
	sc.curTurnNum = 0
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) show(cmd *shellcmd) (*Response, error) {
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) list(cmd *shellcmd) (*Response, error) {
	sc.displayMoveList()
	return nil, nil
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

	if cmd.args == nil {
		numPlays = 15
	} else {
		numPlays, err = strconv.Atoi(cmd.args[0])
		if err != nil {
			return nil, err
		}
	}
	if sc.game == nil {
		return nil, errors.New("please load or create a game first")
	} else {
		sc.genMovesAndDisplay(numPlays)
	}
	return nil, nil
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
	plies, deepening, simpleEval, disablePruning, err := endgameArgs(cmd.args)
	if err != nil {
		return nil, err
	}
	sc.showMessage(fmt.Sprintf(
		"plies %v, deepening %v, simpleEval %v, pruningDisabled %v",
		plies, deepening, simpleEval, disablePruning))

	sc.game.SetStateStackLength(plies)
	sc.game.SetBackupMode(game.SimulationMode)

	// clear out the last value of this endgame node; gc should
	// delete the tree.
	sc.curEndgameNode = nil
	sc.endgameSolver = new(alphabeta.Solver)
	err = sc.endgameSolver.Init(sc.gen, &sc.game.Game)
	if err != nil {
		return nil, err
	}
	sc.endgameSolver.SetIterativeDeepening(deepening)
	sc.endgameSolver.SetSimpleEvaluator(simpleEval)
	sc.endgameSolver.SetPruningDisabled(disablePruning)

	sc.showMessage(sc.game.ToDisplayText())

	val, seq, err := sc.endgameSolver.Solve(plies)
	if err != nil {
		return nil, err
	}
	// And turn off simulation mode again.
	sc.game.SetBackupMode(game.InteractiveGameplayMode)
	sc.game.SetStateStackLength(1)

	sc.showMessage(fmt.Sprintf("Best sequence has a spread difference of %v", val))
	sc.printEndgameSequence(seq)
	return nil, nil
}

func (sc *ShellController) help(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return usage("standard", sc.execPath)
	} else {
		helptopic := cmd.args[0]
		return usageTopic(helptopic, sc.execPath)
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
	analysis, err := automatic.AnalyzeLogFile(filename)
	if err != nil {
		return nil, err
	}
	return msg(analysis), nil
}
