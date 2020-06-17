package shell

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/endgame/alphabeta"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

func (sc *ShellController) newGame(cmd *shellcmd) error {
	rules, err := game.NewGameRules(sc.config, board.CrosswordGameBoard,
		sc.config.DefaultLexicon, sc.config.DefaultLetterDistribution)
	if err != nil {
		return err
	}

	players := []*pb.PlayerInfo{
		{Nickname: "arcadio", RealName: "José Arcadio Buendía"},
		{Nickname: "úrsula", RealName: "Úrsula Iguarán Buendía"},
	}

	sc.game, err = game.NewGame(rules, players)
	if err != nil {
		return err
	}
	sc.game.StartGame()
	sc.game.SetBackupMode(game.InteractiveGameplayMode)
	// Set challenge rule to double by default. This can be overridden.
	sc.game.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	err = sc.initGameDataStructures()
	if err != nil {
		return err
	}
	sc.curTurnNum = 0
	sc.showMessage(sc.game.ToDisplayText())
	return nil
}

func (sc *ShellController) load(cmd *shellcmd) error {
	if cmd.args == nil {
		return errors.New("need arguments for load")
	}
	err := sc.loadGCG(cmd.args)
	if err != nil {
		return err
	}
	sc.curTurnNum = 0
	sc.showMessage(sc.game.ToDisplayText())
	return nil
}

func (sc *ShellController) next(cmd *shellcmd) error {
	err := sc.setToTurn(sc.curTurnNum + 1)
	if err != nil {
		return err
	}
	sc.showMessage(sc.game.ToDisplayText())
	return nil
}

func (sc *ShellController) prev(cmd *shellcmd) error {
	err := sc.setToTurn(sc.curTurnNum - 1)
	if err != nil {
		return err
	}
	sc.showMessage(sc.game.ToDisplayText())
	return nil
}

func (sc *ShellController) turn(cmd *shellcmd) error {
	if cmd.args == nil {
		return errors.New("need argument for turn")
	}
	t, err := strconv.Atoi(cmd.args[0])
	if err != nil {
		return err
	}
	err = sc.setToTurn(t)
	if err != nil {
		return err
	}
	sc.showMessage(sc.game.ToDisplayText())
	return nil
}

func (sc *ShellController) rack(cmd *shellcmd) error {
	if cmd.args == nil {
		return errors.New("need argument for rack")
	}
	rack := cmd.args[0]
	err := sc.addRack(strings.ToUpper(rack))
	if err != nil {
		return err
	}
	sc.showMessage(sc.game.ToDisplayText())
	return nil
}

func (sc *ShellController) setlex(cmd *shellcmd) error {
	if cmd.args == nil {
		return errors.New("must set a lexicon")
	}
	if sc.game == nil {
		return errors.New("please load or create a game first")
	}
	letdist := "english"
	if len(cmd.args) == 2 {
		letdist = cmd.args[1]
	}
	lexname := cmd.args[0]
	rules, err := game.NewGameRules(
		sc.config, board.CrosswordGameBoard, lexname, letdist)
	if err != nil {
		return err
	}
	err = sc.game.SetNewRules(rules)
	if err != nil {
		return err
	}
	return sc.initGameDataStructures()
}

func (sc *ShellController) generate(cmd *shellcmd) error {
	var numPlays int
	var err error

	if cmd.args == nil {
		numPlays = 15
	} else {
		numPlays, err = strconv.Atoi(cmd.args[0])
		if err != nil {
			return err
		}
	}
	if sc.game != nil {
		sc.genMovesAndDisplay(numPlays)
	}
	return nil
}

func (sc *ShellController) autoplay(cmd *shellcmd) error {
	return sc.handleAutoplay(cmd.args, cmd.options)
}

func (sc *ShellController) sim(cmd *shellcmd) error {
	return sc.handleSim(cmd.args)
}

func (sc *ShellController) add(cmd *shellcmd) error {
	return sc.addPlay(cmd.args, false)
}

func (sc *ShellController) commit(cmd *shellcmd) error {
	return sc.addPlay(cmd.args, true)
}

func (sc *ShellController) challenge(cmd *shellcmd) error {
	fields := cmd.args
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

func (sc *ShellController) endgame(cmd *shellcmd) error {
	if sc.game == nil {
		showMessage("please load a game first with the `load` command", sc.l.Stderr())
		return nil
	}
	plies, deepening, simpleEval, disablePruning, err := endgameArgs(cmd.args)
	if err != nil {
		showMessage("Error: "+err.Error(), sc.l.Stderr())
		return nil
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
		return nil
	}
	sc.endgameSolver.SetIterativeDeepening(deepening)
	sc.endgameSolver.SetSimpleEvaluator(simpleEval)
	sc.endgameSolver.SetPruningDisabled(disablePruning)

	showMessage(sc.game.ToDisplayText(), sc.l.Stderr())

	val, seq, err := sc.endgameSolver.Solve(plies)
	if err != nil {
		sc.showError(err)
		return nil
	}
	// And turn off simulation mode again.
	sc.game.SetBackupMode(game.InteractiveGameplayMode)
	sc.showMessage(fmt.Sprintf("Best sequence has a spread difference of %v", val))
	sc.printEndgameSequence(seq)
	return nil
}

func (sc *ShellController) help(cmd *shellcmd) error {
	if cmd.args == nil {
		usage(sc.l.Stderr(), "standard", sc.execPath)
	} else {
		helptopic := cmd.args[0]
		usageTopic(sc.l.Stderr(), helptopic, sc.execPath)
	}
	return nil
}

func (sc *ShellController) setMode(cmd *shellcmd) error {
	mode := cmd.args[0]
	m, err := modeFromStr(mode)
	if err != nil {
		sc.showError(err)
		return err
	}
	sc.showMessage("Setting current mode to " + mode)
	sc.curMode = m
	return nil
}

func (sc *ShellController) export(cmd *shellcmd) error {
	if cmd.args == nil {
		return errors.New("please provide a filename to save to")
	}
	filename := cmd.args[0]
	contents, err := gcgio.GameHistoryToGCG(sc.game.History(), true)
	if err != nil {
		return err
	}
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	log.Debug().Interface("game-history", sc.game.History()).Msg("converted game history to gcg")
	f.WriteString(contents)
	f.Close()
	sc.showMessage("gcg written to " + filename)
	return nil
}

func (sc *ShellController) autoAnalyze(cmd *shellcmd) error {
	if cmd.args == nil {
		return errors.New("please provide a filename to analyze")
	}
	filename := cmd.args[0]
	analysis, err := automatic.AnalyzeLogFile(filename)
	if err != nil {
		return err
	}
	sc.showMessage(analysis)
	return nil
}

func (sc *ShellController) setChallengeRule(cmd *shellcmd) error {
	if cmd.args == nil {
		return errors.New("need rule")
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
	return nil
}
