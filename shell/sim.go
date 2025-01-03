package shell

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/aybabtme/uniplot/histogram"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
)

func (sc *ShellController) handleSim(args []string, options CmdOptions) error {
	var plies, threads int
	var err error
	stoppingCondition := montecarlo.StopNone
	if sc.simmer == nil {
		return errors.New("load a game or something")
	}

	if len(args) > 0 {
		return sc.simControlArguments(args)
	}

	if len(sc.curPlayList) == 0 {
		return errors.New("please generate some plays first")
	}
	if sc.simmer.IsSimming() {
		return errors.New("simming already, please do a `sim stop` first")
	}
	if sc.solving() {
		return errMacondoSolving
	}

	inferMode := montecarlo.InferenceOff
	knownOppRack := ""
	var stopPPscaling, stopitercutoff int
	for opt := range options {
		switch opt {
		case "plies":
			plies, err = options.Int(opt)
			if err != nil {
				return err
			}
		case "threads":
			threads, err = options.Int(opt)
			if err != nil {
				return err
			}
		case "stop":
			sci, err := options.Int(opt)
			if err != nil {
				return err
			}
			switch sci {
			case 90:
				stoppingCondition = montecarlo.Stop90
			case 95:
				stoppingCondition = montecarlo.Stop95
			case 98:
				stoppingCondition = montecarlo.Stop98
			case 99:
				stoppingCondition = montecarlo.Stop99
			case 999:
				stoppingCondition = montecarlo.Stop999
			default:
				return errors.New("only allowed values are 90, 95, 98, 99, and 999 for stopping condition")
			}
		case "stop-ppscaling":
			stopPPscaling, err = options.Int(opt)
			if err != nil {
				return err
			}

		case "stop-itercutoff":
			stopitercutoff, err = options.Int(opt)
			if err != nil {
				return err
			}

		case "opprack":
			knownOppRack = options.String(opt)

		case "useinferences":
			inferences := sc.rangefinder.Inferences()
			if inferences == nil || len(inferences.InferredRacks) == 0 {
				return errors.New("you must run `infer` first")
			}
			switch options.String(opt) {
			case "weightedrandom":
				inferMode = montecarlo.InferenceWeightedRandom
				sc.showMessage(fmt.Sprintf(
					"Set inference mode to 'weightedrandom' with %d inferences", len(inferences.InferredRacks)))
			default:
				return errors.New("that inference mode is not supported")
			}
		case "collect-heatmap":
			if options.Bool(opt) {
				err = sc.simmer.SetCollectHeatmap(true)
			} else {
				err = sc.simmer.SetCollectHeatmap(false)
			}
			if err != nil {
				return err
			}

		default:
			return errors.New("option " + opt + " not recognized")
		}
	}
	if plies == 0 {
		plies = 2
	}

	log.Debug().Int("plies", plies).Int("threads", threads).
		Int("stoppingCondition", int(stoppingCondition)).
		Int("ppscaling", stopPPscaling).
		Int("itercutoff", stopitercutoff).
		Msg("will start sim")

	if sc.game != nil {
		if threads != 0 {
			sc.simmer.SetThreads(threads)
		}
		err := sc.simmer.PrepareSim(plies, sc.curPlayList)
		if err != nil {
			return err
		}
		sc.simmer.SetStoppingCondition(stoppingCondition)
		if stopPPscaling > 0 {
			sc.simmer.SetAutostopPPScaling(stopPPscaling)
		}
		if stopitercutoff > 0 {
			sc.simmer.SetAutostopIterationsCutoff(stopitercutoff)
		}

		if knownOppRack != "" {
			knownOppRack = strings.ToUpper(knownOppRack)
			r, err := tilemapping.ToMachineLetters(knownOppRack, sc.game.Alphabet())
			if err != nil {
				return err
			}
			sc.simmer.SetKnownOppRack(r)
		}
		if inferMode != montecarlo.InferenceOff {
			sc.simmer.SetInferences(sc.rangefinder.Inferences().InferredRacks,
				sc.rangefinder.Inferences().RackLength,
				inferMode)
		}
		sc.startSim()
	}
	return nil
}

func (sc *ShellController) startSim() {
	sc.simCtx, sc.simCancel = context.WithCancel(context.Background())
	sc.simTicker = time.NewTicker(10 * time.Second)
	sc.simTickerDone = make(chan bool)
	sc.showMessage("Simulation started. Please do `sim show` and `sim details` to see more info")

	go func() {
		err := sc.simmer.Simulate(sc.simCtx)
		if err != nil {
			sc.showError(err)
		}
		sc.simTickerDone <- true
		log.Debug().Msg("simulation thread exiting...")
	}()

	go func() {
		for {
			select {
			case <-sc.simTickerDone:
				log.Debug().Msg("ticker thread exiting...")
				return
			case <-sc.simTicker.C:
				log.Info().Msgf("Simmer is at %v iterations...",
					sc.simmer.Iterations())
			}
		}
	}()
}

func (sc *ShellController) simControlArguments(args []string) error {
	var err error
	switch args[0] {
	case "log":
		if sc.simmer.IsSimming() {
			return errors.New("please stop sim before making any log changes")
		}
		sc.simLogFile, err = os.Create(SimLog)
		if err != nil {
			return err
		}
		sc.simmer.SetLogStream(sc.simLogFile)
		sc.showMessage("sim will log to " + SimLog)
	case "stop":
		if !sc.simmer.IsSimming() {
			return errors.New("no running sim to stop")
		}
		sc.simTicker.Stop()
		sc.simTickerDone <- true
		sc.simCancel()
		if sc.simLogFile != nil {
			err := sc.simLogFile.Close()
			if err != nil {
				return err
			}
		}
		// Show the results.
		sc.showMessage(sc.simmer.EquityStats())
	case "details":
		sc.showMessage(sc.simmer.ScoreDetails())
	case "show":
		sc.showMessage(sc.simmer.EquityStats())
	case "winner":
		sc.showMessage(sc.simmer.WinningPlay().Move().ShortDescription())
	case "continue":
		if sc.simmer.IsSimming() {
			return errors.New("there is an ongoing simulation")
		}
		if !sc.simmer.Ready() {
			return errors.New("simmer is not ready; please generate some plays and `sim`")
		}
		if sc.solving() {
			return errMacondoSolving
		}
		// No need to prepare sim. startSim will continue from where it left off.
		sc.startSim()
	case "trim":
		if len(args) != 2 {
			return errors.New("trim needs an argument")
		}
		totrim, err := strconv.Atoi(args[1])
		if err != nil {
			return err
		}
		err = sc.simmer.TrimBottom(totrim)
		if err != nil {
			return err
		}
		sc.showMessage(sc.simmer.EquityStats())
	case "heatmap":
		if len(args) < 2 {
			return errors.New("heatmap needs the play in quotes. See `help sim`")
		}
		play := args[1]
		ply := 0
		if len(args) == 3 {
			ply, err = strconv.Atoi(args[2])
			if err != nil {
				return err
			}
		}

		heatmap, err := sc.simStats.CalculateHeatmap(play, ply)
		if err != nil {
			return err
		}

		// Display with the tiles of our move.
		err = sc.placeMove(sc.game, play)
		if err != nil {
			return err
		}

		heatmap.Display()
		sc.unplaceMove(sc.game)

	case "playstats":
		if len(args) < 2 {
			return errors.New("playstats needs the play in quotes. See `help sim`")
		}
		play := args[1]
		stats, err := sc.simStats.CalculatePlayStats(play)
		if err != nil {
			return err
		}
		sc.showMessage(stats)
		if len(args) == 3 && args[2] == "histogram" {
			oppHist, ourHist := sc.simStats.LastHistogram()
			maxWidth := 40
			fmt.Println("Opponent score histogram")
			fmt.Println("Points")
			err = histogram.Fprint(os.Stdout, oppHist, histogram.Linear(maxWidth))
			fmt.Println("\n\nOur score histogram")
			fmt.Println("Points")
			err = histogram.Fprint(os.Stdout, ourHist, histogram.Linear(maxWidth))
		}

	case "tilestats":
		stats, err := sc.simStats.CalculateTileStats()
		if err != nil {
			return err
		}
		sc.showMessage(stats)
	default:
		return fmt.Errorf("do not understand sim argument %v", args[0])
	}

	return nil
}

func (sc *ShellController) placeMove(g *bot.BotTurnPlayer, play string) error {
	normalizedPlay := stats.Normalize(play)
	m, err := g.ParseMove(
		g.PlayerOnTurn(), sc.options.lowercaseMoves, strings.Fields(normalizedPlay))

	if err != nil {
		return err
	}
	g.SetBackupMode(game.SimulationMode)
	log.Debug().Str("move", m.ShortDescription()).Msg("Playing move")
	err = g.PlayMove(m, false, 0)
	return err
}

func (sc *ShellController) unplaceMove(g *bot.BotTurnPlayer) {
	log.Debug().Msg("Undoing last move")
	g.UnplayLastMove()
	g.SetBackupMode(game.NoBackup)
}
