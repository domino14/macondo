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

	"github.com/aybabtme/uniplot/histogram"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
)

func (sc *ShellController) handleSim(args []string, options CmdOptions) error {
	var plies, threads, fixedplies, fixediters, fixedsimcount int
	var err error
	stoppingCondition := montecarlo.StopNone
	if sc.simmer == nil {
		return errors.New("load a game or something")
	}
	if sc.game == nil {
		return errors.New("game does not exist")
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
	var stopPPscaling, stopitercutoff, stopcheckinterval int
	avoidTrimMoveStrs := options.StringArray("avoid-prune")
	avoidTrimMoves := []*move.Move{}

	for opt := range options {
		switch opt {
		case "avoid-prune":
			for _, ms := range avoidTrimMoveStrs {
				m, err := sc.game.ParseMove(
					sc.game.PlayerOnTurn(),
					sc.options.lowercaseMoves,
					strings.Fields(ms),
				)
				if err != nil {
					return err
				}
				avoidTrimMoves = append(avoidTrimMoves, m)
			}
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
		case "fixedsimiters":
			// single thread, fixed iters, fixed plies
			fixediters, err = options.Int(opt)
			if err != nil {
				return err
			}
		case "fixedsimplies":
			// single thread, fixed iters, fixed plies
			fixedplies, err = options.Int(opt)
			if err != nil {
				return err
			}
		case "fixedsimcount":
			// how many single-thread sims to do
			fixedsimcount, err = options.Int(opt)
			if err != nil {
				return err
			}

		case "autostopcheckinterval":
			stopcheckinterval, err = options.Int(opt)
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
			case "weightedrandomtiles":
				inferMode = montecarlo.InferenceWeightedRandomTiles
				sc.showMessage(fmt.Sprintf(
					"Set inference mode to 'weightedrandomtiles' with %d inferences", len(inferences.InferredRacks)))
			case "weightedrandomracks":
				inferMode = montecarlo.InferenceWeightedRandomRacks
				sc.showMessage(fmt.Sprintf(
					"Set inference mode to 'weightedrandomracks' with %d inferences", len(inferences.InferredRacks)))

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

	var kr []tilemapping.MachineLetter

	if knownOppRack != "" {
		knownOppRack = strings.ToUpper(knownOppRack)
		kr, err = tilemapping.ToMachineLetters(knownOppRack, sc.game.Alphabet())
		if err != nil {
			return err
		}
	}

	params := simParams{
		threads:               threads,
		plies:                 plies,
		stoppingCondition:     stoppingCondition,
		autostopPPScaling:     stopPPscaling,
		autostopIterCutoff:    stopitercutoff,
		autostopCheckInterval: stopcheckinterval,
		knownOppRack:          kr,
		inferMode:             inferMode,
		avoidTrimMoves:        avoidTrimMoves,
	}
	if fixediters != 0 {
		return sc.startMultiSimExperiment(fixediters, fixedplies, fixedsimcount, params)
	}

	sc.setSimmerParams(sc.simmer, params)
	sc.startSim()

	return nil
}

type simParams struct {
	threads               int
	plies                 int
	stoppingCondition     montecarlo.StoppingCondition
	autostopPPScaling     int
	autostopIterCutoff    int
	autostopCheckInterval int
	knownOppRack          []tilemapping.MachineLetter
	inferMode             montecarlo.InferenceMode
	avoidTrimMoves        []*move.Move
}

func (sc *ShellController) setSimmerParams(simmer *montecarlo.Simmer, params simParams) error {
	if params.threads != 0 {
		simmer.SetThreads(params.threads)
	}
	err := simmer.PrepareSim(params.plies, sc.curPlayList)
	if err != nil {
		return err
	}
	simmer.SetStoppingCondition(params.stoppingCondition)
	if params.autostopPPScaling > 0 {
		simmer.SetAutostopPPScaling(params.autostopPPScaling)
	}
	if params.autostopIterCutoff > 0 {
		simmer.SetAutostopIterationsCutoff(params.autostopIterCutoff)
	}
	if params.autostopCheckInterval > 0 {
		simmer.SetAutostopCheckInterval(uint64(params.autostopCheckInterval))
	}

	if params.inferMode != montecarlo.InferenceOff {
		simmer.SetInferences(sc.rangefinder.Inferences().InferredRacks,
			sc.rangefinder.Inferences().RackLength,
			params.inferMode)
	}
	if len(params.avoidTrimMoves) > 0 {
		simmer.AvoidPruningMoves(params.avoidTrimMoves)
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

func (sc *ShellController) startMultiSimExperiment(iters, plies, simCount int, params simParams) error {
	sc.simCtx, sc.simCancel = context.WithCancel(context.Background())
	sc.simTicker = time.NewTicker(10 * time.Second)
	sc.simTickerDone = make(chan bool)
	sc.showMessage("Simulation experiments started.")

	threads := max(1, runtime.NumCPU())

	simmers := []*montecarlo.Simmer{}
	for i := 0; i < threads; i++ {
		mcsimmer := &montecarlo.Simmer{}

		c, err := equity.NewCombinedStaticCalculator(
			sc.game.LexiconName(),
			sc.config, "", equity.PEGAdjustmentFilename)
		if err != nil {
			return err
		}
		mcsimmer.Init(sc.game.Game, []equity.EquityCalculator{c}, c, sc.config)
		simmers = append(simmers, mcsimmer)
	}

	simChan := make(chan struct{}, 32)
	winningPlays := map[string]int{}
	go func() {

		g := errgroup.Group{}

		for i := 0; i < threads; i++ {

			g.Go(func() error {
				for range simChan {
					simmer := simmers[i]
					err := sc.setSimmerParams(simmer, params)
					if err != nil {
						return err
					}
					simmer.SimSingleThread(iters, plies)
					winningPlays[simmer.WinningPlay().Move().ShortDescription()]++
				}
				return nil
			})

		}
		go func() {
			for i := 0; i < simCount; i++ {
				simChan <- struct{}{}
				if i%100 == 0 {
					log.Info().Msgf("Simmer enqueued %d experiments...", i)
				}
			}
			close(simChan)
		}()
		log.Info().Msg("simulation feeder exiting...")
		err := g.Wait()
		if err != nil {
			log.Err(err).Msg("experiment-sim-errgroup")
		}
		fmt.Println("Winning plays:", winningPlays)
	}()

	return nil

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
