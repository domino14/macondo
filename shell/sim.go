package shell

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/montecarlo"
)

func (sc *ShellController) handleSim(args []string, options map[string]string) error {
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
	for opt, val := range options {
		switch opt {
		case "plies":
			plies, err = strconv.Atoi(val)
			if err != nil {
				return err
			}
		case "threads":
			threads, err = strconv.Atoi(val)
			if err != nil {
				return err
			}
		case "stop":
			sci, err := strconv.Atoi(val)
			if err != nil {
				return err
			}
			switch sci {
			case 95:
				stoppingCondition = montecarlo.Stop95
			case 98:
				stoppingCondition = montecarlo.Stop98
			case 99:
				stoppingCondition = montecarlo.Stop99
			default:
				return errors.New("only allowed values are 95, 98, and 99 for stopping condition")
			}
		case "opprack":
			knownOppRack = val

		case "useinferences":
			inferences := sc.rangefinder.Inferences()
			if len(inferences) == 0 {
				return errors.New("you must run `infer` first")
			}
			switch val {
			case "cycle":
				inferMode = montecarlo.InferenceCycle
				sc.showMessage(fmt.Sprintf(
					"Set inference mode to 'cycle' with %d inferences", len(inferences)))
			case "random":
				inferMode = montecarlo.InferenceRandom
				sc.showMessage(fmt.Sprintf(
					"Set inference mode to 'random' with %d inferences", len(inferences)))

			default:
				return errors.New("that inference mode is not supported")
			}

		default:
			return errors.New("option " + opt + " not recognized")
		}
	}
	if plies == 0 {
		plies = 2
	}

	log.Debug().Int("plies", plies).Int("threads", threads).
		Int("stoppingCondition", int(stoppingCondition)).Msg("will start sim")

	if sc.game != nil {
		if threads != 0 {
			sc.simmer.SetThreads(threads)
		}
		err := sc.simmer.PrepareSim(plies, sc.curPlayList)
		if err != nil {
			return err
		}
		sc.simmer.SetStoppingCondition(stoppingCondition)

		if knownOppRack != "" {
			knownOppRack = strings.ToUpper(knownOppRack)
			r, err := tilemapping.ToMachineLetters(knownOppRack, sc.game.Alphabet())
			if err != nil {
				return err
			}
			sc.simmer.SetKnownOppRack(r)
		}
		if inferMode != montecarlo.InferenceOff {
			sc.simmer.SetInferences(sc.rangefinder.Inferences(), inferMode)
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
	default:
		return fmt.Errorf("do not understand sim argument %v", args[0])
	}

	return nil
}
