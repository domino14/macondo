package shell

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/rs/zerolog/log"
)

func (sc *ShellController) handleSim(args []string) error {
	var plies, threads int
	var err error
	if sc.simmer == nil {
		return errors.New("load a game or something")
	}
	// Determine whether the first argument is a string or not.
	if len(args) > 0 {
		plies, err = strconv.Atoi(args[0])
		if err != nil {
			// It was not able to be converted to an integer!
			return sc.simControlArguments(args)
		}
	}

	// Otherwise, this is a command to start a simulation from scratch.

	switch {
	case len(args) == 0:
		plies = 2

	case len(args) == 2:
		threads, err = strconv.Atoi(args[1])
		if err != nil {
			return err
		}
		sc.simmer.SetThreads(threads)

	case len(args) == 1:
		// This is definitely a number, otherwise we would have failed
		// this conversion up above. Do nothing; we already have the
		// number of plies.
	default:
		return errors.New("unhandled arguments")
	}

	if len(sc.curPlayList) == 0 {
		return errors.New("please generate some plays first")
	}
	if sc.simmer.IsSimming() {
		return errors.New("simming already, please do a `sim stop` first")
	}
	log.Debug().Int("plies", plies).Int("threads", threads).Msg("will start sim")

	if sc.game != nil {
		err := sc.simmer.PrepareSim(plies, sc.curPlayList)
		if err != nil {
			return err
		}
		sc.startSim()
	}
	return nil
}

func (sc *ShellController) startSim() {
	sc.simCtx, sc.simCancel = context.WithCancel(context.Background())
	sc.simTicker = time.NewTicker(15 * time.Second)
	sc.simTickerDone = make(chan bool)
	sc.showMessage("Simulation started. Please do `sim show` and `sim details` to see more info")

	go func() {
		err := sc.simmer.Simulate(sc.simCtx)
		if err != nil {
			sc.showError(err)
			sc.simTickerDone <- true
		}
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
