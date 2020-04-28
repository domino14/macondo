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

	switch {
	case len(args) == 0:
		plies = 2

	case len(args) == 2:
		threads, err = strconv.Atoi(args[1])
		if err != nil {
			return err
		}
		sc.simmer.SetThreads(threads)
		fallthrough

	case len(args) == 1:
		// Either a number or a word.
		plies, err = strconv.Atoi(args[0])
		if err != nil {
			// it was probably a word.
			if len(args) != 1 {
				// this can happen because of the fallthrough above
				return errors.New("badly formatted sim command")
			}
			return sc.simControlArg(args[0])
		}
	}

	if len(sc.curPlayList) == 0 {
		return errors.New("please generate some plays first")
	}
	if sc.simmer.IsSimming() {
		return errors.New("simming already, please do a `sim stop` first")
	}

	if sc.game != nil {
		sc.simCtx, sc.simCancel = context.WithCancel(context.Background())
		sc.simTicker = time.NewTicker(15 * time.Second)
		sc.simTickerDone = make(chan bool)

		go sc.simmer.Simulate(sc.simCtx, sc.curPlayList, plies)

		go func() {
			for {
				select {
				case <-sc.simTickerDone:
					return
				case <-sc.simTicker.C:
					log.Info().Msgf("Simmer is at %v iterations...",
						sc.simmer.Iterations())
				}
			}
		}()

		sc.showMessage("Simulation started. Please do `sim show` and `sim details` to see more info")
	}
	return nil

}

func (sc *ShellController) simControlArg(arg string) error {
	var err error
	switch arg {
	case "log":
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
	case "details":
		sc.showMessage(sc.simmer.ScoreDetails())
	case "show":
		sc.showMessage(sc.simmer.EquityStats())

	default:
		return fmt.Errorf("do not understand sim argument %v", arg)
	}

	return nil
}
