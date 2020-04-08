package automatic

// Data collection for automatic game. Allow computer vs computer games, etc.

import (
	"context"
	"errors"
	"expvar"
	"os"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
)

var (
	CVCCounter *expvar.Int
	IsPlaying  *expvar.Int
)

func init() {
	CVCCounter = expvar.NewInt("cvcCounter")
	IsPlaying = expvar.NewInt("isPlaying")
}

// CompVsCompStatic plays out a game to the end using best static turns.
func (r *GameRunner) CompVsCompStatic(gd *gaddag.SimpleGaddag) {
	r.Init(gd)
	r.playFullStatic()
	log.Debug().Msgf("Game over. Score: %v - %v", r.game.PointsFor(0),
		r.game.PointsFor(1))
}

func (r *GameRunner) playFullStatic() {
	r.StartGame()
	for r.game.Playing() {
		// log.Printf("[DEBUG] turn %v", r.game.Turn())
		r.PlayBestStaticTurn(r.game.PlayerOnTurn())
	}
	if r.gamechan != nil {
		r.gamechan <- r.game.Board().ToDisplayText(r.game.Alphabet())
	}
}

type Job struct{}

func StartCompVCompStaticGames(ctx context.Context, cfg *config.Config,
	gd *gaddag.SimpleGaddag, numGames int, threads int, outputFilename string) error {

	if IsPlaying.Value() > 0 {
		return errors.New("games are already being played, please wait till complete")
	}

	logfile, err := os.Create(outputFilename)
	if err != nil {
		return err
	}
	log.Debug().Msgf("Starting %v games, %v threads", numGames, threads)

	CVCCounter.Set(0)
	jobs := make(chan Job, 100)
	logChan := make(chan string, 100)
	var wg sync.WaitGroup
	wg.Add(threads)

	for i := 1; i <= threads; i++ {
		go func(i int) {
			defer wg.Done()
			r := GameRunner{logchan: logChan, config: cfg}
			r.Init(gd)
			IsPlaying.Add(1)
			for range jobs {
				r.playFullStatic()
				CVCCounter.Add(1)
			}
			IsPlaying.Add(-1)
		}(i)
	}

	go func() {
	gameLoop:
		for i := 1; i < numGames+1; i++ {
			jobs <- Job{}
			if i%1000 == 0 {
				log.Printf("Queued %v jobs", i)
			}
			select {
			case <-ctx.Done():
				// exit early
				log.Info().Msg("Got stop signal, exiting soon...")
				break gameLoop
			default:
				// do nothing

			}
		}

		close(jobs)
		log.Info().Msg("Finished queueing all jobs.")
		wg.Wait()
		log.Info().Msg("All games finished.")
		close(logChan)
		log.Info().Msg("Exiting feeder subroutine!")
	}()

	go func() {
		logfile.WriteString("playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining\n")
		for msg := range logChan {
			logfile.WriteString(msg)
		}
		logfile.Close()
		log.Info().Msg("Exiting turn logger goroutine!")
	}()

	return nil

}
