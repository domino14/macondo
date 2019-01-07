package xwordgame

// Data collection for XWordGame. Allow computer vs computer games, etc.

import (
	"errors"
	"expvar"
	"log"
	"os"
	"sync"

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
func (game *XWordGame) CompVsCompStatic(gd *gaddag.SimpleGaddag) {
	game.Init(gd)
	game.playFullStatic()
	log.Printf("[DEBUG] Game over. Score: %v - %v", game.players[0].points,
		game.players[1].points)
}

func (game *XWordGame) playFullStatic() {
	game.StartGame()
	for game.playing {
		game.PlayBestStaticTurn(game.onturn)
	}
	if game.gamechan != nil {
		game.gamechan <- game.board.ToDisplayText(game.gaddag.GetAlphabet())
	}
}

type Job struct{}

func StartCompVCompStaticGames(gd *gaddag.SimpleGaddag, numGames int, threads int,
	outputFilename string) error {

	if IsPlaying.Value() > 0 {
		return errors.New("games are already being played, please wait till complete")
	}

	logfile, err := os.Create(outputFilename)
	if err != nil {
		return err
	}

	CVCCounter.Set(0)
	jobs := make(chan Job, 100)
	logChan := make(chan string, 100)
	var wg sync.WaitGroup
	wg.Add(threads)

	for i := 1; i <= threads; i++ {
		go func(i int) {
			defer wg.Done()
			game := XWordGame{logchan: logChan}
			game.Init(gd)
			IsPlaying.Add(1)
			for range jobs {
				game.playFullStatic()
				CVCCounter.Add(1)
			}
			IsPlaying.Add(-1)
		}(i)
	}

	go func() {
		for i := 1; i < numGames+1; i++ {
			jobs <- Job{}
			if i%1000 == 0 {
				log.Printf("Queued %v jobs", i)
			}
		}
		close(jobs)
		log.Printf("Finished queueing all jobs.")
		wg.Wait()
		log.Printf("All games finished.")
		close(logChan)
		log.Printf("Exiting feeder subroutine!")
	}()

	go func() {
		for msg := range logChan {
			logfile.WriteString(msg)
		}
		logfile.Close()
		log.Printf("Exiting turn logger goroutine!")
	}()

	return nil

}
