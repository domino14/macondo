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

// type Job struct{}

// type Worker struct {
// 	id         int
// 	jobQueue   chan Job
// 	workerPool chan chan Job
// 	quitChan   chan bool
// 	logChan    chan string
// 	gd         *gaddag.SimpleGaddag
// }

// func NewWorker(id int, workerPool chan chan Job, gd *gaddag.SimpleGaddag,
// 	logchan chan string) Worker {
// 	return Worker{
// 		id:         id,
// 		jobQueue:   make(chan Job),
// 		workerPool: workerPool,
// 		quitChan:   make(chan bool),
// 		gd:         gd,
// 		logChan:    logchan,
// 	}
// }

// func (w Worker) start() {
// 	go func() {
// 		game := XWordGame{logchan: w.logChan}
// 		game.Init(w.gd)

// 		for {
// 			// Add my jobQueue to the worker pool.
// 			w.workerPool <- w.jobQueue

// 			select {
// 			case job := <-w.jobQueue:
// 				// Dispatcher has added a job to my jobQueue.
// 				game.playFullStatic()
// 			case <-w.quitChan:
// 				// We have been asked to stop.
// 				fmt.Printf("worker%d stopping\n", w.id)
// 				return
// 			}
// 		}
// 	}()
// }

// func (w Worker) stop() {
// 	go func() {
// 		w.quitChan <- true
// 	}()
// }

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

	// logChan := make(chan string, 100)
	// var wg sync.WaitGroup

	// worker := func(jobChan <-chan struct{}, id int) {
	// 	defer func() {
	// 		IsPlaying.Add(-1)
	// 		defer wg.Done()
	// 		log.Printf("Exiting worker %v goroutine!", id)
	// 	}()
	// 	IsPlaying.Add(1)
	// 	wg.Add(1)
	// 	game := XWordGame{logchan: logChan, gamechan: nil}
	// 	game.Init(gd)

	// 	for range jobChan {
	// 		game.playFullStatic()
	// 		// log.Printf("[DEBUG] Game over. Score: %v - %v",
	// 		// 	game.players[0].points,
	// 		// 	game.players[1].points)
	// 		CVCCounter.Add(1)
	// 	}
	// }

	// go func() {
	// 	for i := 1; i < numGames+1; i++ {
	// 		jobChan <- struct{}{}
	// 		if i%1000 == 0 {
	// 			log.Printf("%v games played...", i)
	// 		}
	// 	}
	// 	close(jobChan)
	// 	wg.Wait()
	// 	// Wait until all games are done, and then close the log channel.
	// 	close(logChan)
	// 	log.Printf("Exiting feeder goroutine!")
	// }()

	// for i := 0; i < threads; i++ {
	// 	go worker(jobChan, i)
	// }

	// return nil
}
