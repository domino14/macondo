// Package montecarlo implements truncated monte-carlo search
// during the regular game. In other words, "simming".
package montecarlo

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/rs/zerolog/log"
	"gopkg.in/yaml.v2"
)

/*
	How to simulate:

	For iteration in iterations:
		For play in plays:
			place on the board, keep track of leave
			shuffle bag
			for ply in plies:
				- generate rack for user on turn. tiles should be drawn
				in the same order from the bag and replaced. basically,
				this rack should be the same for every play in plays, so
				constant on a per-iteration basis, to make
				it easier to debug / minimize number of variables.
				- place highest valuation play on board, keep track of leave

			compute stats so far

*/

// LogIteration is a struct meant for serializing to a log-file, for debug
// and other purposes.
type LogIteration struct {
	Iteration int       `json:"iteration" yaml:"iteration"`
	Plays     []LogPlay `json:"plays" yaml:"plays"`
	Thread    int       `json:"thread" yaml:"thread"`
}

// LogPlay is a single play.
type LogPlay struct {
	Play string `json:"play" yaml:"play"`
	Rack string `json:"rack" yaml:"rack"`
	Pts  int    `json:"pts" yaml:"pts"`
	// Leftover is the equity of the leftover tiles at the end of the sim.
	Leftover float64 `json:"left,omitempty" yaml:"left,omitempty"`
	// Although this is a recursive structure we don't really use it
	// recursively.
	Plies []LogPlay `json:"plies,omitempty" yaml:"plies,omitempty,flow"`
}

type SimmedPlay struct {
	sync.Mutex
	play          *move.Move
	scoreStats    []Statistic
	bingoStats    []Statistic
	equityStats   Statistic
	leftoverStats Statistic
}

func (sp *SimmedPlay) String() string {
	return fmt.Sprintf("<Simmed play: %v (stats: %v %v %v %v)>", sp.play.ShortDescription(),
		sp.scoreStats, sp.bingoStats, sp.equityStats, sp.leftoverStats)
}

func (sp *SimmedPlay) addScoreStat(play *move.Move, ply int) {
	// log.Debug().Msgf("Adding a stat for %v (pidx %v ply %v)", play, pidx, ply)
	var bingos int
	if play.TilesPlayed() == 7 {
		bingos = 1
	}
	sp.Lock()
	defer sp.Unlock()
	sp.scoreStats[ply].Push(float64(play.Score()))
	sp.bingoStats[ply].Push(float64(bingos))
}

func (sp *SimmedPlay) addEquityStat(spread int, leftover float64) {
	sp.equityStats.Push(float64(spread) + leftover)
	sp.leftoverStats.Push(leftover)
}

// Simmer implements the actual look-ahead search
type Simmer struct {
	origGame *game.Game

	gameCopies []*game.Game
	movegens   []movegen.MoveGenerator

	aiplayer player.AIPlayer

	initialSpread int
	maxPlies      int
	// initialPlayer is the player for whom we are simming.
	initialPlayer  int
	iterationCount int
	threads        int

	simming    bool
	readyToSim bool
	plays      []*SimmedPlay

	logStream io.Writer
}

func (s *Simmer) Init(game *game.Game, aiplayer player.AIPlayer) {
	s.origGame = game
	s.aiplayer = aiplayer
	s.threads = int(math.Max(1, float64(runtime.NumCPU()-1)))
}

func (s *Simmer) SetThreads(threads int) {
	s.threads = threads
}

func (s *Simmer) SetLogStream(l io.Writer) {
	s.logStream = l
}

func (s *Simmer) makeGameCopies() error {
	log.Debug().Int("threads", s.threads).Msg("makeGameCopies")
	s.gameCopies = []*game.Game{}
	s.movegens = []movegen.MoveGenerator{}

	gd, err := gaddag.Get(s.origGame.Config(), s.origGame.LexiconName())
	if err != nil {
		return err
	}

	for i := 0; i < s.threads; i++ {
		s.gameCopies = append(s.gameCopies, s.origGame.Copy())
		s.movegens = append(s.movegens,
			movegen.NewGordonGenerator(gd, s.gameCopies[i].Board(),
				s.gameCopies[i].Bag().LetterDistribution()))

	}
	return nil

}

func (s *Simmer) resetStats(plies int, plays []*move.Move) {
	s.iterationCount = 0
	s.maxPlies = plies
	for _, g := range s.gameCopies {
		g.SetStateStackLength(1)
	}
	s.initialSpread = s.gameCopies[0].CurrentSpread()
	s.initialPlayer = s.gameCopies[0].PlayerOnTurn()
	s.plays = make([]*SimmedPlay, len(plays))
	for idx, play := range plays {
		s.plays[idx] = &SimmedPlay{}
		s.plays[idx].play = play
		s.plays[idx].scoreStats = make([]Statistic, plies)
		s.plays[idx].bingoStats = make([]Statistic, plies)
	}

}

func (s *Simmer) IsSimming() bool {
	return s.simming
}

func (s *Simmer) Reset() {
	s.plays = nil
	s.gameCopies = nil
	s.readyToSim = false
}

// PrepareSim resets all the stats before a simulation.
func (s *Simmer) PrepareSim(plies int, plays []*move.Move) error {
	err := s.makeGameCopies()
	if err != nil {
		return err
	}
	s.resetStats(plies, plays)
	s.readyToSim = true
	return nil
}

func (s *Simmer) Ready() bool {
	return s.readyToSim
}

// Simulate sims all the plays. It is a blocking function.
func (s *Simmer) Simulate(ctx context.Context) error {
	if len(s.plays) == 0 || len(s.gameCopies) == 0 {
		return errors.New("please prepare the simulation first")
	}

	s.simming = true
	defer func() {
		s.simming = false
		log.Info().Msgf("Simulation ended after %v iterations", s.iterationCount)
	}()

	// use an errgroup here and listen for a ctx done outside this loop, but
	// in another goroutine.
	// protect the simmed play statistics with a mutex.
	log.Debug().Msgf("Simulating with %v threads", s.threads)
	syncChan := make(chan bool, s.threads)
	logChan := make(chan []byte)
	done := make(chan bool)

	ctrl := errgroup.Group{}
	writer := errgroup.Group{}

	ctrl.Go(func() error {
		defer func() {
			log.Debug().Msgf("Sim controller thread exiting")
		}()
		for {
			select {
			case <-ctx.Done():
				log.Debug().Msgf("Context is done: %v", ctx.Err())
				for t := 0; t < s.threads; t++ {
					syncChan <- true
				}
				log.Debug().Msgf("Sent sync messages to children threads...")
				return ctx.Err()
			default:
				// Do nothing
			}
		}
	})

	if s.logStream != nil {

		writer.Go(func() error {
			defer func() {
				log.Debug().Msgf("Writer routine exiting")
			}()
			for {
				select {
				case bytes := <-logChan:
					s.logStream.Write(bytes)
				case <-done:
					// Ok, actually quit now.
					log.Debug().Msgf("Got quit signal...")
					return nil
				}
			}
		})
	}

	var iterMutex sync.Mutex
	g := errgroup.Group{}
	for t := 0; t < s.threads; t++ {
		t := t
		g.Go(func() error {
			defer func() {
				log.Debug().Msgf("Thread %v exiting sim", t)
			}()
			log.Debug().Msgf("Thread %v starting sim", t)
			for {

				iterMutex.Lock()
				iterNum := s.iterationCount + 1
				s.iterationCount++
				iterMutex.Unlock()

				s.simSingleIteration(s.maxPlies, t, iterNum, logChan)
				select {
				case v := <-syncChan:
					log.Debug().Msgf("Thread %v got sync msg %v", t, v)
					return nil
				default:
					// Do nothing
				}
			}
		})
	}

	// Wait for threads in errgroup:
	err := g.Wait()
	log.Debug().Msgf("errgroup returned err %v", err)

	// Writer thread will exit now:
	if s.logStream != nil {
		close(done)
		writer.Wait()
	}

	ctrlErr := ctrl.Wait()
	log.Debug().Msgf("ctrl errgroup returned err %v", ctrlErr)
	return ctrlErr
}

func (s *Simmer) Iterations() int {
	return s.iterationCount
}

func (s *Simmer) TrimBottom(totrim int) error {
	if s.simming {
		return errors.New("please stop sim before trimming plays")
	}
	if totrim > len(s.plays)-1 {
		return errors.New("there are not that many plays to trim away")
	}
	s.plays = s.plays[:len(s.plays)-totrim]
	return nil
}

func (s *Simmer) simSingleIteration(plies, thread, iterationCount int, logChan chan []byte) {
	// Give opponent a random rack from the bag. Note that this also
	// shuffles the bag!
	opp := (s.initialPlayer + 1) % s.gameCopies[thread].NumPlayers()
	s.gameCopies[thread].SetRandomRack(opp)
	logIter := LogIteration{Iteration: iterationCount, Plays: []LogPlay{}, Thread: thread}

	var logPlay LogPlay
	var plyChild LogPlay

	for _, simmedPlay := range s.plays {
		if s.logStream != nil {
			logPlay = LogPlay{Play: simmedPlay.play.ShortDescription(),
				Rack: simmedPlay.play.FullRack(),
				Pts:  simmedPlay.play.Score()}
		}
		// equity of the leftover tiles at the end of the sim
		leftover := float64(0.0)
		// logIter.Plays = append(logIter.Plays)
		// Play the move, and back up the game state.
		// log.Debug().Msgf("Playing move %v", play)'
		// Set the backup mode to simulation mode only to back up the first move:
		s.gameCopies[thread].SetBackupMode(game.SimulationMode)
		s.gameCopies[thread].PlayMove(simmedPlay.play, false, 0)
		s.gameCopies[thread].SetBackupMode(game.NoBackup)
		// Further plies will NOT be backed up.
		for ply := 0; ply < plies; ply++ {
			// Each ply is a player taking a turn
			onTurn := s.gameCopies[thread].PlayerOnTurn()
			if s.gameCopies[thread].Playing() == pb.PlayState_PLAYING {
				// Assume there are exactly two players.

				bestPlay := s.bestStaticTurn(onTurn, thread)
				// log.Debug().Msgf("Ply %v, Best play: %v", ply+1, bestPlay)
				s.gameCopies[thread].PlayMove(bestPlay, false, 0)
				// log.Debug().Msgf("Score is now %v", s.game.Score())
				if s.logStream != nil {
					plyChild = LogPlay{Play: bestPlay.ShortDescription(), Rack: bestPlay.FullRack(), Pts: bestPlay.Score()}
				}
				if ply == plies-2 || ply == plies-1 {
					// It's either OUR last turn or OPP's last turn.
					// Calculate equity of leftover tiles.
					thisLeftover := s.aiplayer.Strategizer().LeaveValue(bestPlay.Leave())
					if s.logStream != nil {
						plyChild.Leftover = thisLeftover
					}

					// log.Debug().Msgf("Calculated leftover %v", plyChild.Leftover)
					if onTurn == s.initialPlayer {
						leftover += thisLeftover
					} else {
						leftover -= thisLeftover
					}
				}

				logPlay.Plies = append(logPlay.Plies, plyChild)
				simmedPlay.addScoreStat(bestPlay, ply)
			}
		}
		// log.Debug().Msgf("Spread for initial player: %v, leftover: %v",
		// 	s.game.SpreadFor(s.initialPlayer), leftover)
		simmedPlay.addEquityStat(s.gameCopies[thread].SpreadFor(s.initialPlayer)-s.initialSpread, leftover)
		s.gameCopies[thread].ResetToFirstState()
		logIter.Plays = append(logIter.Plays, logPlay)
	}
	if s.logStream != nil {
		out, err := yaml.Marshal([]LogIteration{logIter})
		if err != nil {
			log.Error().Err(err).Msg("marshalling log")
			return
		}
		logChan <- out
	}
}

func (s *Simmer) bestStaticTurn(playerID, thread int) *move.Move {
	return player.GenBestStaticTurn(s.gameCopies[thread], s.movegens[thread], s.aiplayer, playerID)
}

func (s *Simmer) sortPlaysByEquity() {
	// Sort by equity
	// log.Debug().Msgf("Sorting plays: %v", s.plays)
	sort.Slice(s.plays, func(i, j int) bool {
		return s.plays[i].equityStats.Mean() > s.plays[j].equityStats.Mean()
	})
}

func (s *Simmer) printStats() string {
	return s.EquityStats() + "\n Details per play \n" + s.ScoreDetails()
}

func (s *Simmer) EquityStats() string {
	stats := ""

	s.sortPlaysByEquity()
	stats += fmt.Sprintf("%20v%6v%8v\n", "Play", "Score", "Equity")

	for _, play := range s.plays {
		stats += fmt.Sprintf("%20v%6d%8.3f\n", play.play.ShortDescription(),
			play.play.Score(), play.equityStats.Mean())
	}
	stats += fmt.Sprintf("Iterations: %v\n", s.iterationCount)
	return stats
}

func (s *Simmer) ScoreDetails() string {
	stats := ""
	s.sortPlaysByEquity()
	for ply := 0; ply < s.maxPlies; ply++ {
		who := "You"
		if ply%2 == 0 {
			who = "Opponent"
		}
		stats += fmt.Sprintf("**Ply %v (%v)**\n%20v%8v%8v%8v\n%v\n",
			ply+1, who, "Play", "Mean", "Stdev", "Bingo %", strings.Repeat("-", 44))
		for _, play := range s.plays {
			stats += fmt.Sprintf("%20v%8.3f%8.3f%8.3f\n",
				play.play.ShortDescription(), play.scoreStats[ply].Mean(),
				play.scoreStats[ply].Stdev(),
				100.0*play.bingoStats[ply].Mean())
		}
		stats += "\n"
	}
	stats += fmt.Sprintf("Iterations: %v\n", s.iterationCount)

	return stats
}
