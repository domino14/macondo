// Package montecarlo implements truncated monte-carlo search
// during the regular game. In other words, "simming".
package montecarlo

import (
	"context"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/mechanics"
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

// Statistic contains statistics per move
type Statistic struct {
	totalIterations int

	// For Welford's algorithm:
	oldM float64
	newM float64
	oldS float64
	newS float64
}

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

func (s *Statistic) push(val float64) {
	s.totalIterations++
	if s.totalIterations == 1 {
		s.oldM = val
		s.newM = val
		s.oldS = 0
	} else {
		s.newM = s.oldM + (val-s.oldM)/float64(s.totalIterations)
		s.newS = s.oldS + (val-s.oldM)*(val-s.newM)
		s.oldM = s.newM
		s.oldS = s.newS
	}
}

func (s *Statistic) mean() float64 {
	if s.totalIterations > 0 {
		return s.newM
	}
	return 0.0
}

func (s *Statistic) variance() float64 {
	if s.totalIterations <= 1 {
		return 0.0
	}
	return s.newS / float64(s.totalIterations-1)
}

func (s *Statistic) stdev() float64 {
	return math.Sqrt(s.variance())
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
	sp.scoreStats[ply].push(float64(play.Score()))
	sp.bingoStats[ply].push(float64(bingos))
}

func (sp *SimmedPlay) addEquityStat(spread int, leftover float64) {
	sp.equityStats.push(float64(spread) + leftover)
	sp.leftoverStats.push(leftover)
}

// Simmer implements the actual look-ahead search
type Simmer struct {
	movegens   []movegen.MoveGenerator
	gameCopies []*mechanics.XWordGame
	aiplayer   player.AIPlayer

	initialSpread int
	maxPlies      int
	// initialPlayer is the player for whom we are simming.
	initialPlayer  int
	iterationCount int
	threads        int

	simming bool
	plays   []*SimmedPlay

	logStream io.Writer
}

func (s *Simmer) Init(gen movegen.MoveGenerator, game *mechanics.XWordGame,
	aiplayer player.AIPlayer) {

	s.movegens = []movegen.MoveGenerator{gen}
	s.gameCopies = []*mechanics.XWordGame{game}
	s.aiplayer = aiplayer
	s.threads = 1
}

func (s *Simmer) SetThreads(threads int) {
	s.threads = threads
}

func (s *Simmer) SetLogStream(l io.Writer) {
	s.logStream = l
}

func (s *Simmer) makeGameCopies() {
	// Use the first one as the source of truth.
	s.gameCopies = s.gameCopies[:1]
	for i := 1; i < s.threads; i++ {
		s.gameCopies = append(s.gameCopies, s.gameCopies[0].Copy())
	}
	s.movegens = s.movegens[:1]
	for i := 1; i < s.threads; i++ {
		s.movegens = append(s.movegens,
			movegen.NewGordonGenerator(s.gameCopies[0].Gaddag(),
				s.gameCopies[i].Board(), s.gameCopies[0].Bag().LetterDistribution()))
	}
}

func (s *Simmer) resetStats(plies int, plays []*move.Move) {
	s.iterationCount = 0
	s.maxPlies = plies
	for _, g := range s.gameCopies {
		g.SetStateStackLength(plies)
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

// Simulate sims all the plays.
func (s *Simmer) Simulate(ctx context.Context, plays []*move.Move, plies int) error {
	s.simming = true
	defer func() {
		s.simming = false
		log.Info().Msgf("Simulation ended after %v iterations", s.iterationCount)
	}()
	s.makeGameCopies()
	s.resetStats(plies, plays)

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

				s.simSingleIteration(plies, t, iterNum, logChan)
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
		// log.Debug().Msgf("Playing move %v", play)
		s.gameCopies[thread].PlayMove(simmedPlay.play, true)
		for ply := 0; ply < plies; ply++ {
			// Each ply is a player taking a turn
			onTurn := s.gameCopies[thread].PlayerOnTurn()
			if s.gameCopies[thread].Playing() {
				// Assume there are exactly two players.

				bestPlay := s.bestStaticTurn(onTurn, thread)
				// log.Debug().Msgf("Ply %v, Best play: %v", ply+1, bestPlay)
				s.gameCopies[thread].PlayMove(bestPlay, false)
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
		return s.plays[i].equityStats.mean() > s.plays[j].equityStats.mean()
	})
}

func (s *Simmer) printStats() string {
	return s.EquityStats() + "\n Details per play \n" + s.ScoreDetails()
}

func (s *Simmer) EquityStats() string {
	stats := ""

	s.sortPlaysByEquity()
	stats += fmt.Sprintf("%20v%8v\n", "Play", "Equity")

	for _, play := range s.plays {
		stats += fmt.Sprintf("%20v%8.3f\n", play.play.ShortDescription(),
			play.equityStats.mean())
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
				play.play.ShortDescription(), play.scoreStats[ply].mean(),
				play.scoreStats[ply].stdev(),
				100.0*play.bingoStats[ply].mean())
		}
		stats += "\n"
	}
	stats += fmt.Sprintf("Iterations: %v\n", s.iterationCount)

	return stats
}
