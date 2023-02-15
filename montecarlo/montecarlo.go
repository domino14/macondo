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

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cache"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/stats"
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
	WinRatio float64   `json:"win,omitempty" yaml:"win,omitempty"`
	Plies    []LogPlay `json:"plies,omitempty" yaml:"plies,omitempty,flow"`
}

type SimmedPlay struct {
	sync.Mutex
	play          *move.Move
	scoreStats    []stats.Statistic
	bingoStats    []stats.Statistic
	equityStats   stats.Statistic
	leftoverStats stats.Statistic
	winPctStats   stats.Statistic
}

func (sp *SimmedPlay) String() string {
	return fmt.Sprintf("<Simmed play: %v (stats: %v %v %v %v %v)>", sp.play.ShortDescription(),
		sp.scoreStats, sp.bingoStats, sp.equityStats, sp.leftoverStats, sp.winPctStats)
}

func (sp *SimmedPlay) addScoreStat(play *move.Move, ply int) {
	// log.Debug().Msgf("Adding a stat for %v (pidx %v ply %v)", play, pidx, ply)
	var bingos int
	if play.BingoPlayed() {
		bingos = 1
	}
	sp.Lock()
	defer sp.Unlock()
	sp.scoreStats[ply].Push(float64(play.Score()))
	sp.bingoStats[ply].Push(float64(bingos))
}

func (sp *SimmedPlay) addEquityStat(initialSpread int, spread int, leftover float64,
	gameover bool, winpcts [][]float32, tilesUnseen int, pliesAreEven bool) {
	sp.Lock()
	defer sp.Unlock()
	sp.equityStats.Push(float64(spread-initialSpread) + leftover)
	sp.leftoverStats.Push(float64(leftover))
	if gameover || tilesUnseen == 0 {
		if spread == 0 {
			sp.winPctStats.Push(0.5)
		} else if spread > 0 {
			sp.winPctStats.Push(1.0)
		} else {
			sp.winPctStats.Push(0.0)
		}
		return
	}
	if tilesUnseen > 93 {
		// Only for ZOMGWords or similar; this is a bit of a hack.
		tilesUnseen = 93
	}
	// for an even-ply sim, it is our opponent's turn at the end of the sim.
	// the table is calculated from our perspective, so flip the spread.
	// i.e. if we are winning by 20 pts at the end of the sim, and our opponent
	// is on turn, we want to look up -20 as the spread, and then flip the win %
	// as well.
	spreadPlusLeftover := spread + int(math.Round(leftover))
	if pliesAreEven {
		spreadPlusLeftover = -spreadPlusLeftover
	}

	if spreadPlusLeftover > equity.MaxRepresentedWinSpread {
		spreadPlusLeftover = equity.MaxRepresentedWinSpread
	}
	if spreadPlusLeftover < -equity.MaxRepresentedWinSpread {
		spreadPlusLeftover = -equity.MaxRepresentedWinSpread
	}
	// winpcts goes from +MaxRepresentedWinSpread to -MaxRespresentedWinSpread
	// spread = index
	// 200 = 0
	// 199 = 1
	// 99 = 101
	// 0 = 200
	// -1 = 201
	// -101 = 301
	// -200 = 400
	pct := winpcts[equity.MaxRepresentedWinSpread-spreadPlusLeftover][tilesUnseen]
	log.Trace().Int("i1", equity.MaxRepresentedWinSpread-spreadPlusLeftover).Int("i2", tilesUnseen).Float32(
		"pct", pct).Bool("plies-are-even", pliesAreEven).Msg("calc-win%")
	if pliesAreEven {
		// see the above comment re flipping win pct.
		pct = 1 - pct
	}
	sp.winPctStats.Push(float64(pct))
}

// Simmer implements the actual look-ahead search
type Simmer struct {
	origGame *game.Game

	gameCopies []*game.Game
	// movegens          []movegen.MoveGenerator
	equityCalculators []equity.EquityCalculator
	aiplayers         []aiturnplayer.AITurnPlayer
	leaveValues       equity.Leaves

	initialSpread int
	maxPlies      int
	// initialPlayer is the player for whom we are simming.
	initialPlayer  int
	iterationCount int
	threads        int

	simming    bool
	readyToSim bool
	plays      []*SimmedPlay
	winPcts    [][]float32
	cfg        *config.Config

	logStream io.Writer
}

func (s *Simmer) Init(game *game.Game, eqCalcs []equity.EquityCalculator,
	leaves equity.Leaves, cfg *config.Config) {
	s.origGame = game

	s.equityCalculators = eqCalcs
	s.leaveValues = leaves
	s.threads = int(math.Max(1, float64(runtime.NumCPU()-1)))

	// Hard-code the location of the win-pct file for now.
	// If we want to make some for other lexica in the future we'll
	// have to redo the equity calculator stuff.
	s.cfg = cfg
	if s.cfg != nil {
		// some hardcoded stuff here:
		winpct, err := cache.Load(s.cfg, "winpctfile:CSW:winpct.csv", equity.WinPCTLoadFunc)
		if err != nil {
			panic(err)
		}
		var ok bool
		s.winPcts, ok = winpct.([][]float32)
		if !ok {
			panic("win percentages not correct type")
		}
	}
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

	// Pre-shuffle bag so we can make identical copies of it with fixedOrder
	s.origGame.Bag().Shuffle()

	for i := 0; i < s.threads; i++ {
		s.gameCopies = append(s.gameCopies, s.origGame.Copy())
		s.gameCopies[i].Bag().SetFixedOrder(true)

		player, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(s.gameCopies[i], s.origGame.Config(), s.equityCalculators)
		if err != nil {
			return err
		}
		s.aiplayers = append(s.aiplayers, player)
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
		s.plays[idx].scoreStats = make([]stats.Statistic, plies)
		s.plays[idx].bingoStats = make([]stats.Statistic, plies)
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
				s.iterationCount++
				iterNum := s.iterationCount
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

	g := s.gameCopies[thread]

	opp := (s.initialPlayer + 1) % g.NumPlayers()
	g.SetRandomRack(opp)
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
		g.SetBackupMode(game.SimulationMode)
		g.PlayMove(simmedPlay.play, false, 0)
		g.SetBackupMode(game.NoBackup)
		// Further plies will NOT be backed up.
		for ply := 0; ply < plies; ply++ {
			// Each ply is a player taking a turn
			onTurn := g.PlayerOnTurn()
			if g.Playing() == pb.PlayState_PLAYING {
				// Assume there are exactly two players.

				bestPlay := s.bestStaticTurn(onTurn, thread)
				// log.Debug().Msgf("Ply %v, Best play: %v", ply+1, bestPlay)
				g.PlayMove(bestPlay, false, 0)
				// log.Debug().Msgf("Score is now %v", s.game.Score())
				if s.logStream != nil {
					plyChild = LogPlay{Play: bestPlay.ShortDescription(), Rack: bestPlay.FullRack(), Pts: bestPlay.Score()}
				}
				if ply == plies-2 || ply == plies-1 {
					// It's either OUR last turn or OPP's last turn.
					// Calculate equity of leftover tiles.
					thisLeftover := s.leaveValues.LeaveValue(bestPlay.Leave())
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
				// Maybe these add{X}Stat functions can instead write them to
				// a channel to avoid mutices
				simmedPlay.addScoreStat(bestPlay, ply)
			}
		}
		// log.Debug().Msgf("Spread for initial player: %v, leftover: %v",
		// 	s.game.SpreadFor(s.initialPlayer), leftover)
		simmedPlay.addEquityStat(
			s.initialSpread,
			g.SpreadFor(s.initialPlayer),
			leftover,
			g.Playing() == pb.PlayState_GAME_OVER,
			s.winPcts,
			// Tiles unseen: number of tiles in the bag + tiles on my opponent's rack:
			g.Bag().TilesRemaining()+
				int(g.RackFor(1-s.initialPlayer).NumTiles()),
			plies%2 == 0,
		)
		g.ResetToFirstState()
		if s.logStream != nil {
			logPlay.WinRatio = simmedPlay.winPctStats.Last()
			logIter.Plays = append(logIter.Plays, logPlay)
		}
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
	return aiturnplayer.GenBestStaticTurn(s.gameCopies[thread], s.aiplayers[thread], playerID)
}

func (s *Simmer) sortPlaysByEquity() {
	// log.Debug().Msgf("Sorting plays: %v", s.plays)
	sort.Slice(s.plays, func(i, j int) bool {
		return s.plays[i].equityStats.Mean() > s.plays[j].equityStats.Mean()
	})
}

func (s *Simmer) sortPlaysByWinRate() {
	// log.Debug().Msgf("Sorting plays: %v", s.plays)
	sort.Slice(s.plays, func(i, j int) bool {
		if s.plays[i].winPctStats.Mean() == s.plays[j].winPctStats.Mean() {
			return s.plays[i].equityStats.Mean() > s.plays[j].equityStats.Mean()
		}
		return s.plays[i].winPctStats.Mean() > s.plays[j].winPctStats.Mean()
	})
}

func (s *Simmer) printStats() string {
	return s.EquityStats() + "\n Details per play \n" + s.ScoreDetails()
}

func (s *Simmer) EquityStats() string {
	stats := ""
	s.sortPlaysByWinRate()
	stats += fmt.Sprintf("%-20s%6s%8s%8s\n", "Play", "Score", "Win%", "Equity")

	for _, play := range s.plays {
		stats += fmt.Sprintf("%-20s%6d%8.2f%8.3f\n", play.play.ShortDescription(),
			play.play.Score(), 100.0*play.winPctStats.Mean(), play.equityStats.Mean())
	}
	stats += fmt.Sprintf("Iterations: %d\n", s.iterationCount)
	return stats
}

func (s *Simmer) ScoreDetails() string {
	stats := ""
	s.sortPlaysByWinRate()
	for ply := 0; ply < s.maxPlies; ply++ {
		who := "You"
		if ply%2 == 0 {
			who = "Opponent"
		}
		stats += fmt.Sprintf("**Ply %d (%s)**\n%-20s%8s%8s%8s%8s\n%s\n",
			ply+1, who, "Play", "Win%", "Mean", "Stdev", "Bingo %", strings.Repeat("-", 52))
		for _, play := range s.plays {
			stats += fmt.Sprintf("%-20s%8.2f%8.3f%8.3f%8.3f\n",
				play.play.ShortDescription(), 100.0*play.winPctStats.Mean(),
				play.scoreStats[ply].Mean(), play.scoreStats[ply].Stdev(),
				100.0*play.bingoStats[ply].Mean())
		}
		stats += "\n"
	}
	stats += fmt.Sprintf("Iterations: %d\n", s.iterationCount)
	return stats
}
