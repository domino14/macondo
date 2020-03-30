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

	sp.scoreStats[ply].push(float64(play.Score()))
	sp.bingoStats[ply].push(float64(bingos))
}

func (sp *SimmedPlay) addEquityStat(spread int, leftover float64) {
	sp.equityStats.push(float64(spread) + leftover)
	sp.leftoverStats.push(leftover)
}

// Simmer implements the actual look-ahead search
type Simmer struct {
	movegen  movegen.MoveGenerator
	game     *mechanics.XWordGame
	aiplayer player.AIPlayer

	initialSpread int
	maxPlies      int
	// initialPlayer is the player for whom we are simming.
	initialPlayer  int
	iterationCount int
	threads        int

	plays []*SimmedPlay

	logStream io.Writer
}

func (s *Simmer) Init(movegen movegen.MoveGenerator, game *mechanics.XWordGame,
	aiplayer player.AIPlayer) {

	s.movegen = movegen
	s.game = game
	s.aiplayer = aiplayer
}

func (s *Simmer) SetLogStream(l io.Writer) {
	s.logStream = l
}

func (s *Simmer) resetStats(plies int, plays []*move.Move) {
	s.iterationCount = 0
	s.maxPlies = plies
	s.game.SetStateStackLength(plies)
	s.initialSpread = s.game.CurrentSpread()
	s.initialPlayer = s.game.PlayerOnTurn()
	s.plays = make([]*SimmedPlay, len(plays))
	for idx, play := range plays {
		s.plays[idx] = &SimmedPlay{}
		s.plays[idx].play = play
		s.plays[idx].scoreStats = make([]Statistic, plies)
		s.plays[idx].bingoStats = make([]Statistic, plies)
	}

}

// Simulate sims all the plays.
func (s *Simmer) Simulate(ctx context.Context, plays []*move.Move, plies int) error {
	s.resetStats(plies, plays)

	for {
		s.simSingleIteration(plies)
		s.iterationCount++
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Do nothing
		}
	}
}

func (s *Simmer) simSingleIteration(plies int) {
	// Give opponent a random rack from the bag. Note that this also
	// shuffles the bag!
	opp := (s.initialPlayer + 1) % s.game.NumPlayers()
	s.game.SetRandomRack(opp)
	logIter := LogIteration{Iteration: s.iterationCount + 1, Plays: []LogPlay{}}

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
		s.game.PlayMove(simmedPlay.play, true)
		for ply := 0; ply < plies; ply++ {
			// Each ply is a player taking a turn
			onTurn := s.game.PlayerOnTurn()
			if s.game.Playing() {
				// Assume there are exactly two players.

				bestPlay := s.bestStaticTurn(onTurn)
				// log.Debug().Msgf("Ply %v, Best play: %v", ply+1, bestPlay)
				s.game.PlayMove(bestPlay, false)
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
		simmedPlay.addEquityStat(s.game.SpreadFor(s.initialPlayer), leftover)
		s.game.ResetToFirstState()
		logIter.Plays = append(logIter.Plays, logPlay)
	}
	if s.logStream != nil {
		out, err := yaml.Marshal([]LogIteration{logIter})
		if err != nil {
			log.Error().Err(err).Msg("marshalling log")
			return
		}
		s.logStream.Write(out)
	}
}

func (s *Simmer) bestStaticTurn(playerID int) *move.Move {
	return player.GenBestStaticTurn(s.game, s.movegen, s.aiplayer, playerID)
}

func (s *Simmer) sortPlaysByEquity() {
	// Sort by equity
	// log.Debug().Msgf("Sorting plays: %v", s.plays)
	sort.Slice(s.plays, func(i, j int) bool {
		return s.plays[i].equityStats.mean() > s.plays[j].equityStats.mean()
	})
}

func (s *Simmer) printStats() string {
	stats := ""

	s.sortPlaysByEquity()

	// Return a string representation of the stats

	stats += fmt.Sprintf("%20v%8v\n", "Play", "Equity")

	for _, play := range s.plays {
		stats += fmt.Sprintf("%20v%8.3f\n", play.play.ShortDescription(),
			play.equityStats.mean())
	}
	stats += "\n Details per play \n"

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
	return stats
}
