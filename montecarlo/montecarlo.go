// Package montecarlo implements truncated monte-carlo search
// during the regular game. In other words, "simming".
package montecarlo

import (
	"context"
	"fmt"
	"io"
	"math"
	"strings"

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
	move            *move.Move
	bingos          int
	maxPlies        int
	totalIterations int
	totalScore      int

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
	// Although this is a recursive structure we don't really use it
	// recursively.
	Plies []LogPlay `json:"plies,omitempty" yaml:"plies,omitempty"`
}

func (s *Statistic) push(score int) {
	s.totalIterations++
	if s.totalIterations == 1 {
		s.oldM = float64(score)
		s.newM = float64(score)
		s.oldS = 0
	} else {
		s.newM = s.oldM + (float64(score)-s.oldM)/float64(s.totalIterations)
		s.newS = s.oldS + (float64(score)-s.oldM)*(float64(score)-s.newM)
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

func (s *Statistic) bingopct() float64 {
	return 100.0 * float64(s.bingos) / float64(s.totalIterations)
}

// Simmer implements the actual look-ahead search
type Simmer struct {
	movegen       movegen.MoveGenerator
	game          *mechanics.XWordGame
	initialSpread int
	maxPlies      int
	// initialPlayer is the player for whom we are simming.
	initialPlayer  int
	iterationCount int
	threads        int
	// The plays being simmed:
	plays []*move.Move
	stats [][]*Statistic

	logStream io.Writer
}

func (s *Simmer) Init(movegen movegen.MoveGenerator, game *mechanics.XWordGame) {
	s.movegen = movegen
	s.game = game
}

func (s *Simmer) SetLogStream(l io.Writer) {
	s.logStream = l
}

func (s *Simmer) resetStats(plies, numPlays int) {
	s.iterationCount = 0
	s.maxPlies = plies
	s.game.SetStateStackLength(plies)
	s.initialSpread = s.game.CurrentSpread()
	s.initialPlayer = s.game.PlayerOnTurn()
	s.stats = make([][]*Statistic, numPlays)
	for i := 0; i < numPlays; i++ {
		s.stats[i] = make([]*Statistic, plies)
	}
}

func (s *Simmer) addSpreadStat(play *move.Move, pidx, ply, spread int) {
	// log.Debug().Msgf("Adding a stat for %v (pidx %v ply %v)", play, pidx, ply)
	var bingos int
	if play.TilesPlayed() == 7 {
		bingos = 1
	}
	if s.stats[pidx][ply] == nil {
		s.stats[pidx][ply] = &Statistic{
			move: play,
		}
	}

	stat := s.stats[pidx][ply]
	stat.bingos += bingos
	stat.push(play.Score())
}

// Simulate sims all the plays.
func (s *Simmer) Simulate(ctx context.Context, plays []*move.Move, plies int) error {
	s.resetStats(plies, len(plays))
	s.plays = plays
	for {
		s.simSingleIteration(plays, plies)
		s.iterationCount++
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Do nothing
		}
	}
}

func (s *Simmer) simSingleIteration(plays []*move.Move, plies int) {
	// Give opponent a random rack from the bag. Note that this also
	// shuffles the bag!
	opp := (s.initialPlayer + 1) % s.game.NumPlayers()
	s.game.SetRandomRack(opp)
	logIter := LogIteration{Iteration: s.iterationCount + 1, Plays: []LogPlay{}}

	for parentIdx, play := range plays {
		logPlay := LogPlay{Play: play.ShortDescription(), Rack: play.FullRack(), Pts: play.Score()}

		// logIter.Plays = append(logIter.Plays)
		// Play the move, and back up the game state.
		// log.Debug().Msgf("Playing move %v", play)
		s.game.PlayMove(play, true)
		for ply := 0; ply < plies; ply++ {
			// Each ply is a player taking a turn
			if s.game.Playing() {

				bestPlay := s.bestStaticTurn(s.game.PlayerOnTurn())
				// log.Debug().Msgf("Ply %v, Best play: %v", ply+1, bestPlay)
				s.game.PlayMove(bestPlay, false)
				plyChild := LogPlay{Play: bestPlay.ShortDescription(), Rack: bestPlay.FullRack(), Pts: bestPlay.Score()}
				logPlay.Plies = append(logPlay.Plies, plyChild)
				s.addSpreadStat(bestPlay, parentIdx, ply, s.game.SpreadFor(s.initialPlayer))
			}
		}
		// s.game.CurrentSpread()
		// Restore the game state from backup.
		// log.Debug().Msgf("Reset board to beginning")
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
	// log.Debug().Msgf("playing best static turn for player %v", playerID)
	opp := (playerID + 1) % s.game.NumPlayers()
	s.movegen.SetOppRack(s.game.RackFor(opp))
	s.movegen.GenAll(s.game.RackFor(playerID))

	bestPlay := s.movegen.Plays()[0]
	return bestPlay
}

func (s *Simmer) printStats() string {
	stats := ""
	// Return a string representation of the stats
	for ply := 0; ply < s.maxPlies; ply++ {
		who := "You"
		if ply%2 == 0 {
			who = "Opponent"
		}
		stats += fmt.Sprintf("**Ply %v (%v)**\n%20v%8v%8v%8v\n%v\n",
			ply+1, who, "Play", "Mean", "Stdev", "Bingo %", strings.Repeat("-", 44))
		for playIdx, play := range s.plays {
			stats += fmt.Sprintf("%20v%8.3f%8.3f%8.3f\n",
				play.ShortDescription(), s.stats[playIdx][ply].mean(),
				s.stats[playIdx][ply].stdev(),
				s.stats[playIdx][ply].bingopct())
		}
		stats += "\n"
	}
	return stats
}
