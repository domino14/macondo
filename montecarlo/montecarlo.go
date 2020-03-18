// Package montecarlo implements truncated monte-carlo search
// during the regular game. In other words, "simming".
package montecarlo

import (
	"context"

	"github.com/domino14/macondo/mechanics"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
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
	totalIterations int
	totalScore      int
	stdev           float32
}

// Simmer implements the actual look-ahead search
type Simmer struct {
	movegen        movegen.MoveGenerator
	game           *mechanics.XWordGame
	initialSpread  int
	iterationCount int
	threads        int
	// The plays being simmed:
	plays []*move.Move
	stats []*Statistic
}

func (s *Simmer) Init(movegen movegen.MoveGenerator, game *mechanics.XWordGame) {
	s.movegen = movegen
	s.game = game
}

// Simulate sims all the plays.
func (s *Simmer) Simulate(ctx context.Context, plays []*move.Move, plies int) error {
	s.iterationCount = 0
	s.game.SetStateStackLength(plies)
	s.initialSpread = s.game.CurrentSpread()
	for {
		s.game.Bag().Shuffle()
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
	for _, play := range plays {
		// Play the move, and back up the game state.
		s.game.PlayMove(play, true)
		for p := 0; p < plies; p++ {
			// Each ply is a player taking a turn
			s.playBestStaticTurn(s.game.PlayerOnTurn())
		}
		// Restore the game state from backup.
		s.game.ResetToFirstState()
	}
}

func (s *Simmer) playBestStaticTurn(playerID int) {
	opp := (playerID + 1) % s.game.NumPlayers()
	s.movegen.SetOppRack(s.game.RackFor(opp))
	s.movegen.GenAll(s.game.RackFor(playerID))

	bestPlay := s.movegen.Plays()[0]
	// logging here?
	s.game.PlayMove(bestPlay, false)
}
