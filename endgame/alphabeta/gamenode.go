package alphabeta

import (
	"fmt"

	"github.com/domino14/macondo/move"
)

// a game node has to have enough information to allow the game and turns
// to be reconstructed.
type GameNode struct {
	// the move corresponding to the node is the move that is being evaluated.
	move           *move.Move
	parent         *GameNode
	heuristicValue float32
	children       []*GameNode // children should be null until expanded.
	generatedPlays []*move.Move
}

func (g *GameNode) Children() []*GameNode {
	return g.children
}

func (g *GameNode) Parent() *GameNode {
	return g.parent
}

func (g *GameNode) HeuristicValue() float32 {
	return g.heuristicValue
}

func (g *GameNode) Move() *move.Move {
	return g.move
}

func (g *GameNode) GeneratedPlays() []*move.Move {
	return g.generatedPlays
}

func (g *GameNode) String() string {
	return fmt.Sprintf("<gamenode move %v, heuristicVal %v, nchild %v>", g.move,
		g.heuristicValue, len(g.children))
}

func (g *GameNode) value(s *Solver, gameOver bool) float32 {
	g.calculateValue(s, gameOver)
	// log.Debug().Msgf("heuristic value of node %p is %v", g, g.heuristicValue)
	return g.heuristicValue
}

func (g *GameNode) calculateValue(s *Solver, gameOver bool) {
	// calculate the heuristic value of this node, and store it.
	// we start with a max node. At 1-ply (and all odd plies), maximizing
	// is always false.
	// log.Debug().Msgf("Need to calculate value for %v. Player on turn %v, maximizing %v", g.move, s.game.PlayerOnTurn(), maximizing)

	// Because calculateValue is called after PlayMove has been called,
	// the "playerOnTurn" is actually not the player who made the move
	// whose value we are calculating.
	opponent := s.game.PlayerOnTurn()
	playerWhoMadeMove := (opponent + 1) % (s.game.NumPlayers())

	// The initial spread is always from the maximizing point of view.
	initialSpread := s.initialSpread
	spreadNow := s.game.PointsFor(playerWhoMadeMove) - s.game.PointsFor(opponent)
	negateHeurVal := false
	if playerWhoMadeMove != s.maximizingPlayer {
		// spreadNow = -spreadNow
		initialSpread = -initialSpread
		negateHeurVal = true
	}

	// If the game is over, the value should just be the spread change.
	if gameOver {
		// Technically no one is on turn, but the player NOT on turn is
		// the one that just ended the game.
		// Note that because of the way we track state, it is the state
		// in the solver right now; that's why the game node doesn't matter
		// right here:
		g.heuristicValue = float32(spreadNow - initialSpread)
	} else {
		// The valuation is already an estimate of the overall gain or loss
		// in spread for this move (if taken to the end of the game).

		// `player` is NOT the one that just made a move.
		ptValue := g.move.Score()
		// don't double-count score; it's already in the valuation:
		moveVal := g.move.Valuation() - float32(ptValue)
		// What is the spread right now? The valuation should be relative
		// to that.
		// log.Debug().Msgf("calculating heur value for %v as %v + %v - %v",
		// 	g.move, spreadNow, moveVal, initialSpread)
		if g.move.Action() == move.MoveTypePass {
			// Hack to make sure we don't unnecessarily pass when opp
			// is stuck.
			moveVal -= float32(0.001)
		}
		g.heuristicValue = float32(spreadNow) + moveVal - float32(initialSpread)
		// g.heuristicValue = s.game.EndgameSpreadEstimate(player, maximizing) - float32(initialSpread)
		// log.Debug().Msgf("Calculating heuristic value of %v as %v - %v",
		// 	g.move, s.game.EndgameSpreadEstimate(player), float32(initialSpread))
	}
	if negateHeurVal {
		// The maximizing player is always "us" - the player that we are
		// solving the endgame for. So if this not the maximizing node,
		// we want to negate the heuristic value, as it needs to be as
		// negative as possible relative to "us". I know, minimax is
		// hard to reason about, but I think this makes sense. At least
		// it seems to work.
		g.heuristicValue = -g.heuristicValue
		// log.Debug().Msg("Negating since not maximizing player")
	}
}

func (g *GameNode) serialize() []int32 {
	// Climb down tree and serialize nodes.
	return nil
}
