package alphabeta

import (
	"fmt"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

const PerTurnPenalty = float32(0.001)

type nodeValue struct {
	value          float32
	knownEnd       bool
	sequenceLength int
	isPass         bool
}

func (nv nodeValue) String() string {
	return fmt.Sprintf("<val: %v seqLength: %v knownEnd: %v>", nv.value,
		nv.sequenceLength, nv.knownEnd)
}

func (nv *nodeValue) negate() {
	nv.value = -nv.value
}

func (nv nodeValue) less(other nodeValue) bool {
	if nv.value != other.value {
		return nv.value < other.value
	}
	// Second tie-breaker is whether this is a known end or a speculative one.
	if nv.knownEnd != other.knownEnd {
		// basically, we are less than other if we are an unknown end
		// and other is a known end
		// i.e. known end > unknown end
		return other.knownEnd
	}
	// Third tie-breaker is whether this is a pass or not.
	if nv.isPass != other.isPass {
		// we should rank non-passes higher than passes if everything else
		// is equal.
		// i.e. pass < not pass
		return nv.isPass
	}
	// Fourth tie-breaker is length of sequence, favoring shorter sequences
	return nv.sequenceLength > other.sequenceLength

}

// a game node has to have enough information to allow the game and turns
// to be reconstructed.
type GameNode struct {
	// the move corresponding to the node is the move that is being evaluated.
	move           *move.Move
	parent         *GameNode
	heuristicValue nodeValue
	children       []*GameNode // children should be null until expanded.
	generatedPlays []*move.Move
}

func (g *GameNode) Children() []*GameNode {
	return g.children
}

func (g *GameNode) Parent() *GameNode {
	return g.parent
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

func (g *GameNode) value(s *Solver) nodeValue {
	g.calculateValue(s)
	// log.Debug().Msgf("heuristic value of node %p is %v", g, g.heuristicValue)
	return g.heuristicValue
}

func (g *GameNode) calculateValue(s *Solver) {
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
	gameOver := s.game.Playing() != pb.PlayState_PLAYING
	// If the game is over, the value should just be the spread change.
	if gameOver {
		// Technically no one is on turn, but the player NOT on turn is
		// the one that just ended the game.
		// Note that because of the way we track state, it is the state
		// in the solver right now; that's why the game node doesn't matter
		// right here:
		g.heuristicValue = nodeValue{
			value:          float32(spreadNow - initialSpread),
			knownEnd:       true,
			isPass:         g.move.Action() == move.MoveTypePass,
			sequenceLength: s.game.Turn() - s.initialTurnNum}
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
		g.heuristicValue = nodeValue{
			value:          float32(spreadNow) + moveVal - float32(initialSpread),
			knownEnd:       false,
			sequenceLength: s.game.Turn() - s.initialTurnNum,
			isPass:         g.move.Action() == move.MoveTypePass}
		// g.heuristicValue = s.game.EndgameSpreadEstimate(player, maximizing) - float32(initialSpread)
		// log.Debug().Msgf("Calculating heuristic value of %v as %v - %v",
		// 	g.move, s.game.EndgameSpreadEstimate(player), float32(initialSpread))
		// g.heuristicValue.value = 0 // TEMP
	}
	if negateHeurVal {
		// The maximizing player is always "us" - the player that we are
		// solving the endgame for. So if this not the maximizing node,
		// we want to negate the heuristic value, as it needs to be as
		// negative as possible relative to "us". I know, minimax is
		// hard to reason about, but I think this makes sense. At least
		// it seems to work.
		g.heuristicValue.negate()
		// log.Debug().Msg("Negating since not maximizing player")
	}
}

func (g *GameNode) serialize() []int32 {
	// Climb down tree and serialize nodes.
	return nil
}
