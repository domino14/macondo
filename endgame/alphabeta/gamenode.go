package alphabeta

import (
	"fmt"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

const PerTurnPenalty = float32(0.001)

type nodeValue struct {
	value    float32
	knownEnd bool
	isPass   bool
}

func (nv *nodeValue) String() string {
	return fmt.Sprintf("<val: %v knownEnd: %v>", nv.value, nv.knownEnd)
}

func (nv *nodeValue) negate() {
	nv.value = -nv.value
}

func (nv *nodeValue) less(other *nodeValue) bool {
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
	return false

}

// a game node has to have enough information to allow the game and turns
// to be reconstructed.
type GameNode struct {
	move           *move.Move
	parent         *GameNode
	heuristicValue nodeValue
	valuation      float32 // valuation is an initial estimate of the value of a move.
	depth          uint8
	hashKey        uint64
}

func (g *GameNode) Copy() *GameNode {
	mv := &move.Move{}
	mv.CopyFrom(g.move)
	return &GameNode{
		move:           mv,
		parent:         g.parent,
		heuristicValue: g.heuristicValue,
		valuation:      g.valuation,
		depth:          g.depth,
		hashKey:        g.hashKey,
	}
}

func (g *GameNode) CopyFrom(o *GameNode) {
	g.heuristicValue = o.heuristicValue
	g.move.CopyFrom(o.move)
	g.parent = o.parent
	g.valuation = o.valuation
	g.depth = o.depth
	g.hashKey = o.hashKey
}

func (g *GameNode) Parent() *GameNode {
	return g.parent
}

func (g *GameNode) GetDepth() uint8 {
	return g.depth
}

func (g *GameNode) String() string {
	// This function allocates but is only used for test purposes.
	return fmt.Sprintf(
		"<gamenode move %v, heuristicVal %v>",
		g.move, g.heuristicValue)
}

func (g *GameNode) calculateValue(s *Solver, negateHeurVal bool) {
	// calculate the heuristic value of this node, and store it.
	// we start with a max node. At 1-ply (and all odd plies), maximizing
	// is always false.

	// Because calculateValue is called after PlayMove has been called,
	// the "playerOnTurn" is actually not the player who made the move
	// whose value we are calculating.
	opponent := s.game.PlayerOnTurn()
	playerWhoMadeMove := (opponent + 1) % (s.game.NumPlayers())

	// The initial spread is always from the maximizing point of view.
	initialSpread := s.initialSpread
	spreadNow := s.game.PointsFor(playerWhoMadeMove) - s.game.PointsFor(opponent)
	if negateHeurVal {
		// Alpha-Beta (min) measures spread from the perspective of the
		// player who made the move, measures improvement, negates it,
		// then selects the minimum (least improved for opponent) node.
		// https://www.chessprogramming.org/Alpha-Beta#Max_versus_Min
		initialSpread = -initialSpread
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
			isPass:         g.move.Action() == move.MoveTypePass}
	} else {
		// The valuation is already an estimate of the overall gain or loss
		// in spread for this move (if taken to the end of the game).

		// `player` is NOT the one that just made a move.
		ptValue := g.move.Score()
		// don't double-count score; it's already in the valuation:
		moveVal := g.valuation - float32(ptValue)
		// What is the spread right now? The valuation should be relative
		// to that.
		g.heuristicValue = nodeValue{
			value:          float32(spreadNow) + moveVal - float32(initialSpread),
			knownEnd:       false,
			isPass:         g.move.Action() == move.MoveTypePass}
	}
	if negateHeurVal {
		// The maximizing player is always "us" - the player that we are
		// solving the endgame for. So if this not the maximizing node,
		// we want to negate the heuristic value, as it needs to be as
		// negative as possible relative to "us". I know, minimax is
		// hard to reason about, but I think this makes sense. At least
		// it seems to work.
		g.heuristicValue.negate()
	}
}
