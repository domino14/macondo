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
	move             *move.Move
	parent           *GameNode
	heuristicValue   nodeValue
	depth            uint8
	onlyPassPossible bool
}

func (g *GameNode) Copy() *GameNode {
	mv := &move.Move{}
	mv.CopyFrom(g.move)
	return &GameNode{
		move:             mv,
		parent:           g.parent,
		heuristicValue:   g.heuristicValue,
		depth:            g.depth,
		onlyPassPossible: g.onlyPassPossible,
	}
}

func (g *GameNode) CopyFrom(o *GameNode) {
	g.heuristicValue = o.heuristicValue
	g.move.CopyFrom(o.move)
	g.parent = o.parent
	g.depth = o.depth
	g.onlyPassPossible = o.onlyPassPossible
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

func (g *GameNode) Negative() *GameNode {
	hv := g.heuristicValue
	hv.negate()
	return &GameNode{
		move:             g.move,
		parent:           g.parent,
		heuristicValue:   hv,
		depth:            g.depth,
		onlyPassPossible: g.onlyPassPossible,
	}
}

// calculateNValue calculates the value of a node for use in negamax.
// The value must always be from the POV of the maximizing player!
func (g *GameNode) calculateNValue(s *Solver) {
	// The initial spread is always from the maximizing point of view.
	initialSpread := s.initialSpread
	// spreadNow is from the POV of the maximizing player
	spreadNow := s.game.PointsFor(s.maximizingPlayer) -
		s.game.PointsFor(1-s.maximizingPlayer)

	gameOver := s.game.Playing() != pb.PlayState_PLAYING
	// If the game is over, the value should just be the spread change.
	if gameOver {
		// Technically no one is on turn, but the player NOT on turn is
		// the one that just ended the game.
		// Note that because of the way we track state, it is the state
		// in the solver right now; that's why the game node doesn't matter
		// right here:
		g.heuristicValue = nodeValue{
			value:    float32(spreadNow - initialSpread),
			knownEnd: true,
			isPass:   g.move.Action() == move.MoveTypePass}
	} else {
		// The valuation is already an estimate of the overall gain or loss
		// in spread for this move (if taken to the end of the game).

		// `player` is NOT the one that just made a move.
		ptValue := g.move.Score()
		// don't double-count score; it's already in the valuation:
		moveVal := g.move.Valuation() - float32(ptValue)
		// What is the spread right now? The valuation should be relative
		// to that.
		g.heuristicValue = nodeValue{
			value:    float32(spreadNow) + moveVal - float32(initialSpread),
			knownEnd: false,
			isPass:   g.move.Action() == move.MoveTypePass}
	}
}
