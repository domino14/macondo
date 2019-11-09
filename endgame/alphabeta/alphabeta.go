// Package alphabeta implements an endgame solver using depth-limited
// minimax with alpha-beta pruning.
package alphabeta

import (
	"github.com/domino14/macondo/mechanics"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/rs/zerolog/log"
)

// thanks Wikipedia:
/**function alphabeta(node, depth, α, β, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
		for each child of node do
			play(child)
			value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
			unplayLastMove()
            α := max(α, value)
            if α ≥ β then
                break (* β cut-off *)
        return value
    else
        value := +∞
		for each child of node do
			play(child)
			value := min(value, alphabeta(child, depth − 1, α, β, TRUE))
			unplayLastMove()
            β := min(β, value)
            if α ≥ β then
                break (* α cut-off *)
        return value
(* Initial call *)
alphabeta(origin, depth, −∞, +∞, TRUE)
**/

const (
	// Infinity is 10 million.
	Infinity = 10000000
)

// Solver implements the minimax + alphabeta algorithm.
type Solver struct {
	movegen    *movegen.GordonGenerator
	game       *mechanics.XWordGame
	totalNodes int
}

// a game node has to have enough information to allow the game and turns
// to be reconstructed.
type gameNode struct {
	// the move corresponding to the node is the move that is being evaluated.
	move            *move.Move
	heuristicValue  int
	calculatedValue bool
	children        []*gameNode // children should be null until expanded.
}

// max returns the larger of x or y.
func max(x, y int) int {
	if x < y {
		return y
	}
	return x
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func (g *gameNode) value(s *Solver, maximizing bool) int {
	if !g.calculatedValue {
		g.calculateValue(s, maximizing)
		g.calculatedValue = true
	}
	// log.Debug().Msgf("heuristic value of node %p is %v", g, g.heuristicValue)
	return g.heuristicValue
}

func (g *gameNode) calculateValue(s *Solver, maximizing bool) {
	// calculate the heuristic value of this node, and store it.
	// Right now the heuristic is JUST going to be the current spread.
	// note that because of the way we track state, it is the state
	// in the solver right now.
	// log.Debug().Msgf("Need to calculate value for %v. Player on turn %v, maximizing %v", g.move, s.game.PlayerOnTurn(), maximizing)
	player := s.game.PlayerOnTurn()
	otherPlayer := (player + 1) % (s.game.NumPlayers())

	g.heuristicValue = s.game.PointsFor(player) - s.game.PointsFor(otherPlayer)
	if !maximizing {
		// The maximizing player is always "us" - the player that we are
		// solving the endgame for. So if this not the maximizing node,
		// we want to negate the heuristic value, as it needs to be as
		// negative as possible relative to "us". I know, minimax is
		// hard to reason about, but I think this makes sense. At least
		// it seems to work.
		g.heuristicValue = -g.heuristicValue
	}
}

// Init initializes the solver
func (s *Solver) Init(movegen *movegen.GordonGenerator, game *mechanics.XWordGame) {
	s.movegen = movegen
	s.game = game
	s.totalNodes = 0
}

func (s *Solver) generateChildrenNodes(parent *gameNode) []*gameNode {
	// fmt.Printf("Generating children nodes for parent %v, board %v",
	// 	parent, s.game.Board().ToDisplayText(s.game.Alphabet()))
	s.movegen.GenAll(s.game.RackFor(s.game.PlayerOnTurn()))
	// fmt.Println(s.movegen.Plays())
	children := []*gameNode{}
	for _, m := range s.movegen.Plays() {
		children = append(children, &gameNode{
			move: m,
		})
	}

	if len(s.movegen.Plays()) > 0 && s.movegen.Plays()[0].Action() != move.MoveTypePass {
		children = append(children, &gameNode{
			move: move.NewPassMove(s.game.RackFor(s.game.PlayerOnTurn()).TilesOn()),
		})
	}
	s.totalNodes += len(children)
	return children
}

// Solve solves the endgame given the current state of s.game, for the
// current player whose turn it is in that state.
func (s *Solver) Solve(plies int) (int, *move.Move) {
	// Generate children moves.
	s.movegen.SetSortingParameter(movegen.SortByEndgameHeuristic)
	defer s.movegen.SetSortingParameter(movegen.SortByEquity)
	log.Debug().Msgf("Attempting to solve endgame with %v plies...", plies)
	// technically the children are the actual board _states_ but
	// we don't keep track of those exactly
	n := &gameNode{}
	// the root node is basically the board state prior to making any moves.
	// the children of these nodes are the board states after every move.
	// however we treat the children as those actual moves themselves.

	// Look 6 plies for now. This might still be very slow.
	v := s.alphabeta(n, plies, -Infinity, Infinity, true)
	log.Debug().Msgf("Best spread found: %v", v)
	log.Debug().Msgf("Best variant found:")
	var m *move.Move
	// Go down tree and find best variation:
	parent := n
	for {
		for _, child := range parent.children {
			if child.heuristicValue == v {
				if m == nil {
					m = child.move
				}
				log.Debug().Msgf("%v", child.move)
				parent = child
				break
			}
		}
		if len(parent.children) == 0 || parent.children == nil {
			break
		}
	}
	log.Debug().Msgf("Number of expanded nodes: %v", s.totalNodes)

	return v, m
}

func (s *Solver) alphabeta(node *gameNode, depth int, α int, β int,
	maximizingPlayer bool) int {

	// depthDbg := strings.Repeat(" ", depth)

	if depth == 0 || !s.game.Playing() {
		// s.game.Playing() happens if the game is over; i.e. if the
		// current node is terminal.
		// log.Debug().Msgf("%vending recursion, depth: %v, playing: %v", depthDbg, depth, s.game.Playing())
		return node.value(s, maximizingPlayer)
	}
	// Generate children if they don't exist.
	// if node.children == nil {
	// 	node.children = s.generateChildrenNodes(node)
	// }
	var plays []*move.Move
	if node.children == nil {
		s.movegen.GenAll(s.game.RackFor(s.game.PlayerOnTurn()))
		plays = s.movegen.Plays()
		if len(plays) > 0 && plays[0].Action() != move.MoveTypePass {
			// movegen doesn't generate a pass move if unneeded (actually, I'm not
			// totally sure why). So generate it here, as sometimes a pass is beneficial
			// in the endgame.
			plays = append(plays, move.NewPassMove(s.game.RackFor(s.game.PlayerOnTurn()).TilesOn()))
		}
	}
	childGenerator := func() func() (*gameNode, bool) {
		idx := -1
		return func() (*gameNode, bool) {
			idx++
			if idx == len(plays) {
				return nil, false
			}
			s.totalNodes++
			return &gameNode{move: plays[idx]}, true
		}
	}

	if maximizingPlayer {
		value := -Infinity
		iter := childGenerator()
		for child, ok := iter(); ok; child, ok = iter() {
			// Play the child
			// log.Debug().Msgf("%vGoing to play move %v", depthDbg, child.move)
			s.game.PlayMove(child.move, true)
			// log.Debug().Msgf("%vState is now %v", depthDbg,
			// s.game.String())
			v := s.alphabeta(child, depth-1, α, β, false)
			s.game.UnplayLastMove()
			// log.Debug().Msgf("%vAfter unplay, state is now %v", depthDbg, s.game.String())
			if v > value {
				value = v
				// log.Debug().Msgf("%vFound a better move: %v (%v)", depthDbg, value, tm)
			}
			α = max(α, value)
			if α >= β {
				// log.Debug().Msgf("%vBeta cut-off: %v>=%v", depthDbg, α, β)
				break // beta cut-off
			}
			node.children = append(node.children, child)
		}
		node.calculatedValue = true
		node.heuristicValue = value
		return value
	}
	// Otherwise, not maximizing
	value := Infinity
	iter := childGenerator()
	for child, ok := iter(); ok; child, ok = iter() {
		// log.Debug().Msgf("%vGoing to play move %v", depthDbg, child.move)
		s.game.PlayMove(child.move, true)
		// log.Debug().Msgf("%vState is now %v", depthDbg,
		// s.game.String())
		v := s.alphabeta(child, depth-1, α, β, true)
		s.game.UnplayLastMove()
		// log.Debug().Msgf("%vAfter unplay, state is now %v", depthDbg, s.game.String())
		if v < value {
			value = v
			// log.Debug().Msgf("%vFound a worse move: %v (%v)", depthDbg, value, tm)
		}
		β = min(β, value)
		if α >= β {
			// log.Debug().Msgf("%valpha cut-off: %v>=%v", depthDbg, α, β)
			break // alpha cut-off
		}
		node.children = append(node.children, child)
	}
	node.calculatedValue = true
	node.heuristicValue = value
	return value
}
