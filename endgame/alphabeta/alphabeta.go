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
	parent          *gameNode
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

func (g *gameNode) value(s *Solver) int {
	if !g.calculatedValue {
		g.calculateValue(s)
		g.calculatedValue = true
	}
	return g.heuristicValue
}

func (g *gameNode) calculateValue(s *Solver) {
	// calculate the heuristic value of this node, and store it.
	// Right now the heuristic is JUST going to be the current spread.
	// note that because of the way we track state, it is the state
	// in the solver right now.
	player := s.game.PlayerOnTurn()
	otherPlayer := (player + 1) % (s.game.NumPlayers())
	g.heuristicValue = s.game.PointsFor(player) - s.game.PointsFor(otherPlayer)
}

// Init initializes the solver
func (s *Solver) Init(movegen *movegen.GordonGenerator, game *mechanics.XWordGame) {
	s.movegen = movegen
	s.game = game
	s.totalNodes = 0
}

func (s *Solver) generateChildrenNodes(parent *gameNode) []*gameNode {
	s.movegen.GenAll(s.game.RackFor(s.game.PlayerOnTurn()))
	children := []*gameNode{}
	for _, m := range s.movegen.Plays() {
		children = append(children, &gameNode{
			move:   m,
			parent: parent,
		})
	}
	return children
}

// Solve solves the endgame given the current state of s.game, for the
// current player whose turn it is in that state.
func (s *Solver) Solve() *move.Move {
	// Generate children moves.
	s.movegen.SetSortingParameter(movegen.SortByEndgameHeuristic)
	defer s.movegen.SetSortingParameter(movegen.SortByEquity)

	n := &gameNode{}
	// Look 6 plies for now. This might still be very slow.
	m, v := s.alphabeta(n, 6, -Infinity, Infinity, true)
	log.Debug().Msgf("m %v, v %v", m, v)
	return m
}

func (s *Solver) alphabeta(node *gameNode, depth int, α int, β int,
	maximizingPlayer bool) (*move.Move, int) {

	if depth == 0 || !s.game.Playing() {
		// s.game.Playing() happens if the game is over; i.e. if the
		// current node is terminal.
		return node.move, node.value(s)
	}
	// Generate children if they don't exist.
	if node.children == nil {
		node.children = s.generateChildrenNodes(node)
	}

	if maximizingPlayer {
		value := -Infinity
		var tm *move.Move
		for _, child := range node.children {
			// Play the child
			s.game.PlayMove(child.move, true)
			m, v := s.alphabeta(child, depth-1, α, β, false)
			s.game.UnplayLastMove()
			if v > value {
				value = v
				tm = m
			}
			α = max(α, value)
			if α >= β {
				break // beta cut-off
			}
		}
		return tm, value
	}
	// Otherwise, not maximizing
	value := Infinity
	var tm *move.Move
	for _, child := range node.children {
		s.game.PlayMove(child.move, true)
		m, v := s.alphabeta(child, depth-1, α, β, true)
		s.game.UnplayLastMove()
		if v < value {
			value = v
			tm = m
		}
		β = min(β, value)
		if α >= β {
			break // alpha cut-off
		}
	}
	return tm, value

}
