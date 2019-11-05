// Package alphabeta implements an endgame solver using depth-limited
// minimax with alpha-beta pruning.
package alphabeta

import (
	"github.com/domino14/macondo/alphabet"
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
	move *move.Move
	// each node has an entire representation of the game state, prior to making
	// the `move`. This may have to be changed if there are memory concerns.
	// state *board.GameBoard
	// the `rack` that this move is made from.
	rack *alphabet.Rack
	// the opponent's rack
	oppRack         *alphabet.Rack
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

func (g *gameNode) isTerminal() bool {

	return false
}

func (g *gameNode) value() int {
	if !g.calculatedValue {
		g.calculateValue()
		g.calculatedValue = true
	}
	return g.heuristicValue
}

func (g *gameNode) calculateValue() {
	// calculate the heuristic value of this node, and store it.
	// if the player is out it's easy.
	g.heuristicValue = 2
}

// Init initializes the solver
func (s *Solver) Init(movegen *movegen.GordonGenerator, game *mechanics.XWordGame) {
	s.movegen = movegen
	s.game = game
	s.totalNodes = 0
}

func (s *Solver) generateMoves(rack *alphabet.Rack, oppRack *alphabet.Rack,
	parent *gameNode) []*gameNode {
	s.movegen.GenAll(rack)
	children := []*gameNode{}
	for _, m := range s.movegen.Plays() {
		children = append(children, &gameNode{
			move:    m,
			rack:    rack,
			oppRack: oppRack,
			parent:  parent,
		})
	}
	return children
}

// Solve solves the endgame
func (s *Solver) Solve(playerOnTurn, opponent *alphabet.Rack) *move.Move {
	// Generate children moves.
	s.movegen.SetSortingParameter(movegen.SortByScore)
	defer s.movegen.SetSortingParameter(movegen.SortByEquity)
	s.movegen.GenAll(playerOnTurn)
	n := &gameNode{}
	n.children = s.generateMoves(playerOnTurn, opponent, n)
	// Look 6 plies for now. This might still be very slow.
	m, v := s.alphabeta(n, 6, -Infinity, Infinity, true)
	log.Debug().Msgf("m %v, v %v", m, v)
	return m
}

func (s *Solver) alphabeta(node *gameNode, depth int, α int, β int,
	maximizingPlayer bool) (*move.Move, int) {

	if depth == 0 || node.isTerminal() {
		return node.move, node.value()
	}
	// Generate children if they don't exist.
	if node.children == nil {
		node.children = s.generateMoves(node.rack, node.oppRack, node)
	}

	if maximizingPlayer {
		value := -Infinity
		var tm *move.Move
		for _, child := range node.children {
			m, v := s.alphabeta(child, depth-1, α, β, false)
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
		m, v := s.alphabeta(child, depth-1, α, β, true)
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
