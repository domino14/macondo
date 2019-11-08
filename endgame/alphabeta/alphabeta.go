// Package alphabeta implements an endgame solver using depth-limited
// minimax with alpha-beta pruning.
package alphabeta

import (
	"fmt"
	"strings"

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
	// Plies - how many to use for minimax
	Plies = 4
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
	fmt.Printf("Generating children nodes for parent %v, board %v",
		parent, s.game.Board().ToDisplayText(s.game.Alphabet()))
	s.movegen.GenAll(s.game.RackFor(s.game.PlayerOnTurn()))
	// fmt.Println(s.movegen.Plays())
	children := []*gameNode{}
	for _, m := range s.movegen.Plays() {
		children = append(children, &gameNode{
			move:   m,
			parent: parent,
		})
	}
	s.totalNodes += len(children)
	return children
}

// Solve solves the endgame given the current state of s.game, for the
// current player whose turn it is in that state.
func (s *Solver) Solve() *move.Move {
	// Generate children moves.
	s.movegen.SetSortingParameter(movegen.SortByEndgameHeuristic)
	defer s.movegen.SetSortingParameter(movegen.SortByEquity)

	// technically the children are the actual board _states_ but
	// we don't keep track of those exactly
	n := &gameNode{}
	// the root node is basically the board state prior to making any moves.
	// the children of these nodes are the board states after every move.
	// however we treat the children as those actual moves themselves.

	// Look 6 plies for now. This might still be very slow.
	m, v := s.alphabeta(n, Plies, -Infinity, Infinity, true)
	log.Debug().Msgf("Best spread found: %v", v)
	log.Debug().Msgf("Best variant found:")
	// Go down tree and find best variation:
	parent := n
	for {
		for _, child := range parent.children {
			if child.heuristicValue == v {
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

	return m
}

func (s *Solver) alphabeta(node *gameNode, depth int, α int, β int,
	maximizingPlayer bool) (*move.Move, int) {

	depthDbg := strings.Repeat(" ", depth)

	if depth == 0 || !s.game.Playing() {
		// s.game.Playing() happens if the game is over; i.e. if the
		// current node is terminal.
		log.Debug().Msgf("%vending recursion, depth: %v, playing: %v", depthDbg, depth, s.game.Playing())
		return node.move, node.value(s, maximizingPlayer)
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
			log.Debug().Msgf("%vGoing to play move %v", depthDbg, child.move)
			s.game.PlayMove(child.move, true)
			// log.Debug().Msgf("%vState is now %v", depthDbg,
			// s.game.String())
			m, v := s.alphabeta(child, depth-1, α, β, false)
			s.game.UnplayLastMove()
			log.Debug().Msgf("%vAfter unplay, state is now %v", depthDbg, s.game.String())
			if v > value {
				value = v
				tm = m
				// log.Debug().Msgf("%vFound a better move: %v (%v)", depthDbg, value, tm)
			}
			α = max(α, value)
			if α >= β {
				// log.Debug().Msgf("%vBeta cut-off: %v>=%v", depthDbg, α, β)
				break // beta cut-off
			}
		}
		node.calculatedValue = true
		node.heuristicValue = value
		return tm, value
	}
	// Otherwise, not maximizing
	value := Infinity
	var tm *move.Move
	for _, child := range node.children {
		log.Debug().Msgf("%vGoing to play move %v", depthDbg, child.move)
		s.game.PlayMove(child.move, true)
		// log.Debug().Msgf("%vState is now %v", depthDbg,
		// s.game.String())
		m, v := s.alphabeta(child, depth-1, α, β, true)
		s.game.UnplayLastMove()
		log.Debug().Msgf("%vAfter unplay, state is now %v", depthDbg, s.game.String())
		if v < value {
			value = v
			tm = m
			// log.Debug().Msgf("%vFound a worse move: %v (%v)", depthDbg, value, tm)
		}
		β = min(β, value)
		if α >= β {
			// log.Debug().Msgf("%valpha cut-off: %v>=%v", depthDbg, α, β)
			break // alpha cut-off
		}
	}
	node.calculatedValue = true
	node.heuristicValue = value
	return tm, value

}
