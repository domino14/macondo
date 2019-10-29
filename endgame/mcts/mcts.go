// Package mcts implements a Monte-Carlo Tree Search endgame solver.
// Although Scrabble is a perfect-information game during the endgame,
// the branching factor and number of possible turns is huge (e.g. 200 possible
// moves with both opps one-tiling each other results in 200^14 states,
// which makes it unfeasible to implement regular minimax.
// We will implement MCTS to see if it helps in finding the optimal endgame,
// with some minor modifications:
//    1. Instead of tracking wins and losses as the value of each node, we will
//    track spread at the end of the rollout. That way the value we are
//    maximizing is spread.
//    2. The _rollout_ will, instead of picking random moves, pick the
//    _highest scoring_ move. This is a reasonable estimate for the goodness
//    of a play, since MCTS is typically used for games where it is difficult
//    to score a play, whereas in Scrabble the score of a play and winning
//    the game have a strong correlation.
//        - We can experiment with this. Maybe we can pick weighted random
//        or just random to see if that does any better.
//
// We will keep track of total nodes to bound memory, and use the UCB1
// function for picking nodes.
package mcts

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// Solver implements the MCTS algorithm.
type Solver struct {
	movegen *movegen.GordonGenerator
	board   *board.GameBoard
	gd      *gaddag.SimpleGaddag
	bag     *alphabet.Bag

	totalNodes int
	nodes      map[stateHistory]*mctsNode
}

type stateHistory string

type mctsNode struct {
	move *move.Move // or a description, to save space?
	// state should be a string representation of the board, rather
	// than the whole complex Board structure.
	state    string
	nPlays   int
	value    int // The total value of this node. It will be the total spread.
	parent   *mctsNode
	children map[string]*mctsNode // these children nodes should be null, until expanded.
}

// Init initializes the Solver.
func (s *Solver) Init(board *board.GameBoard, movegen *movegen.GordonGenerator,
	bag *alphabet.Bag, gd *gaddag.SimpleGaddag) {

	s.board = board
	s.movegen = movegen
	s.bag = bag
	s.gd = gd

	s.totalNodes = 0
}

// Solve solves the endgame.
func (s *Solver) Solve(playerOnTurn, opponent *alphabet.Rack) *move.Move {
	return nil
}

func (s *Solver) makeMCTSNode(board *board.Board, rack *alphabet.Rack,
	hash stateHistory) {
	// Make a mctsNode from the given board position.
	if n := s.nodes[hash]; n == nil {
		// Generate all moves for the current player.
		s.movegen.GenAll(rack)
		node := &mctsNode{
			parent:          nil,
			move:            nil,
			state:           state,
			unexpandedPlays: genPlays,
		}
	}
}
