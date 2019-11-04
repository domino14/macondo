// Package endgame implements a crossword game endgame solver.
// The endgame solver uses the B* algorithm.
// See "The B* Tree Search Algorithm: A Best-First Proof Procedure"
// by Hans Berliner
package endgame

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// BStarSolver implements the B* algorithm.
type BStarSolver struct {
	movegen *movegen.GordonGenerator
	// playerOnTurnRack *alphabet.Rack
	// opponentRack     *alphabet.Rack
	board *board.GameBoard
	gd    *gaddag.SimpleGaddag
	bag   *alphabet.Bag
}

func (b *BStarSolver) Init(board *board.GameBoard, movegen *movegen.GordonGenerator,
	bag *alphabet.Bag, gd *gaddag.SimpleGaddag) {

	b.board = board
	b.movegen = movegen
	b.bag = bag
	b.gd = gd
}

func (b *BStarSolver) Solve(playerOnTurn, opponent *alphabet.Rack) *move.Move {

	return nil
}

func (b *BStarSolver) evaluationFunc(rack *alphabet.Rack, opponentRack *alphabet.Rack) {
	// From Maven paper.
	// Return an evaluation for this rack. For every move that can be played out of
	// this rack, compute the score of this move plus the evaluation of the remaining
	// tiles assuming that N-1 turns remain. The value of a rack in N turns is
	// the largest value of that function over all possible moves.
	b.movegen.SetSortingParameter(movegen.SortByScore)
	defer b.movegen.SetSortingParameter(movegen.SortByEquity)

	b.movegen.GenAll(rack)
	// myWinningPlay := b.movegen.Plays()[0]

	// Generate all opponent moves.
	b.movegen.GenAll(opponentRack)
	// oppWinningPlay := b.movegen.Plays()[0]

	// Play the move, but back up the board.
	// b.board.PlayMove(winningPlay, b.gd, b.bag, true)
	// winningPlay.Leave()

}
