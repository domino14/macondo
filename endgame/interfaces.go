package endgame

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// Solver is an interface for an endgame solver. The scores don't matter, the
// solver maximizes spread.
type Solver interface {
	Init(board *board.GameBoard, movegen *movegen.GordonGenerator,
		bag *alphabet.Bag, gd *gaddag.SimpleGaddag)
	Solve(playerOnTurn, opponent *alphabet.Rack) *move.Move
}
