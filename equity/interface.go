package equity

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

// EquityCalculator is a calculator of equity.
type EquityCalculator interface {
	// Equity is a catch-all term for most post-score adjustments that
	// need to be made. It includes first-turn placement heuristic,
	// leave calculations, any pre-endgame timing heuristics, and more.
	Equity(play *move.Move, board *board.GameBoard, bag *alphabet.Bag,
		oppRack *alphabet.Rack) float64
}

type Leaves interface {
	LeaveValue(leave alphabet.MachineWord) float64
}
