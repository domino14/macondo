package equity

import (
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
)

// EquityCalculator is a calculator of equity.
type EquityCalculator interface {
	// Equity is a catch-all term for most post-score adjustments that
	// need to be made. It includes first-turn placement heuristic,
	// leave calculations, any pre-endgame timing heuristics, and more.
	Equity(play *move.Move, board *board.GameBoard, bag *tilemapping.Bag,
		oppRack *tilemapping.Rack) float64
}

type Leaves interface {
	LeaveValue(leave tilemapping.MachineWord) float64
}
