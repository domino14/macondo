package equity

import (
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

// EquityCalculator is a calculator of equity.
type EquityCalculator interface {
	// Equity is a catch-all term for most post-score adjustments that
	// need to be made. It includes first-turn placement heuristic,
	// leave calculations, any pre-endgame timing heuristics, and more.
	Equity(play *move.Move, board *board.GameBoard, bag *tilemapping.Bag,
		oppRack *tilemapping.Rack) float64
	// EquityWithLeave is like Equity but uses a pre-computed leave value
	// instead of looking it up. This enables O(1) leave lookups via LeaveMap.
	EquityWithLeave(play *move.Move, board *board.GameBoard, bag *tilemapping.Bag,
		oppRack *tilemapping.Rack, leaveValue float64) float64
	Type() string
}

type Leaves interface {
	LeaveValue(leave tilemapping.MachineWord) float64
}
