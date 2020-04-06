package strategy

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

// NoLeaveStrategy does not take leave into account at all.
type NoLeaveStrategy struct{}

func (nls *NoLeaveStrategy) Equity(play *move.Move, board *board.GameBoard,
	bag *alphabet.Bag, oppRack *alphabet.Rack) float64 {
	score := play.Score()
	adjustment := 0.0
	if board.IsEmpty() {
		adjustment += placementAdjustment(play)
	}
	return float64(score) + adjustment
}

func (nls *NoLeaveStrategy) LeaveValue(leave alphabet.MachineWord) float64 {
	return 0.0
}
