package equity

import (
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
)

// NoLeaveCalculator does not take leave into account at all.
type NoLeaveCalculator struct{}

func NewNoLeaveCalculator() *NoLeaveCalculator {

	return &NoLeaveCalculator{}
}

func (nls *NoLeaveCalculator) Equity(play *move.Move, board *board.GameBoard,
	bag *tilemapping.Bag, oppRack *tilemapping.Rack) float64 {
	score := play.Score()
	otherAdjustments := 0.0

	if board.IsEmpty() {
		otherAdjustments += placementAdjustment(play, board, bag.LetterDistribution())
	}

	if bag.TilesRemaining() == 0 {
		otherAdjustments += endgameAdjustment(play, oppRack, bag.LetterDistribution())
	}
	return float64(score) + otherAdjustments
}

func (nls *NoLeaveCalculator) LeaveValue(leave tilemapping.MachineWord) float64 {
	return 0.0
}
