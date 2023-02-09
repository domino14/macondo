package equity

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

// NoLeaveCalculator does not take leave into account at all.
type NoLeaveCalculator struct{}

func NewNoLeaveCalculator() *NoLeaveCalculator {

	return &NoLeaveCalculator{}
}

func (nls *NoLeaveCalculator) Equity(play *move.Move, board *board.GameBoard,
	bag *alphabet.Bag, oppRack *alphabet.Rack) float64 {
	score := play.Score()
	otherAdjustments := 0.0

	if board.IsEmpty() {
		otherAdjustments += placementAdjustment(play, board)
	}

	if bag.TilesRemaining() == 0 {
		otherAdjustments += endgameAdjustment(play, oppRack, bag.LetterDistribution())
	}
	return float64(score) + otherAdjustments
}

func (nls *NoLeaveCalculator) LeaveValue(leave alphabet.MachineWord) float64 {
	return 0.0
}
