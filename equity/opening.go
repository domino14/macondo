package equity

import (
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

// OpeningAdjustmentCalculator returns an equity adjustment for an opening play.
// It should only be used if the board is empty.
type OpeningAdjustmentCalculator struct{}

func (oac OpeningAdjustmentCalculator) Equity(play *move.Move, board *board.GameBoard,
	bag *tilemapping.Bag, oppRack *tilemapping.Rack) float64 {

	if !board.IsEmpty() {
		return 0.0
	}
	return placementAdjustment(play, board, bag.LetterDistribution())
}

func (oac OpeningAdjustmentCalculator) EquityWithLeave(play *move.Move, board *board.GameBoard,
	bag *tilemapping.Bag, oppRack *tilemapping.Rack, leaveValue float64) float64 {
	// OpeningAdjustmentCalculator doesn't use leave value
	return oac.Equity(play, board, bag, oppRack)
}

func (oac OpeningAdjustmentCalculator) Type() string {
	return "OpeningAdjustmentCalculator"
}

func placementAdjustment(play *move.Move, board *board.GameBoard, ld *tilemapping.LetterDistribution) float64 {
	// Very simply just checks how many vowels are overlapping bonus squares.
	// This only gets considered when the board is empty.
	if play.Action() != move.MoveTypePlay {
		return 0
	}
	row, col, vertical := play.CoordsAndVertical()
	var start, end int
	start = col
	if vertical {
		start = row
	}
	end = start + play.TilesPlayed()

	j := start
	penalty := 0.0
	vPenalty := -0.7 // VERY ROUGH approximation from Maven paper.
	for j < end {
		if play.Tiles()[j-start].IsVowel(ld) {
			if board.Dim() == 15 {
				switch j {
				case 2, 6, 8, 12:
					// row/col below/above have a 2LS. note this only works
					// for specific board configurations
					penalty += vPenalty
				default:

				}
			} else if board.Dim() == 21 {
				switch j {
				case 5, 9, 11, 15:
					// see above, 2LS for this board dim (fix me later, this is ugly)
					penalty += vPenalty
				default:

				}
			}
		}
		j++
	}
	return penalty
}
