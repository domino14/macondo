package strategy

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

// Strategizer is in charge of picking plays using strategy.
type Strategizer interface {
	// Equity is a catch-all term for most post-score adjustments that
	// need to be made. It includes first-turn placement heuristic,
	// leave calculations, any pre-endgame timing heuristics, and more.
	Equity(play *move.Move, board *board.GameBoard, bag *alphabet.Bag,
		oppRack *alphabet.Rack) float64

	LeaveValue(leave alphabet.MachineWord) float64
}

// Global strategy heuristics, available to all strategies.

func placementAdjustment(play *move.Move) float64 {
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
		if play.Tiles()[j-start].IsVowel(play.Alphabet()) {
			switch j {
			case 2, 6, 8, 12:
				// row/col below/above have a 2LS. note this only works
				// for specific board configurations
				penalty += vPenalty
			default:

			}
		}
		j++
	}
	return penalty
}

func endgameAdjustment(play *move.Move, oppRack *alphabet.Rack, ld *alphabet.LetterDistribution) float64 {
	if len(play.Leave()) != 0 {
		// This play is not going out. We should penalize it by our own score
		// plus some constant. XXX: Determine this in a better way.
		return -float64(play.Leave().Score(ld))*2 - 10
	}
	// Otherwise, this play goes out. Apply opp rack.
	if oppRack == nil {
		return 0
	}
	return 2 * float64(oppRack.ScoreOn(ld))
}
