package equity

import (
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
)

// EndgameAdjustmentCalculator returns an equity adjustment for endgame plays.
// It should only be used if the bag is empty. Note that it doesn't solve
// the endgame; this should only be used for simulation estimates!
type EndgameAdjustmentCalculator struct{}

func (eac EndgameAdjustmentCalculator) Equity(play *move.Move, board *board.GameBoard,
	bag *tilemapping.Bag, oppRack *tilemapping.Rack) float64 {

	if bag.TilesRemaining() > 0 {
		return 0.0
	}
	return endgameAdjustment(play, oppRack, bag.LetterDistribution())
}

func endgameAdjustment(play *move.Move, oppRack *tilemapping.Rack, ld *tilemapping.LetterDistribution) float64 {
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
