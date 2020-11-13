package strategy

import (
	"encoding/binary"
	"math"
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cache"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/move"
)

// ExhaustiveLeaveStrategy should apply an equity calculation for all leaves
// exhaustively.
type ExhaustiveLeaveStrategy struct {
	leaveValues                *OldLeaves
	preEndgameAdjustmentValues []float64
}

func float32FromBytes(bytes []byte) float32 {
	bits := binary.BigEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

func defaultForLexicon(lexiconName string) string {
	if strings.HasPrefix(lexiconName, "CSW") ||
		strings.HasPrefix(lexiconName, "TWL") ||
		strings.HasPrefix(lexiconName, "NWL") {

		return "default_english"
	}
	return ""
}

func NewExhaustiveLeaveStrategy(lexiconName string,
	alph *alphabet.Alphabet, cfg *config.Config, leaveFilename, pegfile string) (
	*ExhaustiveLeaveStrategy, error) {

	// Alphabet doesn't matter yet...
	strategy := &ExhaustiveLeaveStrategy{}
	if leaveFilename == "" {
		leaveFilename = LeaveFilename
	}
	if pegfile == "" {
		pegfile = PEGAdjustmentFilename
	}

	leaves, err := cache.Load(cfg, "leavefile:"+lexiconName+":"+leaveFilename, LeaveCacheLoadFunc)
	if err != nil {
		return nil, err
	}
	pegValues, err := cache.Load(cfg, "pegfile:"+lexiconName+":"+pegfile, PEGCacheLoadFunc)
	if err != nil {
		return nil, err
	}
	strategy.leaveValues = leaves.(*OldLeaves)
	strategy.preEndgameAdjustmentValues = pegValues.([]float64)

	return strategy, nil
}

func (els ExhaustiveLeaveStrategy) Equity(play *move.Move, board *board.GameBoard,
	bag *alphabet.Bag, oppRack *alphabet.Rack) float64 {

	leave := play.Leave()
	score := play.Score()

	leaveAdjustment := 0.0
	otherAdjustments := 0.0

	// Use global placement and endgame adjustments; this is only when
	// not overriding this with an endgame player.
	if board.IsEmpty() {
		otherAdjustments += placementAdjustment(play)
	}

	if bag.TilesRemaining() > 0 {
		leaveAdjustment = els.LeaveValue(leave)
		bagPlusSeven := bag.TilesRemaining() - play.TilesPlayed() + 7
		if bagPlusSeven < len(els.preEndgameAdjustmentValues) {
			preEndgameAdjustment := els.preEndgameAdjustmentValues[bagPlusSeven]
			// log.Debug().Float64("peg-adjust", preEndgameAdjustment).Int("bagPlusSeven", bagPlusSeven).Msg("equity calc")
			otherAdjustments += preEndgameAdjustment
		}
	} else {
		// The bag is empty.
		otherAdjustments += endgameAdjustment(play, oppRack, bag.LetterDistribution())
	}

	return float64(score) + leaveAdjustment + otherAdjustments
}

func (els ExhaustiveLeaveStrategy) LeaveValue(leave alphabet.MachineWord) float64 {
	return els.leaveValues.LeaveValue(leave)
}
