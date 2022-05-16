package strategy

import (
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cache"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/move"
)

type Leaves interface {
	LeaveValue(leave alphabet.MachineWord) float64
}

type BlankLeaves struct{}

func (b *BlankLeaves) LeaveValue(leave alphabet.MachineWord) float64 {
	return float64(0.0)
}

// ExhaustiveLeaveStrategy should apply an equity calculation for all leaves
// exhaustively.
type ExhaustiveLeaveStrategy struct {
	leaveValues                Leaves
	preEndgameAdjustmentValues []float64
}

func defaultForLexicon(lexiconName string) string {
	if strings.HasPrefix(lexiconName, "CSW") ||
		strings.HasPrefix(lexiconName, "TWL") ||
		strings.HasPrefix(lexiconName, "NWL") ||
		strings.HasPrefix(lexiconName, "ECWL") || // obsolete name for CEL
		strings.HasPrefix(lexiconName, "CEL") || // common english words
		strings.HasPrefix(lexiconName, "NSWL") {

		return "default_english"
	} else if strings.HasPrefix(lexiconName, "RD") {
		return "german"
	} else if strings.HasPrefix(lexiconName, "NSF") {
		return "norwegian"
	} else if strings.HasPrefix(lexiconName, "FRA") {
		return "french"
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
		log.Err(err).Msg("loading-leaves")
	}
	pegValues, err := cache.Load(cfg, "pegfile:"+lexiconName+":"+pegfile, PEGCacheLoadFunc)
	if err != nil {
		log.Err(err).Msg("loading-peg-values")
	}
	var ok bool
	strategy.leaveValues, ok = leaves.(*OldLeaves)
	if !ok {
		log.Info().Msg("no leaves found, will use greedy strategy")
		strategy.leaveValues = &BlankLeaves{}
	}
	strategy.preEndgameAdjustmentValues, ok = pegValues.([]float64)
	if !ok {
		log.Info().Msg("no peg values found, will use no pre-endgame strategy")
		strategy.preEndgameAdjustmentValues = []float64{}
	}
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
