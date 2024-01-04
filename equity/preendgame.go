package equity

import (
	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/move"
)

// PreEndgameAdjustmentCalculator returns an equity adjustment for preendgame plays.
// It should only be used if the bag is nearly empty. Note that it doesn't solve
// pre-endgames; this should only be used for simulation estimates!
type PreEndgameAdjustmentCalculator struct {
	preEndgameAdjustmentValues []float64
}

func NewPreEndgameAdjustmentCalculator(cfg *config.Config, lexiconName string, pegfile string) (*PreEndgameAdjustmentCalculator, error) {
	if pegfile == "" {
		pegfile = PEGAdjustmentFilename
	}
	pegValues, err := cache.Load(cfg.AllSettings(), "pegfile:"+lexiconName+":"+pegfile, PEGCacheLoadFunc)
	if err != nil {
		log.Err(err).Msg("loading-peg-values")
	}
	var ok bool
	calc := &PreEndgameAdjustmentCalculator{}
	calc.preEndgameAdjustmentValues, ok = pegValues.([]float64)
	if !ok {
		log.Info().Msg("no peg values found, will use no pre-endgame strategy")
		calc.preEndgameAdjustmentValues = []float64{}
	}
	return calc, nil
}

func (pac PreEndgameAdjustmentCalculator) Equity(play *move.Move, board *board.GameBoard,
	bag *tilemapping.Bag, oppRack *tilemapping.Rack) float64 {

	bagPlusSeven := bag.TilesRemaining() - play.TilesPlayed() + 7
	var preEndgameAdjustment float64
	if bagPlusSeven < len(pac.preEndgameAdjustmentValues) {
		preEndgameAdjustment = pac.preEndgameAdjustmentValues[bagPlusSeven]
		log.Debug().Float64("peg-adjust", preEndgameAdjustment).Int("bagPlusSeven", bagPlusSeven).Msg("equity calc")
	}

	return preEndgameAdjustment
}
