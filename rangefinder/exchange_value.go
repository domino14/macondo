package rangefinder

import (
	"math"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/word-golib/tilemapping"
)

// inferExchangeByValue reads an exchange as ExchangeValue: every leave of the
// kept size that the unseen pool can hold, weighted by its hypergeometric
// prior times exp(ExchangeValueBeta x leave value). The posterior is complete,
// so the simmer samples kept leaves from it directly.
//
// It does not see the board: a kept X for a big X spot is valued like any
// other X. The mini-sim measurements could see that, but they were too noisy
// to help (see ExchangeSim).
func (r *RangeFinder) inferExchangeByValue() error {
	k := r.inference.RackLength
	if k < 1 {
		return ErrNoInformation
	}
	valueOf := r.tuning.valueOf
	leaves := enumerateLeaves(r.inferenceBagMap, k)
	if len(leaves) == 0 {
		return nil
	}
	values := make([]float64, len(leaves))
	vmax := math.Inf(-1)
	runs := make([]tileRun, 0, k)
	for i, l := range leaves {
		if valueOf != nil {
			runs = tilesToRuns(l.tiles, runs[:0])
			values[i] = valueOf(runs)
		}
		vmax = max(vmax, values[i])
	}
	racks := make([]montecarlo.InferredRack, len(leaves))
	for i, l := range leaves {
		racks[i] = montecarlo.InferredRack{
			Leave:  l.tiles,
			Weight: l.prior * math.Exp(ExchangeValueBeta*(values[i]-vmax)),
		}
	}
	r.inference.InferredRacks = racks
	r.inference.Complete = true
	log.Info().Int("leaves", len(racks)).Int("kept", k).
		Bool("valued", valueOf != nil).Msg("exchange-value-posterior")
	return nil
}

// tilesToRuns groups sorted tiles into runs of equal tiles, appending to buf.
func tilesToRuns(tiles []tilemapping.MachineLetter, buf []tileRun) []tileRun {
	for _, t := range tiles {
		if n := len(buf); n > 0 && buf[n-1].t == t {
			buf[n-1].c++
			continue
		}
		buf = append(buf, tileRun{t: t, c: 1})
	}
	return buf
}
