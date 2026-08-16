package rangefinder

import (
	"sort"

	"github.com/domino14/word-golib/tilemapping"
)

// AnalyzeInferences renders the posterior for a person to read. This file is
// the same information as data, for callers that have to reason about it -
// the AI explainer, which needs to decide whether the read says anything worth
// telling the player before it says anything at all.

// InferredRackSummary is one leave in the posterior.
type InferredRackSummary struct {
	Leave  string  `json:"leave"`
	Weight float64 `json:"weight"`
	// Pct is this leave's share of the total posterior weight.
	Pct float64 `json:"pct"`
	// Measured is true when the leave's likelihood was evaluated rather than
	// imputed from marginal lifts.
	Measured bool `json:"measured"`
}

// TileDeviation is how much more, or less, of a tile the opponent is likely to
// be holding than a random rack would.
type TileDeviation struct {
	Tile string `json:"tile"`
	// FoundPct is the tile's share of the posterior's mass; ExpectedPct is its
	// share of the unseen pool. Their ratio is what `infer` bins into "more
	// than chance" and so on.
	FoundPct    float64 `json:"found_pct"`
	ExpectedPct float64 `json:"expected_pct"`
	Unseen      int     `json:"unseen"`
	// Tiles is the deviation expressed in tiles of a rack, which is the form
	// that means anything: a tile at three times its expected share is
	// startling as a ratio and irrelevant if that still only amounts to a
	// twentieth of a tile. Positive means the opponent likely holds more of
	// it than chance would give them.
	Tiles float64 `json:"tiles"`
}

// InferenceSummary is what the last inference concluded.
type InferenceSummary struct {
	NumRacks   int     `json:"num_racks"`
	RackLength int     `json:"rack_length"`
	Complete   bool    `json:"complete"`
	Tau        float64 `json:"tau"`
	// ESS is the effective sample size of the posterior weights. A value near
	// NumRacks means the posterior is nearly flat - the read told us little.
	ESS float64 `json:"ess"`
	// TopWeightPct is the share of the weight held by the top three leaves.
	TopWeightPct float64 `json:"top_weight_pct"`
	// Racks is the most likely leaves, most likely first.
	Racks []InferredRackSummary `json:"racks"`
	// Tiles is every tile with a non-zero presence, ordered by how far it
	// deviates from chance.
	Tiles []TileDeviation `json:"tiles"`
}

// InferenceSummary summarizes the last inference. topRacks caps how many
// leaves come back; pass 0 for all of them. It returns nil when no inference
// has been run, or when the one that ran concluded nothing.
func (r *RangeFinder) InferenceSummary(topRacks int) *InferenceSummary {
	if r.inference == nil || len(r.inference.InferredRacks) == 0 {
		return nil
	}

	// Posterior mass per tile, and the unseen pool to compare it against.
	mass := map[tilemapping.MachineLetter]float64{}
	totalMass := 0.0
	for _, ir := range r.inference.InferredRacks {
		for _, ml := range ir.Leave {
			mass[ml] += ir.Weight
			totalMass += ir.Weight
		}
	}
	bagmap := r.inferenceBagMap
	unseen := 0
	for i := range bagmap {
		unseen += int(bagmap[i])
	}
	if totalMass == 0 || unseen == 0 {
		return nil
	}

	sumW, sumW2 := 0.0, 0.0
	for _, ir := range r.inference.InferredRacks {
		sumW += ir.Weight
		sumW2 += ir.Weight * ir.Weight
	}
	s := &InferenceSummary{
		NumRacks:   len(r.inference.InferredRacks),
		RackLength: r.inference.RackLength,
		Complete:   r.inference.Complete,
		Tau:        r.Tau(),
	}
	if sumW2 > 0 {
		s.ESS = (sumW * sumW) / sumW2
	}

	alph := r.origGame.Alphabet()
	ranked := r.rankedRacks()
	for i, row := range ranked {
		if i < 3 {
			s.TopWeightPct += row.pct
		}
		if topRacks > 0 && i >= topRacks {
			break
		}
		s.Racks = append(s.Racks, InferredRackSummary{
			Leave:    tilemapping.MachineWord(row.leave).UserVisible(alph),
			Weight:   row.weight,
			Pct:      row.pct,
			Measured: row.measured,
		})
	}

	for i := range int(alph.NumLetters()) {
		ml := tilemapping.MachineLetter(i)
		if bagmap[i] == 0 && mass[ml] == 0 {
			continue
		}
		found := 100.0 * mass[ml] / totalMass
		expected := 100.0 * float64(bagmap[i]) / float64(unseen)
		s.Tiles = append(s.Tiles, TileDeviation{
			Tile:        ml.UserVisible(alph, false),
			FoundPct:    found,
			ExpectedPct: expected,
			Unseen:      int(bagmap[i]),
			Tiles:       float64(s.RackLength) * (found - expected) / 100.0,
		})
	}
	sort.Slice(s.Tiles, func(i, j int) bool {
		return abs(s.Tiles[i].Tiles) > abs(s.Tiles[j].Tiles)
	})
	return s
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
