package rangefinder

import (
	"fmt"
	"sort"
	"strings"

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
	// HoldsPct is the share of the posterior's weight sitting on racks that
	// contain at least one of this tile: the chance they are holding one.
	// ChanceHoldsPct is the same probability for a rack drawn at random from
	// the unseen pool.
	//
	// This is the pair to quote. "They have the Z about four times in five,
	// against one time in thirty by chance" is a fact a player can act on,
	// and it is exact - a weight sum, not a derivation.
	HoldsPct       float64 `json:"holds_pct"`
	ChanceHoldsPct float64 `json:"chance_holds_pct"`

	// FoundPct is the tile's share of the posterior's tile slots, and
	// ExpectedPct its share of the unseen pool. Their ratio is what `infer`
	// bins into "more than chance" and so on. Neither is a probability, and
	// neither reads like one, so they stay out of anything a model sees.
	FoundPct    float64 `json:"found_pct"`
	ExpectedPct float64 `json:"expected_pct"`
	Unseen      int     `json:"unseen"`
	// Tiles is the deviation expressed in tiles of a rack: the expected
	// number of copies they hold, minus the expected number a random rack
	// would hold. It is the right measure of how big a read is - a tile at
	// three times its expected share is startling as a ratio and irrelevant
	// if that still only amounts to a twentieth of a tile - and unlike
	// HoldsPct it can see the difference between holding one and holding two.
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

	tiles := r.tileDeviations()
	if tiles == nil {
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

	// Tiles nobody could be holding and nobody is holding say nothing; the
	// rest come back with the biggest read first.
	for _, t := range tiles {
		if t.Unseen > 0 || t.FoundPct > 0 {
			s.Tiles = append(s.Tiles, t)
		}
	}
	sort.Slice(s.Tiles, func(i, j int) bool {
		return abs(s.Tiles[i].Tiles) > abs(s.Tiles[j].Tiles)
	})
	return s
}

// tileDeviations computes the per-tile read for every letter of the alphabet,
// in alphabet order. It is shared by InferenceSummary and by the table
// AnalyzeInferences prints, so the two can't drift apart. Returns nil when
// there is no posterior to read.
func (r *RangeFinder) tileDeviations() []TileDeviation {
	if r.inference == nil || len(r.inference.InferredRacks) == 0 {
		return nil
	}

	// Two different tallies per tile. mass counts copies, so it answers "how
	// many are they holding"; holding counts each rack once however many
	// copies it has, so it answers "are they holding one at all".
	mass := map[tilemapping.MachineLetter]float64{}
	holding := map[tilemapping.MachineLetter]float64{}
	totalMass, sumW := 0.0, 0.0
	for _, ir := range r.inference.InferredRacks {
		sumW += ir.Weight
		seen := map[tilemapping.MachineLetter]bool{}
		for _, ml := range ir.Leave {
			mass[ml] += ir.Weight
			totalMass += ir.Weight
			if !seen[ml] {
				seen[ml] = true
				holding[ml] += ir.Weight
			}
		}
	}
	bagmap := r.inferenceBagMap
	unseen := 0
	for i := range bagmap {
		unseen += int(bagmap[i])
	}
	if totalMass == 0 || unseen == 0 || sumW == 0 {
		return nil
	}

	rackLen := r.inference.RackLength
	alph := r.origGame.Alphabet()
	out := make([]TileDeviation, 0, alph.NumLetters())
	for i := range int(alph.NumLetters()) {
		ml := tilemapping.MachineLetter(i)
		found := 100.0 * mass[ml] / totalMass
		expected := 100.0 * float64(bagmap[i]) / float64(unseen)
		out = append(out, TileDeviation{
			Tile:           ml.UserVisible(alph, false),
			HoldsPct:       100.0 * holding[ml] / sumW,
			ChanceHoldsPct: 100.0 * chanceOfHolding(unseen, int(bagmap[i]), rackLen),
			FoundPct:       found,
			ExpectedPct:    expected,
			Unseen:         int(bagmap[i]),
			Tiles:          float64(rackLen) * (found - expected) / 100.0,
		})
	}
	return out
}

// NotableTiles is how far a tile has to sit from chance, measured in tiles of
// a rack, before it is worth pointing at. A ratio is the wrong scale here: a
// tile at three times its expected share is startling until you notice that
// still only amounts to a twentieth of a tile.
const NotableTiles = 0.4

const (
	graphBands    = 10 // 0-10%, 10-20%, ... 90-100%
	graphBandSize = 100.0 / graphBands
)

// HoldingGraph draws the read as a picture: one row per band of "chance they
// are holding at least one", with the tiles that land in it. Tiles nobody can
// be holding are listed underneath instead of being given a band, because 0%
// for a tile that has all been played means something different from 0% for
// one the read has ruled out.
//
// The bands are on the probability rather than on the ratio to chance, which
// is what this view used to bin. The ratio buries the thing that matters: with
// one Z left in the pool, a read putting it in their rack 70% of the time is
// enormous, and "way more than chance" is the same phrase that view would give
// a tile it had nudged from 1% to 3%.
func HoldingGraph(tiles []TileDeviation) string {
	bands := make([][]string, graphBands)
	played := []string{}
	standouts := []TileDeviation{}

	for _, t := range tiles {
		if t.Unseen == 0 {
			played = append(played, t.Tile)
			continue
		}
		label := t.Tile
		switch {
		case t.Tiles >= NotableTiles:
			label += "+"
			standouts = append(standouts, t)
		case t.Tiles <= -NotableTiles:
			label += "-"
			standouts = append(standouts, t)
		}
		band := min(int(t.HoldsPct/graphBandSize), graphBands-1)
		bands[band] = append(bands[band], label)
	}

	var ss strings.Builder
	ss.WriteString("How likely they are to be holding at least one of each tile.\n")
	ss.WriteString("+ and - mark tiles the read puts well above or below what chance alone gives.\n\n")
	for b := graphBands - 1; b >= 0; b-- {
		row := fmt.Sprintf("  %3d-%3d%% | %s", b*int(graphBandSize), (b+1)*int(graphBandSize),
			strings.Join(bands[b], " "))
		ss.WriteString(strings.TrimRight(row, " ") + "\n")
	}
	if len(played) > 0 {
		fmt.Fprintf(&ss, "\n  All played, none left to hold: %s\n", strings.Join(played, " "))
	}

	// The graph shows the shape; this says what the shape is made of, because
	// a band is a ten-point range and the difference between 71% and 79% is
	// worth having.
	if len(standouts) > 0 {
		sort.Slice(standouts, func(i, j int) bool {
			return abs(standouts[i].Tiles) > abs(standouts[j].Tiles)
		})
		parts := make([]string, 0, len(standouts))
		for _, t := range standouts {
			parts = append(parts, fmt.Sprintf("%s %.0f%% (chance %.0f%%)",
				t.Tile, t.HoldsPct, t.ChanceHoldsPct))
		}
		fmt.Fprintf(&ss, "\n  Standouts: %s\n", strings.Join(parts, ", "))
	}
	return ss.String()
}

// chanceOfHolding is the probability that a rack of rackLen tiles drawn at
// random from a pool of unseen tiles containing copies of some tile picks up
// at least one of them. It is one minus the hypergeometric probability of
// missing every copy, written as a running product so that pools far larger
// than a rack don't need factorials.
//
// For a tile with a single copy this comes out to rackLen/unseen, which is the
// familiar answer: three chances out of eighty-three to draw the only Z.
func chanceOfHolding(unseen, copies, rackLen int) float64 {
	if copies <= 0 || rackLen <= 0 || unseen <= 0 {
		return 0
	}
	if unseen-copies < rackLen {
		return 1 // not enough other tiles to fill a rack without one
	}
	missAll := 1.0
	for i := range rackLen {
		missAll *= float64(unseen-copies-i) / float64(unseen-i)
	}
	return 1 - missAll
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
