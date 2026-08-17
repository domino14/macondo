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
	// Shape is what the read says about the makeup of the rack rather than
	// about any one tile. Some reads live entirely here: no letter moves far
	// on its own and the sum of them is the whole finding.
	Shape *RackShape `json:"shape,omitempty"`
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
	// Ordered by how far the read moved each tile, in probability, which is
	// what makes the list readable as a finding: what the read favours at the
	// top, what it rules out at the bottom.
	sort.Slice(s.Tiles, func(i, j int) bool {
		return abs(s.Tiles[i].HoldsPct-s.Tiles[i].ChanceHoldsPct) >
			abs(s.Tiles[j].HoldsPct-s.Tiles[j].ChanceHoldsPct)
	})
	s.Shape = r.rackShape()
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

// graphHalf is how many characters each side of the centre line gets. The
// widest movement in the read fills it and everything else is drawn to scale
// against that, because the interesting comparison is between tiles in this
// read, not against some absolute that most positions never approach.
const graphHalf = 22

// MovementGraph draws what the read changed: one row per tile, sorted by how
// far it moved, with a centre line at "exactly what chance gives" and a bar
// running right for tiles the read makes likelier and left for tiles it rules
// out.
//
// It replaced a histogram banded on the probability itself, which inverted the
// signal it was supposed to show. Height on that graph was mostly a function
// of how many copies were left in the bag - nine unseen I's put I near the top
// whatever the read said - so on a position whose whole finding was "they kept
// consonants", the vowels floated to the top and the consonants sat in the
// bottom band. Sorting by movement takes pool size out of the picture
// entirely, and the shape of the list becomes the finding.
//
// Bars are scaled against the largest movement in this read rather than
// against a fixed span. Most reads move nothing more than ten points, and a
// fixed scale spends the width on room nothing ever uses.
func MovementGraph(tiles []TileDeviation) string {
	moved := []TileDeviation{}
	played := []string{}
	widest := 0.0
	for _, t := range tiles {
		if t.Unseen == 0 {
			// Not "they don't have it" but "nobody can", which is a different
			// fact and belongs in a different sentence.
			played = append(played, t.Tile)
			continue
		}
		moved = append(moved, t)
		widest = max(widest, abs(t.HoldsPct-t.ChanceHoldsPct))
	}
	if len(moved) == 0 {
		return ""
	}
	sort.SliceStable(moved, func(i, j int) bool {
		return moved[i].HoldsPct-moved[i].ChanceHoldsPct > moved[j].HoldsPct-moved[j].ChanceHoldsPct
	})

	var ss strings.Builder
	ss.WriteString("Per tile, sorted by how far the read moved it.\n")
	ss.WriteString("Left of the line: less likely than chance. Right of it: more.\n\n")
	for _, t := range moved {
		diff := t.HoldsPct - t.ChanceHoldsPct
		n := 0
		if widest > 0 {
			n = int(abs(diff)/widest*graphHalf + 0.5)
		}
		var bar string
		if diff >= 0 {
			bar = strings.Repeat(" ", graphHalf) + "|" + strings.Repeat("+", n)
		} else {
			bar = strings.Repeat(" ", graphHalf-n) + strings.Repeat("-", n) + "|"
		}
		fmt.Fprintf(&ss, "  %-2s %-*s  %5.1f -> %5.1f%%  %+5.1f  (%d unseen)\n",
			t.Tile, 2*graphHalf+1, bar, t.ChanceHoldsPct, t.HoldsPct, diff, t.Unseen)
	}
	if len(played) > 0 {
		fmt.Fprintf(&ss, "\n  All played, none left to hold: %s\n", strings.Join(played, " "))
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
