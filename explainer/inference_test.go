package explainer

import (
	"slices"
	"strings"
	"testing"

	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/rangefinder"
	"github.com/matryer/is"
)

// tile builds one row of the per-tile read. pct figures are shares; the
// deviation in tiles of a rack is what the gate actually looks at.
func tile(name string, found, expected float64, unseen, rackLen int) rangefinder.TileDeviation {
	return rangefinder.TileDeviation{
		Tile: name, FoundPct: found, ExpectedPct: expected, Unseen: unseen,
		Tiles: float64(rackLen) * (found - expected) / 100.0,
	}
}

// inferredFacts is fakeFacts with a read attached. baselineBest names the play
// the no-inference sim preferred, and baselineWin is what it gave the
// recommended play.
func inferredFacts(tiles []rangefinder.TileDeviation, baselineBest string, baselineWin, baselineCI float64) *PositionFacts {
	f := fakeFacts()
	baseline := []montecarlo.CandidateStats{
		{Play: baselineBest, WinPct: baselineWin + 5, WinPctCI: baselineCI},
	}
	// The recommended play's own showing in the baseline. When the baseline
	// preferred a different play, it goes in second.
	if baselineBest == f.Best.Play {
		baseline[0].WinPct = baselineWin
	} else {
		baseline = append(baseline, montecarlo.CandidateStats{
			Play: f.Best.Play, WinPct: baselineWin, WinPctCI: baselineCI,
		})
	}

	f.Inference = buildInference(f, &InferenceInput{
		Summary: &rangefinder.InferenceSummary{
			NumRacks: 240, RackLength: 7, Complete: true, ESS: 12.4,
			TopWeightPct: 41.0,
			Racks: []rangefinder.InferredRackSummary{
				{Leave: "AEINRS", Pct: 18.2, Measured: true},
				{Leave: "AEINRT", Pct: 13.5, Measured: true},
				{Leave: "AEIRST", Pct: 9.3},
			},
			Tiles: tiles,
		},
		Baseline: baseline,
	})
	f.Flags = computeFlags(f)
	return f
}

// A tile three times likelier than chance is startling as a ratio and
// meaningless if it still amounts to a twentieth of a tile. The gate measures
// in tiles of a rack for exactly that reason.
func TestOutliersAreMeasuredInTiles(t *testing.T) {
	is := is.New(t)

	f := inferredFacts([]rangefinder.TileDeviation{
		tile("S", 22.0, 8.0, 3, 7),  // +0.98 tiles - a real read
		tile("Q", 3.0, 1.0, 1, 7),   // +0.14 tiles - 3x expected, and noise
		tile("V", 0.2, 4.0, 2, 7),   // -0.27 tiles - likelier absent, but barely
		tile("E", 4.0, 15.0, 6, 7),  // -0.77 tiles - a real absence
		tile("A", 11.0, 11.4, 5, 7), // about as expected
	}, "5D (S)CAP(A)", 30.0, 0.9)

	inf := f.Inference
	is.True(inf.Informative)

	got := []string{}
	for _, o := range inf.Outliers {
		got = append(got, o.Tile)
	}
	// Biggest deviation first, and only the ones worth a sentence.
	is.Equal(got, []string{"S", "E"})
}

// Both halves have to be true before the subject comes up at all: a read that
// concluded nothing, and a read that changed nothing, are both silence.
func TestInferenceIsSilentUnlessItMatters(t *testing.T) {
	is := is.New(t)

	// Nothing deviates from chance: no read to speak of, even though the
	// baseline would have played something else.
	flat := inferredFacts([]rangefinder.TileDeviation{
		tile("S", 8.2, 8.0, 3, 7),
		tile("E", 15.1, 15.0, 6, 7),
	}, "5D (S)CAP(A)", 30.0, 0.9)
	is.True(!flat.Inference.Informative)
	is.True(!flat.Inference.Matters)
	is.True(!flat.Flags["has_inference"])

	// A strong read that moved neither the recommendation nor the win% past
	// the confidence intervals. Interesting, but not a lesson.
	inert := inferredFacts([]rangefinder.TileDeviation{
		tile("S", 22.0, 8.0, 3, 7),
	}, "12K QU(ID)", 37.5, 2.0)
	is.True(inert.Inference.Informative)
	is.True(!inert.Inference.ChangedTopPlay)
	is.True(!inert.Inference.Established)
	is.True(!inert.Inference.Matters)
	is.True(!inert.Flags["has_inference"])

	p, err := BuildPrompt(inert, false)
	is.NoErr(err)
	is.True(!slices.Contains(p.Concepts, "inference"))
	is.True(!strings.Contains(p.User, "gave away"))
	is.True(!strings.Contains(p.User, "AEINRS"))
}

// Statistical significance is not enough on its own. Two well-converged sims
// have narrow intervals, and in a position that is already won the win
// probabilities saturate and their intervals shrink to almost nothing - so a
// shift can clear the interval test while meaning nothing to a player. These
// are the real figures from a position that behaved exactly that way.
func TestInferenceIsSilentOnDifferencesThatDontMatter(t *testing.T) {
	is := is.New(t)
	strongRead := []rangefinder.TileDeviation{tile("R", 19.1, 5.1, 2, 7)}

	// withWinPct rebuilds the read after setting what the recommended play
	// scored with it, and how tight that figure is.
	withWinPct := func(winPct, ci, baselineWin float64) *PositionFacts {
		f := inferredFacts(strongRead, "12K QU(ID)", baselineWin, 0.2)
		f.Best.WinPct, f.Best.WinPctCI = winPct, ci
		f.Inference = buildInference(f, &InferenceInput{
			Summary: f.Inference.Summary, Baseline: f.Inference.Baseline,
		})
		f.Flags = computeFlags(f)
		return f
	}

	// 96.97% with the read against 97.91% without it: outside the intervals,
	// and the game is over either way.
	won := withWinPct(96.97, 0.2, 97.91)
	is.True(won.Inference.Established) // the statistics are real
	is.True(!won.Inference.Decisive)   // and they don't matter
	is.True(!won.Flags["has_inference"])

	// A contested position, tight intervals, but the read only moved things
	// half a point.
	slight := withWinPct(37.8, 0.1, 37.3)
	is.True(slight.Inference.Established)
	is.True(!slight.Inference.Decisive)
	is.True(!slight.Flags["has_inference"])

	// The same contested position with the read worth a couple of points is
	// worth telling the player about.
	real := withWinPct(37.8, 0.1, 35.3)
	is.True(real.Inference.Decisive)
	is.True(real.Flags["has_inference"])
}

// The strongest thing a read can do is recommend a different play.
func TestInferenceThatChangesThePlay(t *testing.T) {
	is := is.New(t)

	// Figures shaped like a real read: the game's only Z, three tiles kept,
	// 83 unseen. HoldsPct is counted straight off the posterior - the weight
	// of the racks that contain a Z - and 3.6% is 3 draws from 83.
	//
	// The tile-slot figures agree with it here, which is worth knowing when
	// reading a dump by hand: a rack holds 3 slots and the single Z can only
	// fill one, so its 26.2% share of slots is a 3 x 26.2% = 78.6% share of
	// racks. That identity only holds for a tile with one copy, which is why
	// HoldsPct is counted rather than derived.
	z := tile("Z", 26.2, 1.2, 1, 3)
	z.HoldsPct, z.ChanceHoldsPct = 78.6, 3.6
	e := tile("E", 4.0, 15.0, 6, 7)
	e.HoldsPct, e.ChanceHoldsPct = 21.0, 62.0

	f := inferredFacts([]rangefinder.TileDeviation{z, e}, "5D (S)CAP(A)", 30.0, 0.9)

	inf := f.Inference
	is.True(inf.Matters)
	is.True(inf.ChangedTopPlay)
	is.Equal(inf.BaselineBest.Play, "5D (S)CAP(A)")
	is.Equal(inf.BaselineOfBest.Play, "12K QU(ID)")
	is.True(inf.WinPctShift > 0) // 37.8 with the read, 30.0 without
	is.True(inf.Established)     // and outside both intervals

	is.True(f.Flags["has_inference"])
	is.True(f.Flags["inference_changed_play"])

	p, err := BuildPrompt(f, false)
	is.NoErr(err)
	is.True(slices.Contains(p.Concepts, "inference"))
	is.True(slices.Contains(p.Concepts, "inference-changed-play"))

	is.True(strings.Contains(p.User, "### What the opponent's last play gave away"))
	is.True(strings.Contains(p.User, "AEINRS"))

	// The probability leads, because it is the thing a player can act on.
	is.True(strings.Contains(p.User,
		"Z    holding one 78.6% of the time, against 3.6% by chance"))
	is.True(strings.Contains(p.User, "+0.75 tiles more than chance"))
	is.True(strings.Contains(p.User, "-0.77 tiles fewer than chance"))
	is.True(strings.Contains(p.User,
		"Without the read, 5D (S)CAP(A) was the best play. With it, 12K QU(ID) is."))

	// A share of the posterior's tile slots is not a probability and reads
	// exactly like one, so it stays out of the prompt entirely. The model
	// really did turn "26.2% of the read" into "26% of their likely racks".
	is.True(!strings.Contains(p.User, "of the read vs"))
	is.True(!strings.Contains(p.User, "of the unseen pool"))

	// The tail of near-expected tiles is left out.
	is.True(!strings.Contains(p.User, "A    "))
}

// The two sims assume different things about the opponent's rack, so a play's
// win% in one is not the same quantity as its win% in the other. When the top
// play changes, what the reader needs is how the two plays stood inside each
// simulation.
func TestChangedPlayShowsBothPlaysInBothSims(t *testing.T) {
	is := is.New(t)

	f := inferredFacts([]rangefinder.TileDeviation{
		tile("S", 22.0, 8.0, 3, 7),
	}, "5D (S)CAP(A)", 30.0, 0.9)
	// Give the dethroned play a showing in the inference sim too, so both
	// standings can be printed.
	f.Candidates[1].WinPct = 26.4

	p, err := BuildPrompt(f, false)
	is.NoErr(err)

	// 5D (S)CAP(A) led by 5.00 without the read; 12K QU(ID) leads by 11.40 with it.
	is.True(strings.Contains(p.User,
		"Ignoring the read: 5D (S)CAP(A) 35.00%, 12K QU(ID) 30.00% - 5D (S)CAP(A) ahead by 5.00."))
	is.True(strings.Contains(p.User,
		"Believing it:      12K QU(ID) 37.80%, 5D (S)CAP(A) 26.40% - 12K QU(ID) ahead by 11.40."))
	is.True(strings.Contains(p.User, "rather than one play across"))

	// The cross-sim figure for a single play is gone: it was the number that
	// looked like nothing had happened while the recommendation flipped.
	is.True(!strings.Contains(p.User, "with the read and"))
}

// A win% shift big enough to clear both intervals is worth reporting even when
// the recommendation stands.
func TestInferenceThatOnlyMovesTheWinPct(t *testing.T) {
	is := is.New(t)

	f := inferredFacts([]rangefinder.TileDeviation{
		tile("S", 22.0, 8.0, 3, 7),
	}, "12K QU(ID)", 30.0, 0.9)

	is.True(f.Inference.Matters)
	is.True(!f.Inference.ChangedTopPlay)
	is.True(f.Inference.Established)

	p, err := BuildPrompt(f, false)
	is.NoErr(err)
	is.True(slices.Contains(p.Concepts, "inference"))
	is.True(!slices.Contains(p.Concepts, "inference-changed-play"))
	is.True(strings.Contains(p.User, "The recommendation is 12K QU(ID) either way."))
	is.True(strings.Contains(p.User,
		"Believing the read moves its win% from 30.00% to 37.80% (+7.80)"))
}

// Without a read at all, nothing about inference reaches the prompt - which is
// what every position that doesn't pass -infer looks like.
func TestNoInferenceAtAll(t *testing.T) {
	is := is.New(t)
	f := fakeFacts()
	is.True(f.Inference == nil)
	is.True(!f.Flags["has_inference"])

	is.True(buildInference(f, nil) == nil)
	is.True(buildInference(f, &InferenceInput{}) == nil)

	p, err := BuildPrompt(f, false)
	is.NoErr(err)
	is.True(!slices.Contains(p.Concepts, "inference"))
}
