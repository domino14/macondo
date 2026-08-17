package explainer

import (
	"slices"
	"strings"
	"testing"

	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/rangefinder"
	"github.com/matryer/is"
)

// tile builds one row of the per-tile read from the pair that matters: the
// chance they are holding one, against the chance a random rack would be. The
// slot shares are derived back out of it so the row stays self-consistent, and
// so is the deviation in tiles - which for a tile you are unlikely to hold two
// of is the same quantity as the gap in probability, divided by a hundred.
func tile(name string, holds, chance float64, unseen, rackLen int) rangefinder.TileDeviation {
	return rangefinder.TileDeviation{
		Tile: name, HoldsPct: holds, ChanceHoldsPct: chance, Unseen: unseen,
		FoundPct: holds / float64(rackLen), ExpectedPct: chance / float64(rackLen),
		Tiles: (holds - chance) / 100.0,
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

// The gate is on how far the read moves the chance they hold a tile. A tile at
// three times its expected share is startling as a ratio and can still be
// nothing - Q here goes from 1.5% to 4.5% - while a tile the read moves by
// twenty points is worth a sentence whatever the ratio.
//
// Measuring in expected tiles instead, which this replaced, made the bar depend
// on how many tiles the opponent held: a 3-tile rack has only three slots for
// any deviation to live in, so a read taking a tile from 9% to 34% came out as
// a quarter of a tile and was thrown away.
func TestOutliersAreMeasuredInProbability(t *testing.T) {
	is := is.New(t)

	f := inferredFacts([]rangefinder.TileDeviation{
		tile("S", 35.0, 12.0, 3, 7), // +23 points - a real read
		tile("Q", 4.5, 1.5, 1, 7),   // +3 points - three times expected, and noise
		tile("V", 3.0, 7.0, 2, 7),   // -4 points - likelier absent, but barely
		tile("E", 20.0, 50.0, 6, 7), // -30 points - a real absence
		tile("A", 30.0, 31.0, 5, 7), // about as expected
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
		tile("S", 24.0, 23.5, 3, 7),
		tile("E", 50.2, 50.0, 6, 7),
	}, "5D (S)CAP(A)", 30.0, 0.9)
	is.True(!flat.Inference.Informative)
	is.True(!flat.Inference.Matters)
	is.True(!flat.Flags["has_inference"])

	// A strong read that moved neither the recommendation nor the win% past
	// the confidence intervals. Interesting, but not a lesson.
	inert := inferredFacts([]rangefinder.TileDeviation{
		tile("S", 35.0, 12.0, 3, 7),
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
	strongRead := []rangefinder.TileDeviation{tile("R", 40.0, 12.0, 2, 7)}

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
	z := tile("Z", 78.6, 3.6, 1, 3)
	e := tile("E", 21.0, 62.0, 6, 7)

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
	is.True(strings.Contains(p.User, "-0.41 tiles fewer than chance"))
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
		tile("S", 35.0, 12.0, 3, 7),
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
		tile("S", 35.0, 12.0, 3, 7),
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

// A read is not always about a tile. These are the real figures from a
// position whose entire finding was that the opponent kept consonants: the
// largest single move is N at +8.9 points - a tenth of a tile - and not one
// letter comes near the per-tile bar. The read is the sum of twenty-six small
// shifts, and a fact pack that only knew how to talk about tiles said nothing
// at all about the most useful thing on the board.
func TestAShapeReadWithNoStandoutTile(t *testing.T) {
	is := is.New(t)

	f := fakeFacts()
	f.Inference = buildInference(f, &InferenceInput{
		Summary: &rangefinder.InferenceSummary{
			NumRacks: 3404, RackLength: 3, ESS: 812.4, TopWeightPct: 2.2,
			Tiles: []rangefinder.TileDeviation{
				tile("N", 25.00, 16.10, 5, 3),
				tile("B", 12.25, 6.66, 2, 3),
				tile("A", 23.66, 24.87, 8, 3),
				tile("U", 3.54, 13.03, 4, 3),
			},
			Shape: &rangefinder.RackShape{
				RackLength: 3,
				Vowels:     rangefinder.CountPair{Read: 0.9026, Chance: 1.2809},
				Consonants: rangefinder.CountPair{Read: 2.0084, Chance: 1.6517},
				Blanks:     rangefinder.CountPair{Read: 0.0890, Chance: 0.0674},
				VowelCount: []rangefinder.CountPair{
					{Read: 31.2, Chance: 18.3}, {Read: 45.1, Chance: 42.7},
					{Read: 20.4, Chance: 31.6}, {Read: 3.3, Chance: 7.4},
				},
			},
		},
		Baseline: []montecarlo.CandidateStats{
			{Play: f.Best.Play, WinPct: 30.0, WinPctCI: 0.9},
		},
	})
	f.Flags = computeFlags(f)

	// Not one tile is worth naming...
	is.Equal(len(f.Inference.Outliers), 0)
	// ...and the read is still the best thing anyone could be told.
	is.True(f.Inference.ShapeRead != nil)
	is.True(f.Inference.Informative)
	is.True(f.Inference.Matters)
	is.True(f.Flags["has_inference"])

	p, err := BuildPrompt(f, false)
	is.NoErr(err)
	is.True(slices.Contains(p.Concepts, "inference"))
	is.True(strings.Contains(p.User,
		"2.01 consonants and 0.90 vowels, where a random rack from this pool holds 1.65 and 1.28"))
	is.True(strings.Contains(p.User, "consonant-heavy by about a third of a tile"))

	// The distribution too, because a mean of 0.9 vowels is equally consistent
	// with "usually one" and with "half the time two, half the time none".
	is.True(strings.Contains(p.User, "vowels held"))
	is.True(strings.Contains(p.User, "31.2%"))

	// With no tile clearing the bar, the per-tile section is absent entirely
	// rather than printed empty.
	is.True(!strings.Contains(p.User, "What they are holding, against"))

	// A read that left the shape alone doesn't raise it.
	flat := fakeFacts()
	flat.Inference = buildInference(flat, &InferenceInput{
		Summary: &rangefinder.InferenceSummary{
			NumRacks: 3404, RackLength: 3,
			Tiles: []rangefinder.TileDeviation{tile("N", 16.5, 16.10, 5, 3)},
			Shape: &rangefinder.RackShape{
				RackLength: 3,
				Vowels:     rangefinder.CountPair{Read: 1.30, Chance: 1.2809},
				Consonants: rangefinder.CountPair{Read: 1.63, Chance: 1.6517},
			},
		},
	})
	is.True(flat.Inference.ShapeRead == nil)
	is.True(!flat.Inference.Informative)
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
