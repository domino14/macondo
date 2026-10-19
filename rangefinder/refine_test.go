package rangefinder

import (
	"math"
	"math/rand"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/montecarlo"
)

// Systematic PPS must give each item exactly ⌊m·q⌋ or ⌈m·q⌉ draws, whatever
// the random offset — that expectation is what keeps u = P/q valid.
func TestSystematicSampleExpectation(t *testing.T) {
	is := is.New(t)
	q := []float64{0.5, 0.25, 0.15, 0.09, 0.01}
	rng := rand.New(rand.NewSource(7))

	for _, m := range []int{4, 8, 20, 100} {
		totals := make([]int, len(q))
		trials := 200
		for tr := 0; tr < trials; tr++ {
			counts := make([]int, len(q))
			picks := systematicSample(q, m, rng)
			is.Equal(len(picks), m) // always exactly m draws
			for _, p := range picks {
				counts[p]++
			}
			for i := range q {
				want := float64(m) * q[i]
				if float64(counts[i]) < math.Floor(want) || float64(counts[i]) > math.Ceil(want) {
					t.Fatalf("m=%d item %d: drawn %d times, want ⌊%v⌋..⌈%v⌉",
						m, i, counts[i], want, want)
				}
				totals[i] += counts[i]
			}
		}
		// And the mean over trials lands on m·q.
		for i := range q {
			got := float64(totals[i]) / float64(trials)
			want := float64(m) * q[i]
			if math.Abs(got-want) > 0.5 {
				t.Fatalf("m=%d item %d: mean draws %v, want %v", m, i, got, want)
			}
		}
	}
}

// Weight truncation bounds what a single draw from the model's tail can do to
// the lifts.
func TestProposalWeightsTruncated(t *testing.T) {
	is := is.New(t)
	cands := make([]refineCandidate, 100)
	for i := range cands {
		cands[i].u = 1
	}
	cands[0].u = 10000 // a leave the model thought was near-impossible

	capped := truncateWeights(cands)
	is.Equal(capped, 1)

	// limit = sqrt(n)·mean, with mean computed before capping.
	mean := (10000 + 99) / 100.0
	limit := math.Sqrt(100) * mean
	is.True(math.Abs(cands[0].u-limit) < 1e-9)
	is.Equal(cands[1].u, 1.0) // untouched
	is.True(cands[0].u < 10000)
}

// The round's ratio statistic is the out-of-sample calibration check: it must
// read 1 when the model predicted the batch exactly, and recover a known
// scale error otherwise, with a standard error that vanishes on a perfect fit.
func TestRatioStatistic(t *testing.T) {
	is := is.New(t)
	us := []float64{1, 2, 0.5, 3}
	lhats := []float64{0.1, 0.4, 0.02, 0.9}

	perfect := append([]float64{}, lhats...)
	ratio, se := ratioStatistic(us, perfect, lhats)
	is.True(math.Abs(ratio-1) < 1e-12)
	is.True(se < 1e-12)

	doubled := make([]float64, len(lhats))
	for i, l := range lhats {
		doubled[i] = 2 * l
	}
	ratio, se = ratioStatistic(us, doubled, lhats)
	is.True(math.Abs(ratio-2) < 1e-12)
	is.True(se < 1e-12) // a uniform scale error is estimated exactly

	// Scatter around the prediction leaves the ratio near 1 but the standard
	// error positive, which is what stops the loop from calling it converged.
	noisy := []float64{0.05, 0.8, 0.02, 0.45}
	ratio, se = ratioStatistic(us, noisy, lhats)
	is.True(ratio > 0)
	is.True(se > 0)
}

// The ranking view carries each leave's origin round and keeps ranks assigned
// over the full posterior, so a filtered view still says where its rows sit.
func TestRankedRacksTagsOriginRound(t *testing.T) {
	is := is.New(t)
	r := &RangeFinder{
		inference: &Inference{
			InferredRacks: []montecarlo.InferredRack{
				{Leave: mls(1, 2), Weight: 0.2},
				{Leave: mls(1, 3), Weight: 0.9},
				{Leave: mls(2, 3), Weight: 0.5},
				{Leave: mls(3, 3), Weight: 0.1},
			},
		},
		measured: map[string]*measuredLeave{
			leaveKey(mls(1, 3)): {count: 2, round: 0},
			leaveKey(mls(2, 3)): {count: 1, round: 3},
		},
	}
	rows := r.rankedRacks()
	is.Equal(len(rows), 4)

	// Sorted by weight, ranks 1..n over the whole posterior.
	is.Equal(rows[0].rank, 1)
	is.True(rows[0].measured)
	is.Equal(rows[0].round, 0)
	is.Equal(rows[0].source(), "measured ×2 r0")

	is.Equal(rows[1].rank, 2)
	is.Equal(rows[1].round, 3)
	is.Equal(rows[1].source(), "measured ×1 r3")

	// The imputed ones keep their overall rank, which is the point of the
	// filtered view: the first imputed leave here sits at rank 3, not 1.
	is.True(!rows[2].measured)
	is.Equal(rows[2].rank, 3)
	is.Equal(rows[2].source(), "imputed")

	// Weight shares are over the full posterior.
	total := 0.2 + 0.9 + 0.5 + 0.1
	is.True(math.Abs(rows[0].pct-100*0.9/total) < 1e-9)

	first, count, pct := imputedSummary(rows)
	is.True(first != nil)
	is.Equal(first.rank, 3) // not 1: the top of the posterior is measured
	is.Equal(count, 2)
	is.True(math.Abs(pct-100*(0.2+0.1)/total) < 1e-9) // the two unmeasured ones
}

// With everything measured there is no imputed leave to point at.
func TestImputedSummaryFullyMeasured(t *testing.T) {
	is := is.New(t)
	r := &RangeFinder{
		inference: &Inference{
			InferredRacks: []montecarlo.InferredRack{{Leave: mls(1, 2), Weight: 1}},
		},
		measured: map[string]*measuredLeave{leaveKey(mls(1, 2)): {count: 1}},
	}
	first, count, pct := imputedSummary(r.rankedRacks())
	is.True(first == nil)
	is.Equal(count, 0)
	is.Equal(pct, 0.0)
}

// A degenerate batch must not produce a NaN convergence test.
func TestRatioStatisticDegenerate(t *testing.T) {
	is := is.New(t)
	ratio, se := ratioStatistic(nil, nil, nil)
	is.Equal(ratio, 0.0)
	is.Equal(se, 0.0)
	// All predictions zero: no denominator, no statistic.
	ratio, se = ratioStatistic([]float64{1}, []float64{0.5}, []float64{0})
	is.Equal(ratio, 0.0)
	is.Equal(se, 0.0)
}
