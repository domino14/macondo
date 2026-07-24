package rangefinder

import (
	"math"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
)

func mls(ts ...int) []tilemapping.MachineLetter {
	out := make([]tilemapping.MachineLetter, len(ts))
	for i, t := range ts {
		out[i] = tilemapping.MachineLetter(t)
	}
	return out
}

func TestMarginalOrder(t *testing.T) {
	is := is.New(t)
	is.Equal(marginalOrder(1), 1)
	is.Equal(marginalOrder(2), 1)
	is.Equal(marginalOrder(3), 2)
	is.Equal(marginalOrder(4), 2)
	is.Equal(marginalOrder(5), 3)
	is.Equal(marginalOrder(6), 3)
	is.Equal(marginalOrder(7), 3)
}

// Each distinct sub-multiset of a recorded leave is counted exactly once,
// including with repeated tiles.
func TestAccumulatorDistinctSubMultisets(t *testing.T) {
	is := is.New(t)
	acc := newSubleaveAccumulator(4, 3)
	acc.record(mls(1, 1, 2), 0.5)

	is.Equal(acc.n, 1)
	is.Equal(acc.wTotal, 0.5)

	is.Equal(acc.cnt1[1], 1.0)
	is.Equal(acc.cnt1[2], 1.0)
	is.Equal(acc.cnt1[3], 0.0)

	is.Equal(acc.cnt2[acc.idx2(1, 1)], 1.0) // {1,1}
	is.Equal(acc.cnt2[acc.idx2(1, 2)], 1.0) // {1,2}
	is.Equal(acc.cnt2[acc.idx2(2, 2)], 0.0)

	is.Equal(acc.cnt3[acc.idx3(1, 1, 2)], 1.0) // {1,1,2}
	is.Equal(acc.cnt3[acc.idx3(1, 1, 1)], 0.0)

	is.Equal(acc.wsum1[1], 0.5)
	is.Equal(acc.wsum2[acc.idx2(1, 1)], 0.5)
	is.Equal(acc.wsum3[acc.idx3(1, 1, 2)], 0.5)
}

// With m = k, no shrinkage, and no clamping, the Möbius expansion must
// telescope: logImputed(L) equals the raw log-lift of L itself. This
// exercises the divisor-lattice Möbius handling for repeated tiles
// ({1,1,2}, {1,1,1}) as well as the all-distinct case.
func TestMobiusTelescoping(t *testing.T) {
	acc := newSubleaveAccumulator(5, 3)

	leaves := [][]tilemapping.MachineLetter{
		mls(1, 2, 3),
		mls(1, 1, 2),
		mls(1, 1, 1),
		mls(2, 3, 3),
		mls(1, 3, 4),
		mls(2, 2, 4),
	}
	weights := []float64{8, 2, 5, 1, 3, 0.25}
	for i, l := range leaves {
		acc.record(l, weights[i])
	}

	mod := buildImputationModel(acc, 0 /* no shrinkage */, 1e9, 1e9)

	wMean := acc.wTotal / float64(acc.n)
	var runBuf []tileRun
	for i, l := range leaves {
		runBuf = runsOf(l, runBuf)
		got := mod.logImputed(runBuf)
		// Each recorded 3-multiset is contained only in itself among
		// size-3 leaves, so its containment lift is w_i / wMean.
		want := math.Log(weights[i] / wMean)
		if math.Abs(got-want) > 1e-9 {
			t.Fatalf("leaf %v: logImputed=%v want=%v", l, got, want)
		}
	}
}

// subleaveTerms mirrors the sub-multiset walk of logImputed: its φ terms must
// sum to logImputed exactly, for leaves with and without repeated tiles.
func TestSubleaveTermsSumMatchesLogImputed(t *testing.T) {
	acc := newSubleaveAccumulator(5, 3)
	leaves := [][]tilemapping.MachineLetter{
		mls(1, 2, 3),
		mls(1, 1, 2),
		mls(1, 1, 1),
		mls(2, 3, 3),
		mls(1, 3, 4),
		mls(2, 2, 4),
	}
	weights := []float64{8, 2, 5, 1, 3, 0.25}
	for i, l := range leaves {
		acc.record(l, weights[i])
	}
	mod := buildImputationModel(acc, imputationLambda, maxAbsLogLift, maxAbsInteraction)

	queries := append(leaves, mls(2, 3, 4), mls(4, 4, 4), mls(1, 2, 2))
	var runBuf []tileRun
	for _, l := range queries {
		runBuf = runsOf(l, runBuf)
		terms := mod.subleaveTerms(runBuf)
		sum := 0.0
		for _, tm := range terms {
			sum += tm.phi
		}
		want := mod.logImputed(runBuf)
		if math.Abs(sum-want) > 1e-12 {
			t.Fatalf("leaf %v: Σφ over subleaveTerms=%v, logImputed=%v", l, sum, want)
		}
	}
}

// The pieces persisted on imputationResult (model, logCalib, maxLogW) must
// reconstruct every posterior weight exactly — this is the formula the
// AnalyzeLeave walkthrough displays.
func TestImputationResultReconstructsWeights(t *testing.T) {
	is := is.New(t)
	bagMap := []uint8{0, 5, 5, 5, 5}
	k := 3
	acc := newSubleaveAccumulator(len(bagMap), marginalOrder(k))
	measured := map[string]*measuredLeave{}

	truth := func(tiles []tilemapping.MachineLetter) float64 {
		w := 1.0
		for _, t := range tiles {
			if t == 1 {
				w *= 2
			}
		}
		return w
	}
	holdout := map[string]bool{
		leaveKey(mls(1, 2, 3)): true,
		leaveKey(mls(2, 3, 4)): true,
	}
	for _, l := range enumerateLeaves(bagMap, k) {
		key := leaveKey(l.tiles)
		if holdout[key] {
			continue
		}
		acc.record(l.tiles, truth(l.tiles))
		if measured[key] == nil {
			measured[key] = &measuredLeave{}
		}
		measured[key].sumW += truth(l.tiles)
		measured[key].count++
	}

	res := imputeFullPosterior(bagMap, k, acc, measured, 2)
	is.True(res.model != nil)
	is.True(len(res.racks) > 0)

	var runBuf []tileRun
	for _, r := range res.racks {
		prior := combinatorialPrior(r.Leave, bagMap)
		var logLik float64
		if ml := measured[leaveKey(r.Leave)]; ml != nil && ml.count > 0 {
			logLik = math.Log(ml.mean())
		} else {
			runBuf = runsOf(r.Leave, runBuf)
			logLik = res.logCalib + res.model.logImputed(runBuf)
		}
		got := math.Exp(math.Log(prior) + logLik - res.maxLogW)
		if math.Abs(got-r.Weight) > 1e-9*r.Weight {
			t.Fatalf("leaf %v: reconstructed %v, stored %v", r.Leave, got, r.Weight)
		}
	}
}

// The calibration constant must moment-match the arithmetic mean of measured
// likelihoods: Σ c·exp(logCalib + Σφ) = Σ c·w. With heavily skewed
// likelihoods a mean-of-logs anchor would land orders of magnitude below the
// arithmetic scale, starving imputed leaves of posterior mass.
func TestCalibrationMomentMatched(t *testing.T) {
	is := is.New(t)
	bagMap := []uint8{0, 5, 5, 5, 5}
	k := 3
	acc := newSubleaveAccumulator(len(bagMap), marginalOrder(k))
	measured := map[string]*measuredLeave{}

	// Heavily right-skewed: likelihoods span ~5 orders of magnitude.
	truth := func(tiles []tilemapping.MachineLetter) float64 {
		w := 1.0
		for _, tl := range tiles {
			w *= math.Pow(10, -float64(tl)+2)
		}
		return w
	}
	holdout := map[string]bool{
		leaveKey(mls(1, 2, 3)): true,
		leaveKey(mls(2, 3, 4)): true,
	}
	for _, l := range enumerateLeaves(bagMap, k) {
		key := leaveKey(l.tiles)
		if holdout[key] {
			continue
		}
		acc.record(l.tiles, truth(l.tiles))
		if measured[key] == nil {
			measured[key] = &measuredLeave{}
		}
		measured[key].sumW += truth(l.tiles)
		measured[key].count++
	}

	res := imputeFullPosterior(bagMap, k, acc, measured, 2)
	is.True(res.model != nil)

	var predicted, actual float64
	var runBuf []tileRun
	tiles := make([]tilemapping.MachineLetter, 0, k)
	for key, ml := range measured {
		if ml.mean() <= 0 {
			continue
		}
		tiles = tiles[:0]
		for i := 0; i < len(key); i++ {
			tiles = append(tiles, tilemapping.MachineLetter(key[i]))
		}
		runBuf = runsOf(tiles, runBuf)
		c := float64(ml.count)
		predicted += c * math.Exp(res.logCalib+res.model.logImputed(runBuf))
		actual += c * ml.mean()
	}
	if math.Abs(predicted-actual) > 1e-9*actual {
		t.Fatalf("moment mismatch: predicted Σc·ℓ̂=%v, actual Σc·w=%v", predicted, actual)
	}
}

// A fully-measured leave space reproduces exact Bayesian weights:
// posterior ∝ prior × measured likelihood.
func TestImputeFullPosteriorMeasuredExact(t *testing.T) {
	is := is.New(t)
	bagMap := []uint8{0, 3, 2, 1} // tiles 1,2,3 with counts 3,2,1
	k := 2
	acc := newSubleaveAccumulator(len(bagMap), marginalOrder(k))
	measured := map[string]*measuredLeave{}

	leaves := enumerateLeaves(bagMap, k)
	is.True(len(leaves) > 0)
	// Assign each leaf a distinct positive likelihood.
	likelihood := func(tiles []tilemapping.MachineLetter) float64 {
		s := 0.13
		for _, t := range tiles {
			s += float64(t) * 0.71
		}
		return s
	}
	for _, l := range leaves {
		w := likelihood(l.tiles)
		acc.record(l.tiles, w)
		key := leaveKey(l.tiles)
		if measured[key] == nil {
			measured[key] = &measuredLeave{}
		}
		measured[key].sumW += w
		measured[key].count++
	}

	res := imputeFullPosterior(bagMap, k, acc, measured, 2)
	is.Equal(res.measuredLeaves, len(leaves))
	is.Equal(res.imputedLeaves, 0)
	is.True(res.measuredMass > 0.999)

	// Cross-check weight ratios against prior × likelihood.
	want := map[string]float64{}
	for _, l := range leaves {
		want[leaveKey(l.tiles)] = l.prior * likelihood(l.tiles)
	}
	// Normalize both sides by their max and compare.
	maxGot, maxWant := 0.0, 0.0
	for _, r := range res.racks {
		if r.Weight > maxGot {
			maxGot = r.Weight
		}
	}
	for _, w := range want {
		if w > maxWant {
			maxWant = w
		}
	}
	for _, r := range res.racks {
		w, ok := want[leaveKey(r.Leave)]
		is.True(ok)
		if math.Abs(r.Weight/maxGot-w/maxWant) > 1e-9 {
			t.Fatalf("leaf %v: got %v want %v", r.Leave, r.Weight/maxGot, w/maxWant)
		}
	}
}

// Unmeasured leaves get likelihoods imputed from marginal lifts: with a
// multiplicative ground truth (containing tile 1 doubles the likelihood),
// the imputed ratio between two unmeasured leaves that differ only by
// tile 1 vs tile 4 should be close to 2.
func TestImputeUnmeasuredLift(t *testing.T) {
	is := is.New(t)
	bagMap := []uint8{0, 5, 5, 5, 5} // tiles 1..4, five copies each
	k := 3
	acc := newSubleaveAccumulator(len(bagMap), marginalOrder(k))
	measured := map[string]*measuredLeave{}

	truth := func(tiles []tilemapping.MachineLetter) float64 {
		w := 1.0
		for _, t := range tiles {
			if t == 1 {
				w *= 2
			}
		}
		return w
	}

	holdout := map[string]bool{
		leaveKey(mls(1, 2, 3)): true,
		leaveKey(mls(2, 3, 4)): true,
	}

	leaves := enumerateLeaves(bagMap, k)
	for _, l := range leaves {
		key := leaveKey(l.tiles)
		if holdout[key] {
			continue
		}
		// Record repeatedly so shrinkage (λ=10) has little effect.
		for rep := 0; rep < 30; rep++ {
			acc.record(l.tiles, truth(l.tiles))
		}
		if measured[key] == nil {
			measured[key] = &measuredLeave{}
		}
		measured[key].sumW += truth(l.tiles) * 30
		measured[key].count += 30
	}

	res := imputeFullPosterior(bagMap, k, acc, measured, 2)
	is.Equal(res.imputedLeaves, 2)

	var w123, w234 float64
	for _, r := range res.racks {
		switch leaveKey(r.Leave) {
		case leaveKey(mls(1, 2, 3)):
			w123 = r.Weight
		case leaveKey(mls(2, 3, 4)):
			w234 = r.Weight
		}
	}
	is.True(w123 > 0)
	is.True(w234 > 0)
	// Equal priors (symmetric counts), so the weight ratio is the imputed
	// likelihood ratio; ground truth is 2.
	ratio := w123 / w234
	if ratio < 1.4 || ratio > 2.6 {
		t.Fatalf("imputed lift ratio %v, want ≈2", ratio)
	}
}

// With no recorded signal at all, the model must degrade to the prior.
func TestImputeNoSignalIsPrior(t *testing.T) {
	is := is.New(t)
	bagMap := []uint8{0, 4, 3, 2}
	k := 2
	acc := newSubleaveAccumulator(len(bagMap), marginalOrder(k))
	res := imputeFullPosterior(bagMap, k, acc, map[string]*measuredLeave{}, 1)

	leaves := enumerateLeaves(bagMap, k)
	is.Equal(len(res.racks), len(leaves))

	prior := map[string]float64{}
	maxPrior := 0.0
	for _, l := range leaves {
		prior[leaveKey(l.tiles)] = l.prior
		if l.prior > maxPrior {
			maxPrior = l.prior
		}
	}
	for _, r := range res.racks {
		want := prior[leaveKey(r.Leave)] / maxPrior
		if math.Abs(r.Weight-want) > 1e-9 {
			t.Fatalf("leaf %v: got %v want %v (prior)", r.Leave, r.Weight, want)
		}
	}
}

// Zero-likelihood measurements are evidence of impossibility, not missing
// data: they must be excluded from the posterior rather than imputed.
func TestMeasuredZeroExcluded(t *testing.T) {
	is := is.New(t)
	bagMap := []uint8{0, 2, 2}
	k := 1
	acc := newSubleaveAccumulator(len(bagMap), marginalOrder(k))
	measured := map[string]*measuredLeave{}

	acc.record(mls(1), 1.0)
	measured[leaveKey(mls(1))] = &measuredLeave{sumW: 1.0, count: 1}
	acc.record(mls(2), 0.0)
	measured[leaveKey(mls(2))] = &measuredLeave{sumW: 0.0, count: 1}

	res := imputeFullPosterior(bagMap, k, acc, measured, 1)
	is.Equal(len(res.racks), 1)
	is.Equal(res.racks[0].Leave[0], tilemapping.MachineLetter(1))
}

// Worst realistic case: 6-tile leaves from a near-full English bag.
func BenchmarkImputeFullPosteriorK6(b *testing.B) {
	// Standard English distribution: blank(0)=2, A=9, B=2, C=2, D=4, E=12,
	// F=2, G=3, H=2, I=9, J=1, K=1, L=4, M=2, N=6, O=8, P=2, Q=1, R=6, S=4,
	// T=6, U=4, V=2, W=2, X=1, Y=2, Z=1; minus a played tile and our rack of
	// 7 is immaterial for the benchmark.
	bagMap := []uint8{2, 9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1}
	k := 6
	acc := newSubleaveAccumulator(len(bagMap), marginalOrder(k))
	measured := map[string]*measuredLeave{}

	// Simulate ~2000 evaluated samples with a deterministic pseudo-pattern.
	tiles := make([]tilemapping.MachineLetter, k)
	seed := uint64(12345)
	next := func(n int) int {
		seed = seed*6364136223846793005 + 1442695040888963407
		return int((seed >> 33) % uint64(n))
	}
	for i := 0; i < 2000; i++ {
		for j := range tiles {
			tiles[j] = tilemapping.MachineLetter(next(len(bagMap)))
		}
		w := 1.0 / float64(1+next(50))
		l := make([]tilemapping.MachineLetter, k)
		copy(l, tiles)
		// record sorts in place.
		acc0 := l
		sortMLs(acc0)
		acc.record(acc0, w)
		key := leaveKey(acc0)
		if measured[key] == nil {
			measured[key] = &measuredLeave{}
		}
		measured[key].sumW += w
		measured[key].count++
	}

	b.ResetTimer()
	var leafCount int
	for i := 0; i < b.N; i++ {
		res := imputeFullPosterior(bagMap, k, acc, measured, 8)
		leafCount = len(res.racks)
	}
	b.ReportMetric(float64(leafCount), "leaves")
}

func sortMLs(l []tilemapping.MachineLetter) {
	for i := 1; i < len(l); i++ {
		for j := i; j > 0 && l[j-1] > l[j]; j-- {
			l[j-1], l[j] = l[j], l[j-1]
		}
	}
}
