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
	acc.record(mls(1, 1, 2), 0.5, 1)

	is.Equal(acc.n, 1)
	is.Equal(acc.likTotal, 0.5)

	is.Equal(acc.wt1[1], 1.0)
	is.Equal(acc.wt1[2], 1.0)
	is.Equal(acc.wt1[3], 0.0)

	is.Equal(acc.wt2[acc.idx2(1, 1)], 1.0) // {1,1}
	is.Equal(acc.wt2[acc.idx2(1, 2)], 1.0) // {1,2}
	is.Equal(acc.wt2[acc.idx2(2, 2)], 0.0)

	is.Equal(acc.wt3[acc.idx3(1, 1, 2)], 1.0) // {1,1,2}
	is.Equal(acc.wt3[acc.idx3(1, 1, 1)], 0.0)

	is.Equal(acc.lik1[1], 0.5)
	is.Equal(acc.lik2[acc.idx2(1, 1)], 0.5)
	is.Equal(acc.lik3[acc.idx3(1, 1, 2)], 0.5)
}

// The importance-weighted accumulator must reduce exactly to plain counting
// when every draw carries u = 1, so prior-sampled inference is bit-for-bit
// what it was before weights existed.
func TestUnitWeightsMatchCounts(t *testing.T) {
	is := is.New(t)
	acc := newSubleaveAccumulator(6, 3)
	leaves := [][]tilemapping.MachineLetter{
		mls(1, 2, 3), mls(1, 1, 4), mls(2, 3, 5), mls(1, 2, 3), mls(4, 4, 4),
	}
	for i, l := range leaves {
		acc.record(l, 0.1*float64(i+1), 1)
	}

	// wt is the containment count, wtsq equals it, so ESS is the count.
	for _, sub := range [][]tilemapping.MachineLetter{
		mls(1), mls(2), mls(4), mls(1, 2), mls(4, 4), mls(1, 2, 3), mls(4, 4, 4),
	} {
		e, wt, _ := acc.accStats(sub)
		is.Equal(e, wt)
		is.Equal(wt, math.Round(wt)) // integral: it is a count
	}
	e, wt, lik := acc.accStats(mls(1, 2))
	is.Equal(wt, 2.0) // leaves 0 and 3
	is.Equal(e, 2.0)
	is.True(math.Abs(lik-(0.1+0.4)) < 1e-12)
	is.Equal(acc.wtTotal, 5.0)
	is.True(math.Abs(acc.likTotal-(0.1+0.2+0.3+0.4+0.5)) < 1e-12)
}

// The lift estimator must recover the same population quantity under a
// non-prior proposal — the whole point of importance weighting. A proposal
// that draws some leaves more often than the prior would, corrected by
// u = P/q, must produce exactly the prior-weighted lifts.
//
// The design is made deterministic by realizing the proposal as integer
// multiplicities: leaves containing tile 1 are drawn 3× as often as the rest,
// so q ∝ multiplicity and u = P/q. Since the accumulator is linear in u, this
// is the exact-expectation version of that sampler, with no Monte Carlo noise
// to tolerate.
func TestWeightedLiftUnbiasedUnderSkewedProposal(t *testing.T) {
	is := is.New(t)
	bagMap := []uint8{0, 4, 4, 4, 4}
	k := 2
	order := marginalOrder(k)
	truth := func(tiles []tilemapping.MachineLetter) float64 {
		w := 1.0
		for _, tl := range tiles {
			w *= 1.0 + 0.5*float64(tl)
		}
		return w
	}
	leaves := enumerateLeaves(bagMap, k)

	// Reference: each leave once, weighted by its exact prior — the
	// population lift with no sampling noise at all.
	ref := newSubleaveAccumulator(len(bagMap), order)
	for _, l := range leaves {
		ref.record(l.tiles, truth(l.tiles), l.prior)
	}

	multOf := func(l enumLeaf) int {
		for _, tl := range l.tiles {
			if tl == 1 {
				return 3
			}
		}
		return 1
	}
	totalMult := 0
	for _, l := range leaves {
		totalMult += multOf(l)
	}
	skewed := newSubleaveAccumulator(len(bagMap), order)
	nDraws := 0
	for _, l := range leaves {
		m := multOf(l)
		q := float64(m) / float64(totalMult) // proposal probability
		u := l.prior / q
		for i := 0; i < m; i++ { // drawn m times by this proposal
			skewed.record(l.tiles, truth(l.tiles), u)
			nDraws++
		}
	}
	is.True(nDraws > len(leaves)) // the proposal really is skewed

	// Lifts must agree exactly. ESS differs — that is the price of the skew —
	// so shrinkage is off for the comparison.
	refMod := buildImputationModel(ref, 0, 1e9, 1e9)
	skewMod := buildImputationModel(skewed, 0, 1e9, 1e9)
	for _, l := range leaves {
		runs := runsOf(l.tiles, nil)
		a, b := refMod.logImputed(runs), skewMod.logImputed(runs)
		if math.Abs(a-b) > 1e-9 {
			t.Fatalf("leave %v: prior-weighted %v, importance-weighted %v", l.tiles, a, b)
		}
	}

	// And the skew must cost effective sample size relative to the unweighted
	// count, which is what shrinkage keys off.
	e, _, _ := skewed.accStats(mls(1))
	is.True(e < 3*float64(len(leaves)))
}

// A sub-multiset never observed is the most uncertain term in the model, not
// the least: it must carry σ₀ rather than the zero its φ gets.
func TestUncertaintyFavorsThinSupport(t *testing.T) {
	is := is.New(t)
	acc := newSubleaveAccumulator(8, 2)
	// Tiles 1-3 richly observed; tiles 6,7 never seen.
	for i := 0; i < 60; i++ {
		acc.record(mls(1, 2), 1.0, 1)
		acc.record(mls(1, 3), 3.0, 1)
		acc.record(mls(2, 3), 0.2, 1)
	}
	mod := buildImputationModel(acc, imputationLambda, maxAbsLogLift, maxAbsInteraction)

	wellSupported := mod.uncertainty(runsOf(mls(1, 2), nil))
	unseen := mod.uncertainty(runsOf(mls(6, 7), nil))
	is.True(unseen > wellSupported)
	// The unseen pair's φ is zero, so uncertainty is the only thing that can
	// make the refine loop look at it.
	is.Equal(mod.logImputed(runsOf(mls(6, 7), nil)), 0.0)
	is.True(unseen > 0)
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
		acc.record(l, weights[i], 1)
	}

	mod := buildImputationModel(acc, 0 /* no shrinkage */, 1e9, 1e9)

	wMean := acc.likTotal / acc.wtTotal
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
		acc.record(l, weights[i], 1)
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
		acc.record(l.tiles, truth(l.tiles), 1)
		if measured[key] == nil {
			measured[key] = &measuredLeave{}
		}
		measured[key].sumW += truth(l.tiles)
		measured[key].count++
		measured[key].sumU++
	}

	res := imputeFullPosterior(bagMap, k, acc, nil, measured, 2)
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
		acc.record(l.tiles, truth(l.tiles), 1)
		if measured[key] == nil {
			measured[key] = &measuredLeave{}
		}
		measured[key].sumW += truth(l.tiles)
		measured[key].count++
		measured[key].sumU++
	}

	res := imputeFullPosterior(bagMap, k, acc, nil, measured, 2)
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

// recordSample mirrors RangeFinder.recordPlacementSample: one sample feeds
// the full accumulator, its leave's cross-fitting fold, and the measured map.
func recordSample(acc *subleaveAccumulator, folds []*subleaveAccumulator,
	measured map[string]*measuredLeave, tiles []tilemapping.MachineLetter, w float64) {

	key := leaveKey(tiles)
	acc.record(tiles, w, 1)
	if len(folds) > 0 {
		folds[foldForKey(key, len(folds))].record(tiles, w, 1)
	}
	if measured[key] == nil {
		measured[key] = &measuredLeave{}
	}
	measured[key].sumW += w
	measured[key].count++
	measured[key].sumU++
}

func newFoldAccumulators(alphaSize, maxOrder, n int) []*subleaveAccumulator {
	folds := make([]*subleaveAccumulator, n)
	for i := range folds {
		folds[i] = newSubleaveAccumulator(alphaSize, maxOrder)
	}
	return folds
}

// Cross-fitting requires that the folds partition the samples exactly (so a
// complement is a subtraction) and that every measurement of the same leave
// lands in the same fold (so a complement model has never seen the leave it
// predicts).
func TestCrossFitFoldPartition(t *testing.T) {
	is := is.New(t)
	bagMap := []uint8{0, 4, 4, 4, 4}
	k := 3
	acc := newSubleaveAccumulator(len(bagMap), marginalOrder(k))
	folds := newFoldAccumulators(len(bagMap), marginalOrder(k), calibrationFolds)
	measured := map[string]*measuredLeave{}

	for _, l := range enumerateLeaves(bagMap, k) {
		// Measure each leave three times, so the same-fold invariant has
		// something to violate.
		for rep := 0; rep < 3; rep++ {
			recordSample(acc, folds, measured, l.tiles, 0.5+0.1*float64(rep))
		}
	}

	// Partition: fold counts and sums add back up to the full accumulator.
	sumN, sumW := 0, 0.0
	for _, f := range folds {
		sumN += f.n
		sumW += f.likTotal
	}
	is.Equal(sumN, acc.n)
	is.True(math.Abs(sumW-acc.likTotal) < 1e-9)

	for _, f := range folds {
		comp := acc.minus(f)
		is.Equal(comp.n, acc.n-f.n)
		for i := range comp.wt1 {
			is.True(math.Abs(comp.wt1[i]-(acc.wt1[i]-f.wt1[i])) < 1e-9)
			is.True(math.Abs(comp.wtsq1[i]-(acc.wtsq1[i]-f.wtsq1[i])) < 1e-9)
			is.True(math.Abs(comp.lik1[i]-(acc.lik1[i]-f.lik1[i])) < 1e-9)
		}
	}

	// Same leave, same fold: all measurements of one leave go to a single
	// fold, so its complement model has never seen it.
	solo := newFoldAccumulators(len(bagMap), marginalOrder(k), calibrationFolds)
	soloAcc := newSubleaveAccumulator(len(bagMap), marginalOrder(k))
	for rep := 0; rep < 4; rep++ {
		recordSample(soloAcc, solo, map[string]*measuredLeave{}, mls(1, 2, 3), 0.25)
	}
	nonEmpty := 0
	for _, f := range solo {
		if f.n > 0 {
			nonEmpty++
			is.Equal(f.n, 4)
		}
	}
	is.Equal(nonEmpty, 1)
}

// The calibration constant is fit on measured leaves but applied to the
// unmeasured ones, so the predictions in its denominator must be
// out-of-sample. Fitting it in-sample lets a leave's own measurements set the
// very lifts that predict it: the denominator is inflated and C comes out too
// low, starving every imputed leaf of posterior mass.
//
// Here tiles 7 and 8 occur only in one heavily-measured, high-likelihood
// leave. The full-data model explains that leave through lifts it supplied
// itself; the model fit on the folds excluding it has never seen those tiles
// and predicts the flat baseline. Cross-fitting must therefore land
// materially above the in-sample constant.
func TestCrossFitRemovesSelfFit(t *testing.T) {
	is := is.New(t)
	bagMap := []uint8{0, 5, 5, 5, 5, 5, 5, 5, 5}
	k := 2 // marginal order 1: φ over single tiles only, so this is by hand
	order := marginalOrder(k)

	accX := newSubleaveAccumulator(len(bagMap), order)
	folds := newFoldAccumulators(len(bagMap), order, calibrationFolds)
	measuredX := map[string]*measuredLeave{}
	accIn := newSubleaveAccumulator(len(bagMap), order)
	measuredIn := map[string]*measuredLeave{}

	record := func(tiles []tilemapping.MachineLetter, w float64, reps int) {
		for i := 0; i < reps; i++ {
			recordSample(accX, folds, measuredX, tiles, w)
			recordSample(accIn, nil, measuredIn, tiles, w)
		}
	}
	// Baseline: ordinary leaves over tiles 1-6, likelihood 1.
	for a := 1; a <= 6; a++ {
		for b := a + 1; b <= 6; b++ {
			record(mls(a, b), 1.0, 1)
		}
	}
	// The self-fit leave: tiles 7 and 8 appear nowhere else.
	record(mls(7, 8), 100.0, 40)

	resX := imputeFullPosterior(bagMap, k, accX, folds, measuredX, 2)
	resIn := imputeFullPosterior(bagMap, k, accIn, nil, measuredIn, 2)
	is.True(resX.crossFitted)
	is.True(!resIn.crossFitted)
	// Identical samples, so the full-data models agree: only C differs.
	is.True(math.Abs(resX.logCalibInSample-resIn.logCalib) < 1e-12)

	// The full model has learned a lift for tiles 7/8 from that leave alone.
	runs := runsOf(mls(7, 8), nil)
	is.True(resX.model.logImputed(runs) > 0.5)

	t.Logf("C: cross-fit %.4f, in-sample %.4f (%.3fx)", resX.logCalib,
		resX.logCalibInSample, math.Exp(resX.logCalib-resX.logCalibInSample))
	// Deterministic inputs: the observed gap is 0.407 (1.50x).
	if resX.logCalib <= resX.logCalibInSample+0.3 {
		t.Fatalf("cross-fitting failed to undo the self-fit: C %.4f vs in-sample %.4f",
			resX.logCalib, resX.logCalibInSample)
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
		acc.record(l.tiles, w, 1)
		key := leaveKey(l.tiles)
		if measured[key] == nil {
			measured[key] = &measuredLeave{}
		}
		measured[key].sumW += w
		measured[key].count++
		measured[key].sumU++
	}

	res := imputeFullPosterior(bagMap, k, acc, nil, measured, 2)
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
			acc.record(l.tiles, truth(l.tiles), 1)
		}
		if measured[key] == nil {
			measured[key] = &measuredLeave{}
		}
		measured[key].sumW += truth(l.tiles) * 30
		measured[key].count += 30
	}

	res := imputeFullPosterior(bagMap, k, acc, nil, measured, 2)
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
	res := imputeFullPosterior(bagMap, k, acc, nil, map[string]*measuredLeave{}, 1)

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

	acc.record(mls(1), 1.0, 1)
	measured[leaveKey(mls(1))] = &measuredLeave{sumW: 1.0, count: 1}
	acc.record(mls(2), 0.0, 1)
	measured[leaveKey(mls(2))] = &measuredLeave{sumW: 0.0, count: 1}

	res := imputeFullPosterior(bagMap, k, acc, nil, measured, 1)
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
		acc.record(acc0, w, 1)
		key := leaveKey(acc0)
		if measured[key] == nil {
			measured[key] = &measuredLeave{}
		}
		measured[key].sumW += w
		measured[key].count++
		measured[key].sumU++
	}

	b.ResetTimer()
	var leafCount int
	for i := 0; i < b.N; i++ {
		res := imputeFullPosterior(bagMap, k, acc, nil, measured, 8)
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
