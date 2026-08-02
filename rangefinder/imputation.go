package rangefinder

import (
	"math"

	"github.com/domino14/word-golib/tilemapping"
	"golang.org/x/sync/errgroup"

	"github.com/domino14/macondo/montecarlo"
)

// This file implements marginal-lift imputation: from the leaves the
// inference engine actually evaluated (each with a measured likelihood
// P(play | leave)), we estimate low-order containment marginals and use them
// to impute a likelihood for every feasible leave that was never evaluated.
// The result is one complete posterior over the whole leave space, so the
// simmer can sample opponent leaves directly from it with no ad-hoc
// fallback-to-random gate.
//
// For a sub-multiset S, the "lift" is
//
//	lift(S) = E[likelihood | S ⊆ leave] / E[likelihood]
//
// estimated over the evaluated leaves (numerator and denominator share the
// same sample set, so selection noise partially cancels). Lifts are combined
// into a likelihood estimate with a truncated log-linear expansion whose
// interaction terms come from Möbius inversion on the sub-multiset (divisor)
// lattice: bottom-up, φ(S) = log lift(S) − Σ_{T⊊S, T≠∅} φ(T), which for
// distinct tiles reduces to the familiar inclusion-exclusion forms but
// handles repeated tiles correctly (e.g. φ({A,A}) = log lift(AA) − log
// lift(A); the empty set gets Möbius coefficient 0 there, not +1).
//
// Then log ℓ̂(L) = Σ_{S ⊆ L, 1 ≤ |S| ≤ m} φ(S) over distinct sub-multisets.
// When m = |L| and no shrinkage/clamping is applied this telescopes to
// exactly log lift(L).

const (
	// imputationLambda is the shrinkage pseudo-count: a term estimated from c
	// samples is scaled by c/(c+λ), so thin support decays toward 0 (lift 1 /
	// no interaction) instead of adding variance.
	imputationLambda = 10.0

	// maxAbsLogLift clamps raw log-lifts for numerical safety (a sub-multiset
	// whose every containing sample had likelihood 0 would otherwise be -Inf).
	maxAbsLogLift = 13.8 // ~log(1e6)

	// maxAbsInteraction clamps pair/triple interaction terms; marginal (order
	// 1) terms are only clamped by maxAbsLogLift.
	maxAbsInteraction = 2.302585 // log(10)

	// negligibleWeightFraction drops leaves whose posterior weight, after
	// normalization by the max weight, falls below this fraction. The total
	// discarded mass is bounded by leafCount * this, i.e. ~1e-9 worst case.
	negligibleWeightFraction = 1e-15

	// calibrationFolds is the number of cross-fitting folds used to estimate
	// the calibration constant. Each distinct measured leave is assigned to
	// one fold; the constant is computed from predictions made by models fit
	// on the other folds. See calibrateLogConstant.
	calibrationFolds = 5
)

// foldForKey assigns a distinct leave (identified by its sorted-tile key) to
// one of nFolds cross-fitting folds, by FNV-1a hash. Every measurement of the
// same leave lands in the same fold, so a fold-complement model has never
// seen the leave whose prediction it supplies.
func foldForKey(key string, nFolds int) int {
	var h uint32 = 2166136261
	for i := 0; i < len(key); i++ {
		h ^= uint32(key[i])
		h *= 16777619
	}
	return int(h % uint32(nFolds))
}

// marginalOrder returns the maximum sub-multiset order m estimated for leaves
// of size k: ceil(k/2), capped at 3.
func marginalOrder(k int) int {
	m := (k + 1) / 2
	if m > 3 {
		m = 3
	}
	if m < 1 {
		m = 1
	}
	return m
}

// tileRun is a (tile, count) run of a sorted leave.
type tileRun struct {
	t tilemapping.MachineLetter
	c int
}

// runsOf collapses a sorted leave into distinct-tile runs. buf is reused.
func runsOf(sorted []tilemapping.MachineLetter, buf []tileRun) []tileRun {
	buf = buf[:0]
	for _, t := range sorted {
		if n := len(buf); n > 0 && buf[n-1].t == t {
			buf[n-1].c++
		} else {
			buf = append(buf, tileRun{t: t, c: 1})
		}
	}
	return buf
}

// subleaveAccumulator accumulates containment-marginal statistics for every
// distinct sub-multiset of order ≤ maxOrder of each recorded leave. Not
// goroutine-safe; callers synchronize.
//
// Each draw carries an importance weight u = P(L)/q(L), where q is whatever
// proposal produced it, so the statistics estimate prior-weighted population
// quantities under any sampling scheme (docs/development/
// iterative_inference_plan.md §1):
//
//	wt(S)   = Σ u          over draws containing S — the generalized count
//	wtsq(S) = Σ u²         for effective sample size
//	lik(S)  = Σ u·w
//
// With prior sampling every u = 1, so wt and wtsq both collapse to the raw
// containment count and ESS = wt²/wtsq = count: the unweighted estimator is
// the special case, not a different code path.
type subleaveAccumulator struct {
	alphaSize int
	maxOrder  int

	n         int     // raw number of recorded draws
	wtTotal   float64 // Σ u
	wtsqTotal float64 // Σ u²
	likTotal  float64 // Σ u·w

	wt1, wtsq1, lik1 []float64 // [A], index t
	wt2, wtsq2, lik2 []float64 // [A*A], index t*A+u with t ≤ u
	wt3, wtsq3, lik3 []float64 // [A*A*A], index (t*A+u)*A+v with t ≤ u ≤ v

	runBuf []tileRun
}

func newSubleaveAccumulator(alphaSize, maxOrder int) *subleaveAccumulator {
	acc := &subleaveAccumulator{alphaSize: alphaSize, maxOrder: maxOrder}
	acc.wt1 = make([]float64, alphaSize)
	acc.wtsq1 = make([]float64, alphaSize)
	acc.lik1 = make([]float64, alphaSize)
	if maxOrder >= 2 {
		acc.wt2 = make([]float64, alphaSize*alphaSize)
		acc.wtsq2 = make([]float64, alphaSize*alphaSize)
		acc.lik2 = make([]float64, alphaSize*alphaSize)
	}
	if maxOrder >= 3 {
		acc.wt3 = make([]float64, alphaSize*alphaSize*alphaSize)
		acc.wtsq3 = make([]float64, alphaSize*alphaSize*alphaSize)
		acc.lik3 = make([]float64, alphaSize*alphaSize*alphaSize)
	}
	return acc
}

func (acc *subleaveAccumulator) idx2(t, u tilemapping.MachineLetter) int {
	return int(t)*acc.alphaSize + int(u)
}

func (acc *subleaveAccumulator) idx3(t, u, v tilemapping.MachineLetter) int {
	return (int(t)*acc.alphaSize+int(u))*acc.alphaSize + int(v)
}

// ess returns the effective sample size of a (wt, wtsq) pair: wt²/wtsq, which
// is the plain observation count when every draw carried weight 1.
func ess(wt, wtsq float64) float64 {
	if wtsq <= 0 {
		return 0
	}
	return wt * wt / wtsq
}

// record adds one evaluated leave (sorted ascending) with its measured
// likelihood w (which may be 0) and its importance weight u to the
// accumulator. Prior-sampled draws pass u = 1.
func (acc *subleaveAccumulator) record(sorted []tilemapping.MachineLetter, w, u float64) {
	acc.n++
	acc.wtTotal += u
	acc.wtsqTotal += u * u
	acc.likTotal += u * w
	acc.runBuf = runsOf(sorted, acc.runBuf)
	runs := acc.runBuf
	uu, uw := u*u, u*w

	add := func(wt, wtsq, lik []float64, j int) {
		wt[j] += u
		wtsq[j] += uu
		lik[j] += uw
	}

	for _, r := range runs {
		add(acc.wt1, acc.wtsq1, acc.lik1, int(r.t))
	}
	if acc.maxOrder >= 2 {
		for i, r := range runs {
			if r.c >= 2 {
				add(acc.wt2, acc.wtsq2, acc.lik2, acc.idx2(r.t, r.t))
			}
			for _, r2 := range runs[i+1:] {
				add(acc.wt2, acc.wtsq2, acc.lik2, acc.idx2(r.t, r2.t))
			}
		}
	}
	if acc.maxOrder >= 3 {
		for i, r := range runs {
			if r.c >= 3 {
				add(acc.wt3, acc.wtsq3, acc.lik3, acc.idx3(r.t, r.t, r.t))
			}
			for j2 := i + 1; j2 < len(runs); j2++ {
				r2 := runs[j2]
				if r.c >= 2 {
					add(acc.wt3, acc.wtsq3, acc.lik3, acc.idx3(r.t, r.t, r2.t))
				}
				if r2.c >= 2 {
					add(acc.wt3, acc.wtsq3, acc.lik3, acc.idx3(r.t, r2.t, r2.t))
				}
				for l := j2 + 1; l < len(runs); l++ {
					add(acc.wt3, acc.wtsq3, acc.lik3, acc.idx3(r.t, r2.t, runs[l].t))
				}
			}
		}
	}
}

// minus returns a new accumulator holding acc's statistics with sub's
// removed. Every field is an additive count or sum, so the complement of a
// cross-fitting fold is an exact subtraction — no need to re-walk the
// samples. Rounding can leave a likelihood sum a hair below zero; those are
// clamped, since a negative mass is meaningless downstream.
func (acc *subleaveAccumulator) minus(sub *subleaveAccumulator) *subleaveAccumulator {
	out := newSubleaveAccumulator(acc.alphaSize, acc.maxOrder)
	out.n = acc.n - sub.n
	out.wtTotal = math.Max(0, acc.wtTotal-sub.wtTotal)
	out.wtsqTotal = math.Max(0, acc.wtsqTotal-sub.wtsqTotal)
	out.likTotal = math.Max(0, acc.likTotal-sub.likTotal)
	subInto := func(dst, a, b []float64) {
		for i := range dst {
			if d := a[i] - b[i]; d > 0 {
				dst[i] = d
			}
		}
	}
	subInto(out.wt1, acc.wt1, sub.wt1)
	subInto(out.wtsq1, acc.wtsq1, sub.wtsq1)
	subInto(out.lik1, acc.lik1, sub.lik1)
	if acc.maxOrder >= 2 {
		subInto(out.wt2, acc.wt2, sub.wt2)
		subInto(out.wtsq2, acc.wtsq2, sub.wtsq2)
		subInto(out.lik2, acc.lik2, sub.lik2)
	}
	if acc.maxOrder >= 3 {
		subInto(out.wt3, acc.wt3, sub.wt3)
		subInto(out.wtsq3, acc.wtsq3, sub.wtsq3)
		subInto(out.lik3, acc.lik3, sub.lik3)
	}
	return out
}

// imputationModel holds the Möbius interaction terms derived from an
// accumulator. logImputed sums them over the sub-multisets of a leave;
// uncertainty does the same for the per-term uncertainties, which drive the
// refine loop's exploration bonus.
type imputationModel struct {
	alphaSize int
	maxOrder  int
	phi1      []float64
	phi2      []float64
	phi3      []float64

	// unc* is how much of each term's raw signal was suppressed for lack of
	// support: (1 − shrink)·|log lift|, and σ₀ for sub-multisets never seen
	// at all. Same indexing as phi*.
	unc1 []float64
	unc2 []float64
	unc3 []float64

	lambda     float64
	clampLift  float64
	clampInter float64
}

func clampAbs(x, bound float64) float64 {
	if x > bound {
		return bound
	}
	if x < -bound {
		return -bound
	}
	return x
}

// buildImputationModel derives interaction terms from the accumulated
// containment marginals. lambda/clamps are parameters so tests can disable
// shrinkage and verify the exact Möbius telescoping identity.
func buildImputationModel(acc *subleaveAccumulator, lambda, clampLift, clampInter float64) *imputationModel {
	A := acc.alphaSize
	mod := &imputationModel{
		alphaSize:  A,
		maxOrder:   acc.maxOrder,
		phi1:       make([]float64, A),
		unc1:       make([]float64, A),
		lambda:     lambda,
		clampLift:  clampLift,
		clampInter: clampInter,
	}
	if acc.n == 0 || acc.likTotal <= 0 {
		// No usable signal: all φ stay 0 and the imputed likelihood is
		// constant, so the posterior degrades to the prior.
		if acc.maxOrder >= 2 {
			mod.phi2 = make([]float64, A*A)
			mod.unc2 = make([]float64, A*A)
		}
		if acc.maxOrder >= 3 {
			mod.phi3 = make([]float64, A*A*A)
			mod.unc3 = make([]float64, A*A*A)
		}
		return mod
	}

	// Importance-weighted global mean: Σu·w / Σu. With unit weights this is
	// the plain sample mean.
	wMean := acc.likTotal / acc.wtTotal
	shrink := func(c float64) float64 {
		return c / (c + lambda)
	}
	// rawLogLift = log( (lik/wt) / wMean ), clamped. lik == 0 with wt > 0 is
	// genuine "containing S kills this play" signal; it clamps low.
	rawLogLift := func(lik, wt float64) float64 {
		if lik <= 0 {
			return -clampLift
		}
		return clampAbs(math.Log(lik/wt/wMean), clampLift)
	}
	// sumSq/count of |raw| per order, for the σ₀ assigned to sub-multisets
	// never observed at all — the most uncertain terms in the model, not the
	// least.
	var sigSum [4]float64
	var sigCnt [4]float64
	// uncertainty of a term: the share of its raw signal that shrinkage
	// suppressed for lack of support.
	unc := func(order int, raw, e float64) float64 {
		sigSum[order] += raw * raw
		sigCnt[order]++
		return (1 - shrink(e)) * math.Abs(raw)
	}

	for t := 0; t < A; t++ {
		if acc.wt1[t] > 0 {
			e := ess(acc.wt1[t], acc.wtsq1[t])
			raw := rawLogLift(acc.lik1[t], acc.wt1[t])
			mod.phi1[t] = shrink(e) * raw
			mod.unc1[t] = unc(1, raw, e)
		}
	}

	if acc.maxOrder >= 2 {
		mod.phi2 = make([]float64, A*A)
		mod.unc2 = make([]float64, A*A)
		for t := 0; t < A; t++ {
			for u := t; u < A; u++ {
				j := t*A + u
				if acc.wt2[j] == 0 {
					continue
				}
				e := ess(acc.wt2[j], acc.wtsq2[j])
				raw := rawLogLift(acc.lik2[j], acc.wt2[j])
				// Subtract φ over proper nonempty sub-multisets: {t} and, if
				// t ≠ u, {u}. (Divisor lattice: for {t,t} the only proper
				// nonempty sub-multiset is {t}.)
				inter := raw - mod.phi1[t]
				if u != t {
					inter -= mod.phi1[u]
				}
				inter = clampAbs(inter, clampInter)
				mod.phi2[j] = shrink(e) * inter
				mod.unc2[j] = unc(2, inter, e)
			}
		}
	}

	if acc.maxOrder >= 3 {
		mod.phi3 = make([]float64, A*A*A)
		mod.unc3 = make([]float64, A*A*A)
		for t := 0; t < A; t++ {
			for u := t; u < A; u++ {
				for v := u; v < A; v++ {
					j := (t*A+u)*A + v
					if acc.wt3[j] == 0 {
						continue
					}
					e := ess(acc.wt3[j], acc.wtsq3[j])
					raw := rawLogLift(acc.lik3[j], acc.wt3[j])
					// Subtract φ over all distinct proper nonempty
					// sub-multisets of {t,u,v}.
					var inter float64
					switch {
					case t == u && u == v: // {t,t,t}: subs {t}, {t,t}
						inter = raw - mod.phi1[t] - mod.phi2[t*A+t]
					case t == u: // {t,t,v}: subs {t}, {v}, {t,t}, {t,v}
						inter = raw - mod.phi1[t] - mod.phi1[v] -
							mod.phi2[t*A+t] - mod.phi2[t*A+v]
					case u == v: // {t,u,u}: subs {t}, {u}, {u,u}, {t,u}
						inter = raw - mod.phi1[t] - mod.phi1[u] -
							mod.phi2[u*A+u] - mod.phi2[t*A+u]
					default: // all distinct
						inter = raw - mod.phi1[t] - mod.phi1[u] - mod.phi1[v] -
							mod.phi2[t*A+u] - mod.phi2[t*A+v] - mod.phi2[u*A+v]
					}
					inter = clampAbs(inter, clampInter)
					mod.phi3[j] = shrink(e) * inter
					mod.unc3[j] = unc(3, inter, e)
				}
			}
		}
	}

	// Sub-multisets with no support at all keep φ = 0 but are the *most*
	// uncertain terms, so they inherit σ₀: the RMS raw magnitude observed at
	// their order (0.5 nats if nothing at that order was ever seen).
	for order := 1; order <= mod.maxOrder; order++ {
		sigma0 := 0.5
		if sigCnt[order] > 0 {
			sigma0 = math.Sqrt(sigSum[order] / sigCnt[order])
		}
		switch order {
		case 1:
			fillUnobserved(mod.unc1, acc.wt1, sigma0)
		case 2:
			fillUnobserved(mod.unc2, acc.wt2, sigma0)
		case 3:
			fillUnobserved(mod.unc3, acc.wt3, sigma0)
		}
	}
	return mod
}

func fillUnobserved(unc, wt []float64, sigma0 float64) {
	for i := range unc {
		if wt[i] == 0 {
			unc[i] = sigma0
		}
	}
}

// logImputed returns log ℓ̂(L) (up to the calibration constant) for the leave
// described by runs: the sum of φ over all distinct sub-multisets of L with
// order ≤ maxOrder. Its sub-multiset walk is mirrored by subleaveTerms; keep
// the two in sync.
func (mod *imputationModel) logImputed(runs []tileRun) float64 {
	s := 0.0
	mod.walkSubleaves(runs, func(order, idx int) {
		s += mod.phiAt(order, idx)
	})
	return s
}

// uncertainty returns how unsure the model is about logImputed(runs): the
// terms' individual uncertainties combined in quadrature. It is large for
// leaves resting on thin or absent support, which is what the refine loop's
// exploration bonus keys off.
func (mod *imputationModel) uncertainty(runs []tileRun) float64 {
	s := 0.0
	mod.walkSubleaves(runs, func(order, idx int) {
		u := mod.uncAt(order, idx)
		s += u * u
	})
	return math.Sqrt(s)
}

func (mod *imputationModel) phiAt(order, idx int) float64 {
	switch order {
	case 1:
		return mod.phi1[idx]
	case 2:
		return mod.phi2[idx]
	default:
		return mod.phi3[idx]
	}
}

func (mod *imputationModel) uncAt(order, idx int) float64 {
	switch order {
	case 1:
		return mod.unc1[idx]
	case 2:
		return mod.unc2[idx]
	default:
		return mod.unc3[idx]
	}
}

// tilesAt inverts the index packing of walkSubleaves, recovering the
// sub-multiset itself. Diagnostic use only.
func (mod *imputationModel) tilesAt(order, idx int) []tilemapping.MachineLetter {
	A := mod.alphaSize
	switch order {
	case 1:
		return []tilemapping.MachineLetter{tilemapping.MachineLetter(idx)}
	case 2:
		return []tilemapping.MachineLetter{
			tilemapping.MachineLetter(idx / A), tilemapping.MachineLetter(idx % A)}
	default:
		return []tilemapping.MachineLetter{
			tilemapping.MachineLetter(idx / (A * A)),
			tilemapping.MachineLetter((idx / A) % A),
			tilemapping.MachineLetter(idx % A)}
	}
}

// walkSubleaves calls fn once for every distinct sub-multiset of order ≤
// maxOrder of the leave described by runs, with the sub-multiset's order and
// its index into the matching phi/unc array. This is the single definition of
// the expansion's term set: logImputed, uncertainty and subleaveTerms all
// drive off it, so they cannot drift apart.
func (mod *imputationModel) walkSubleaves(runs []tileRun, fn func(order, idx int)) {
	A := mod.alphaSize
	for _, r := range runs {
		fn(1, int(r.t))
	}
	if mod.maxOrder >= 2 {
		for i, r := range runs {
			if r.c >= 2 {
				fn(2, int(r.t)*A+int(r.t))
			}
			for _, r2 := range runs[i+1:] {
				fn(2, int(r.t)*A+int(r2.t))
			}
		}
	}
	if mod.maxOrder >= 3 {
		for i, r := range runs {
			if r.c >= 3 {
				fn(3, (int(r.t)*A+int(r.t))*A+int(r.t))
			}
			for j := i + 1; j < len(runs); j++ {
				r2 := runs[j]
				if r.c >= 2 {
					fn(3, (int(r.t)*A+int(r.t))*A+int(r2.t))
				}
				if r2.c >= 2 {
					fn(3, (int(r.t)*A+int(r2.t))*A+int(r2.t))
				}
				for l := j + 1; l < len(runs); l++ {
					fn(3, (int(r.t)*A+int(r2.t))*A+int(runs[l].t))
				}
			}
		}
	}
}

// subleaveTerm describes one φ term contributing to logImputed.
type subleaveTerm struct {
	tiles []tilemapping.MachineLetter // the sub-multiset, e.g. [Q] or [A,Q]
	phi   float64
}

// subleaveTerms enumerates every distinct sub-multiset of order ≤ maxOrder of
// the leave described by runs, with its φ term. Diagnostic-only, not on the
// hot path.
func (mod *imputationModel) subleaveTerms(runs []tileRun) []subleaveTerm {
	var terms []subleaveTerm
	mod.walkSubleaves(runs, func(order, idx int) {
		terms = append(terms, subleaveTerm{
			tiles: mod.tilesAt(order, idx),
			phi:   mod.phiAt(order, idx),
		})
	})
	return terms
}

// accStats returns the accumulator's containment stats for a sorted
// sub-multiset of order 1–3, for diagnostics: effective sample size, the
// importance-weight total, and the weighted likelihood sum.
func (acc *subleaveAccumulator) accStats(sub []tilemapping.MachineLetter) (e, wt, lik float64) {
	switch len(sub) {
	case 1:
		t := sub[0]
		return ess(acc.wt1[t], acc.wtsq1[t]), acc.wt1[t], acc.lik1[t]
	case 2:
		j := acc.idx2(sub[0], sub[1])
		return ess(acc.wt2[j], acc.wtsq2[j]), acc.wt2[j], acc.lik2[j]
	case 3:
		j := acc.idx3(sub[0], sub[1], sub[2])
		return ess(acc.wt3[j], acc.wtsq3[j]), acc.wt3[j], acc.lik3[j]
	}
	return 0, 0, 0
}

// measuredLeave accumulates repeated likelihood measurements of one distinct
// leave, so the posterior uses their mean instead of first-wins.
type measuredLeave struct {
	sumW  float64
	count int
	// sumU is the total importance weight of the draws that produced those
	// measurements — the leave's weight when estimating prior-weighted
	// quantities such as the calibration constant. It equals count for
	// prior-sampled draws.
	sumU float64
	// round is where the leave entered the measured set: 0 for round 0
	// (prior sampling or exhaustive enumeration), r for refine round r. A
	// leave has exactly one origin round, since refine rounds only ever draw
	// leaves that are still unmeasured.
	round int
	// predicted is the imputed likelihood the model assigned this leave just
	// before it was measured — a genuinely out-of-sample prediction, unlike
	// anything the final model produces for it. 0 for round-0 leaves, which
	// were never predicted before being drawn.
	predicted float64
}

// mean is the plain average of the measurements, deliberately unweighted:
// repeat measurements of one leave are equally noisy replicates of the same
// quantity however the leave came to be selected.
func (m *measuredLeave) mean() float64 {
	if m.count == 0 {
		return 0
	}
	return m.sumW / float64(m.count)
}

// imputationResult is the complete posterior plus diagnostics for display.
type imputationResult struct {
	racks []montecarlo.InferredRack

	measuredLeaves int
	imputedLeaves  int
	measuredMass   float64 // fraction of posterior mass on measured leaves
	marginalOrder  int

	// Retained so AnalyzeLeave can replay the calculation for any leave.
	model    *imputationModel
	logCalib float64
	maxLogW  float64 // log of the pre-normalization max weight

	// logCalibInSample is the constant the full-data model would have
	// produced without cross-fitting. Diagnostic only: the gap to logCalib
	// measures how much the lifts are fitting their own samples.
	logCalibInSample float64
	crossFitted      bool
}

// tileArena hands out small tile slices carved from large chunks, avoiding
// one allocation per leaf when the leave space is large.
type tileArena struct {
	buf []tilemapping.MachineLetter
}

func (a *tileArena) alloc(src []tilemapping.MachineLetter) []tilemapping.MachineLetter {
	if len(a.buf)+len(src) > cap(a.buf) {
		a.buf = make([]tilemapping.MachineLetter, 0, 1<<16)
	}
	start := len(a.buf)
	a.buf = append(a.buf, src...)
	return a.buf[start:len(a.buf):len(a.buf)]
}

// logSumExp returns log Σ e^x stably. Returns -Inf for an empty slice.
func logSumExp(xs []float64) float64 {
	maxX := math.Inf(-1)
	for _, x := range xs {
		if x > maxX {
			maxX = x
		}
	}
	if math.IsInf(maxX, -1) {
		return maxX
	}
	var sum float64
	for _, x := range xs {
		sum += math.Exp(x - maxX)
	}
	return maxX + math.Log(sum)
}

// calibrateLogConstant returns the constant C that aligns the model's lift
// scale with the raw softmax scale of the measured likelihoods. Measured
// likelihoods live on the raw softmax scale while logImputed is a sum of lift
// terms, so we moment-match the arithmetic means over the measured leaves:
// choose C so that Σ U·exp(C + Σφ) = Σ U·w, where U is the leave's total
// importance weight (its draw count under prior sampling). (A mean-of-logs anchor would
// target the geometric mean, which for heavily right-skewed likelihoods sits
// orders of magnitude below the arithmetic scale the lifts are estimated on,
// starving every imputed leaf of posterior mass relative to measured ones.)
//
// C is fit on the measured leaves but applied to the *unmeasured* ones, so
// the predictions entering the denominator must be out-of-sample. With the
// full-data model they are not: every measured leave helped set the very
// lifts that predict it, which inflates Σ c·e^Σφ and biases C low. When
// foldModels is non-empty each leave is instead predicted by the model fit on
// the folds excluding it (cross-fitting), which removes that self-fit. The
// second return is the uncorrected in-sample constant, for diagnostics.
//
// Degenerate folds need no special case: a fold-complement model with no data
// leaves every φ at 0, so it predicts a flat ê = 1 and C falls back to
// log(mean w) — the right answer when the lifts carry no information.
func calibrateLogConstant(measured map[string]*measuredLeave, k int,
	full *imputationModel, foldModels []*imputationModel) (logCalib, inSample float64) {

	var sumCW float64            // Σ U·w
	var lpFull, lpFold []float64 // log U + Σφ per measured leave with w > 0
	var runBuf []tileRun
	tiles := make([]tilemapping.MachineLetter, 0, k)
	for key, ml := range measured {
		w := ml.mean()
		if w <= 0 || ml.sumU <= 0 {
			continue
		}
		tiles = tiles[:0]
		for i := 0; i < len(key); i++ {
			tiles = append(tiles, tilemapping.MachineLetter(key[i]))
		}
		runBuf = runsOf(tiles, runBuf)
		logC := math.Log(ml.sumU)
		sumCW += ml.sumU * w
		lpFull = append(lpFull, logC+full.logImputed(runBuf))
		if len(foldModels) > 0 {
			lpFold = append(lpFold,
				logC+foldModels[foldForKey(key, len(foldModels))].logImputed(runBuf))
		}
	}
	if sumCW <= 0 || len(lpFull) == 0 {
		return 0, 0
	}
	logSumCW := math.Log(sumCW)
	inSample = logSumCW - logSumExp(lpFull)
	if len(lpFold) > 0 {
		return logSumCW - logSumExp(lpFold), inSample
	}
	return inSample, inSample
}

// imputeFullPosterior walks every feasible leave of size k drawable from
// bagMap and assigns posterior weight prior(L) × likelihood(L), where the
// likelihood is the measured mean when L was evaluated and the calibrated
// imputed value otherwise. Weights are normalized so the max is 1.
//
// foldAccs, when non-nil, holds the same samples as acc partitioned by leave
// into cross-fitting folds; it is used only to calibrate (see
// calibrateLogConstant). The imputed likelihoods themselves always come from
// the full-data model.
func imputeFullPosterior(bagMap []uint8, k int, acc *subleaveAccumulator,
	foldAccs []*subleaveAccumulator, measured map[string]*measuredLeave,
	threads int) *imputationResult {

	mod := buildImputationModel(acc, imputationLambda, maxAbsLogLift, maxAbsInteraction)

	foldModels := make([]*imputationModel, 0, len(foldAccs))
	for _, fa := range foldAccs {
		foldModels = append(foldModels, buildImputationModel(
			acc.minus(fa), imputationLambda, maxAbsLogLift, maxAbsInteraction))
	}
	logCalib, logCalibInSample := calibrateLogConstant(measured, k, mod, foldModels)

	N := 0
	for _, c := range bagMap {
		N += int(c)
	}
	if k <= 0 || N < k {
		return &imputationResult{marginalOrder: acc.maxOrder, model: mod,
			logCalib: logCalib, logCalibInSample: logCalibInSample,
			crossFitted: len(foldModels) > 0}
	}
	logChooseNK := logBinomial(N, k)

	// Enumerate in parallel, splitting subtrees on the smallest tile index.
	type scoredLeaf struct {
		tiles    []tilemapping.MachineLetter
		logW     float64
		measured bool
	}
	perFirst := make([][]scoredLeaf, len(bagMap))
	if threads < 1 {
		threads = 1
	}
	sem := make(chan struct{}, threads)
	eg := errgroup.Group{}

	for first := 0; first < len(bagMap); first++ {
		if bagMap[first] == 0 {
			continue
		}
		first := first
		sem <- struct{}{}
		eg.Go(func() error {
			defer func() { <-sem }()
			var local []scoredLeaf
			arena := &tileArena{}
			buf := make([]tilemapping.MachineLetter, 0, k)
			keyBuf := make([]byte, 0, k)
			var runBuf []tileRun

			// visit is called with buf holding a complete sorted leave and
			// logNum = Σ_t log C(bagMap[t], count_t).
			visit := func(logNum float64) {
				logPrior := logNum - logChooseNK
				keyBuf = keyBuf[:0]
				for _, t := range buf {
					keyBuf = append(keyBuf, byte(t))
				}
				var logW float64
				isMeasured := false
				if ml, ok := measured[string(keyBuf)]; ok {
					isMeasured = true
					w := ml.mean()
					if w <= 0 {
						return // measured to be (numerically) impossible
					}
					logW = logPrior + math.Log(w)
				} else {
					runBuf = runsOf(buf, runBuf)
					logW = logPrior + logCalib + mod.logImputed(runBuf)
				}
				local = append(local, scoredLeaf{
					tiles:    arena.alloc(buf),
					logW:     logW,
					measured: isMeasured,
				})
			}

			var recurse func(minIdx, remaining int, logNum float64)
			recurse = func(minIdx, remaining int, logNum float64) {
				if remaining == 0 {
					visit(logNum)
					return
				}
				for i := minIdx; i < len(bagMap); i++ {
					if bagMap[i] == 0 {
						continue
					}
					maxCopies := int(bagMap[i])
					if remaining < maxCopies {
						maxCopies = remaining
					}
					prevLen := len(buf)
					ln := logNum
					K := float64(bagMap[i])
					for cnt := 1; cnt <= maxCopies; cnt++ {
						buf = append(buf, tilemapping.MachineLetter(i))
						// C(K, cnt)/C(K, cnt-1) = (K-cnt+1)/cnt
						ln += math.Log((K - float64(cnt) + 1) / float64(cnt))
						recurse(i+1, remaining-cnt, ln)
					}
					buf = buf[:prevLen]
				}
			}

			// The subtree rooted at `first`: take 1..maxCopies of it, then
			// recurse over strictly larger tile indices.
			maxCopies := int(bagMap[first])
			if k < maxCopies {
				maxCopies = k
			}
			ln := 0.0
			K := float64(bagMap[first])
			for cnt := 1; cnt <= maxCopies; cnt++ {
				buf = append(buf, tilemapping.MachineLetter(first))
				ln += math.Log((K - float64(cnt) + 1) / float64(cnt))
				recurse(first+1, k-cnt, ln)
			}

			perFirst[first] = local
			return nil
		})
	}
	eg.Wait()

	// Normalize in log space and drop negligible leaves.
	maxLogW := math.Inf(-1)
	total := 0
	for _, local := range perFirst {
		total += len(local)
		for i := range local {
			if local[i].logW > maxLogW {
				maxLogW = local[i].logW
			}
		}
	}
	res := &imputationResult{
		marginalOrder:    acc.maxOrder,
		model:            mod,
		logCalib:         logCalib,
		logCalibInSample: logCalibInSample,
		crossFitted:      len(foldModels) > 0,
		maxLogW:          maxLogW,
	}
	if total == 0 || math.IsInf(maxLogW, -1) {
		return res
	}
	res.racks = make([]montecarlo.InferredRack, 0, total)
	logThreshold := math.Log(negligibleWeightFraction)
	var measuredMass, totalMass float64
	for _, local := range perFirst {
		for i := range local {
			rel := local[i].logW - maxLogW
			if rel < logThreshold {
				continue
			}
			w := math.Exp(rel)
			res.racks = append(res.racks, montecarlo.InferredRack{
				Leave:  local[i].tiles,
				Weight: w,
			})
			totalMass += w
			if local[i].measured {
				measuredMass += w
				res.measuredLeaves++
			} else {
				res.imputedLeaves++
			}
		}
	}
	if totalMass > 0 {
		res.measuredMass = measuredMass / totalMass
	}
	return res
}
