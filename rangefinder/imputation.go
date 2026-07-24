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
)

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

// subleaveAccumulator accumulates containment-marginal statistics
// (count and likelihood-sum) for every distinct sub-multiset of order ≤
// maxOrder of each recorded leave. Not goroutine-safe; callers synchronize.
type subleaveAccumulator struct {
	alphaSize int
	maxOrder  int

	n      int     // total recorded leaves
	wTotal float64 // total likelihood mass

	cnt1, wsum1 []float64 // [A], index t
	cnt2, wsum2 []float64 // [A*A], index t*A+u with t ≤ u
	cnt3, wsum3 []float64 // [A*A*A], index (t*A+u)*A+v with t ≤ u ≤ v

	runBuf []tileRun
}

func newSubleaveAccumulator(alphaSize, maxOrder int) *subleaveAccumulator {
	acc := &subleaveAccumulator{alphaSize: alphaSize, maxOrder: maxOrder}
	acc.cnt1 = make([]float64, alphaSize)
	acc.wsum1 = make([]float64, alphaSize)
	if maxOrder >= 2 {
		acc.cnt2 = make([]float64, alphaSize*alphaSize)
		acc.wsum2 = make([]float64, alphaSize*alphaSize)
	}
	if maxOrder >= 3 {
		acc.cnt3 = make([]float64, alphaSize*alphaSize*alphaSize)
		acc.wsum3 = make([]float64, alphaSize*alphaSize*alphaSize)
	}
	return acc
}

func (acc *subleaveAccumulator) idx2(t, u tilemapping.MachineLetter) int {
	return int(t)*acc.alphaSize + int(u)
}

func (acc *subleaveAccumulator) idx3(t, u, v tilemapping.MachineLetter) int {
	return (int(t)*acc.alphaSize+int(u))*acc.alphaSize + int(v)
}

// record adds one evaluated leave (sorted ascending) with its measured
// likelihood w (which may be 0) to the accumulator.
func (acc *subleaveAccumulator) record(sorted []tilemapping.MachineLetter, w float64) {
	acc.n++
	acc.wTotal += w
	acc.runBuf = runsOf(sorted, acc.runBuf)
	runs := acc.runBuf

	for _, r := range runs {
		acc.cnt1[r.t]++
		acc.wsum1[r.t] += w
	}
	if acc.maxOrder >= 2 {
		for i, r := range runs {
			if r.c >= 2 {
				j := acc.idx2(r.t, r.t)
				acc.cnt2[j]++
				acc.wsum2[j] += w
			}
			for _, r2 := range runs[i+1:] {
				j := acc.idx2(r.t, r2.t)
				acc.cnt2[j]++
				acc.wsum2[j] += w
			}
		}
	}
	if acc.maxOrder >= 3 {
		for i, r := range runs {
			if r.c >= 3 {
				j := acc.idx3(r.t, r.t, r.t)
				acc.cnt3[j]++
				acc.wsum3[j] += w
			}
			for j2 := i + 1; j2 < len(runs); j2++ {
				r2 := runs[j2]
				if r.c >= 2 {
					j := acc.idx3(r.t, r.t, r2.t)
					acc.cnt3[j]++
					acc.wsum3[j] += w
				}
				if r2.c >= 2 {
					j := acc.idx3(r.t, r2.t, r2.t)
					acc.cnt3[j]++
					acc.wsum3[j] += w
				}
				for l := j2 + 1; l < len(runs); l++ {
					j := acc.idx3(r.t, r2.t, runs[l].t)
					acc.cnt3[j]++
					acc.wsum3[j] += w
				}
			}
		}
	}
}

// imputationModel holds the Möbius interaction terms derived from an
// accumulator. logImputed sums them over the sub-multisets of a leave.
type imputationModel struct {
	alphaSize int
	maxOrder  int
	phi1      []float64
	phi2      []float64
	phi3      []float64

	lambda      float64
	clampLift   float64
	clampInter  float64
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
		lambda:     lambda,
		clampLift:  clampLift,
		clampInter: clampInter,
	}
	if acc.n == 0 || acc.wTotal <= 0 {
		// No usable signal: all φ stay 0 and the imputed likelihood is
		// constant, so the posterior degrades to the prior.
		if acc.maxOrder >= 2 {
			mod.phi2 = make([]float64, A*A)
		}
		if acc.maxOrder >= 3 {
			mod.phi3 = make([]float64, A*A*A)
		}
		return mod
	}

	wMean := acc.wTotal / float64(acc.n)
	shrink := func(c float64) float64 {
		return c / (c + lambda)
	}
	// rawLogLift = log( (wsum/cnt) / wMean ), clamped. wsum == 0 with cnt > 0
	// is genuine "containing S kills this play" signal; it clamps low.
	rawLogLift := func(wsum, cnt float64) float64 {
		if wsum <= 0 {
			return -clampLift
		}
		return clampAbs(math.Log(wsum/cnt/wMean), clampLift)
	}

	for t := 0; t < A; t++ {
		if acc.cnt1[t] > 0 {
			mod.phi1[t] = shrink(acc.cnt1[t]) * rawLogLift(acc.wsum1[t], acc.cnt1[t])
		}
	}

	if acc.maxOrder >= 2 {
		mod.phi2 = make([]float64, A*A)
		for t := 0; t < A; t++ {
			for u := t; u < A; u++ {
				j := t*A + u
				if acc.cnt2[j] == 0 {
					continue
				}
				raw := rawLogLift(acc.wsum2[j], acc.cnt2[j])
				// Subtract φ over proper nonempty sub-multisets: {t} and, if
				// t ≠ u, {u}. (Divisor lattice: for {t,t} the only proper
				// nonempty sub-multiset is {t}.)
				inter := raw - mod.phi1[t]
				if u != t {
					inter -= mod.phi1[u]
				}
				mod.phi2[j] = shrink(acc.cnt2[j]) * clampAbs(inter, clampInter)
			}
		}
	}

	if acc.maxOrder >= 3 {
		mod.phi3 = make([]float64, A*A*A)
		for t := 0; t < A; t++ {
			for u := t; u < A; u++ {
				for v := u; v < A; v++ {
					j := (t*A+u)*A + v
					if acc.cnt3[j] == 0 {
						continue
					}
					raw := rawLogLift(acc.wsum3[j], acc.cnt3[j])
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
					mod.phi3[j] = shrink(acc.cnt3[j]) * clampAbs(inter, clampInter)
				}
			}
		}
	}
	return mod
}

// logImputed returns log ℓ̂(L) (up to the calibration constant) for the leave
// described by runs: the sum of φ over all distinct sub-multisets of L with
// order ≤ maxOrder. Its sub-multiset walk is mirrored by subleaveTerms; keep
// the two in sync.
func (mod *imputationModel) logImputed(runs []tileRun) float64 {
	A := mod.alphaSize
	s := 0.0
	for _, r := range runs {
		s += mod.phi1[r.t]
	}
	if mod.maxOrder >= 2 {
		for i, r := range runs {
			if r.c >= 2 {
				s += mod.phi2[int(r.t)*A+int(r.t)]
			}
			for _, r2 := range runs[i+1:] {
				s += mod.phi2[int(r.t)*A+int(r2.t)]
			}
		}
	}
	if mod.maxOrder >= 3 {
		for i, r := range runs {
			if r.c >= 3 {
				s += mod.phi3[(int(r.t)*A+int(r.t))*A+int(r.t)]
			}
			for j := i + 1; j < len(runs); j++ {
				r2 := runs[j]
				if r.c >= 2 {
					s += mod.phi3[(int(r.t)*A+int(r.t))*A+int(r2.t)]
				}
				if r2.c >= 2 {
					s += mod.phi3[(int(r.t)*A+int(r2.t))*A+int(r2.t)]
				}
				for l := j + 1; l < len(runs); l++ {
					s += mod.phi3[(int(r.t)*A+int(r2.t))*A+int(runs[l].t)]
				}
			}
		}
	}
	return s
}

// subleaveTerm describes one φ term contributing to logImputed.
type subleaveTerm struct {
	tiles []tilemapping.MachineLetter // the sub-multiset, e.g. [Q] or [A,Q]
	phi   float64
}

// subleaveTerms enumerates every distinct sub-multiset of order ≤ maxOrder of
// the leave described by runs, with its φ term. It mirrors the walk in
// logImputed (keep the two in sync); it is diagnostic-only and not on the hot
// path.
func (mod *imputationModel) subleaveTerms(runs []tileRun) []subleaveTerm {
	A := mod.alphaSize
	var terms []subleaveTerm
	for _, r := range runs {
		terms = append(terms, subleaveTerm{
			tiles: []tilemapping.MachineLetter{r.t},
			phi:   mod.phi1[r.t],
		})
	}
	if mod.maxOrder >= 2 {
		for i, r := range runs {
			if r.c >= 2 {
				terms = append(terms, subleaveTerm{
					tiles: []tilemapping.MachineLetter{r.t, r.t},
					phi:   mod.phi2[int(r.t)*A+int(r.t)],
				})
			}
			for _, r2 := range runs[i+1:] {
				terms = append(terms, subleaveTerm{
					tiles: []tilemapping.MachineLetter{r.t, r2.t},
					phi:   mod.phi2[int(r.t)*A+int(r2.t)],
				})
			}
		}
	}
	if mod.maxOrder >= 3 {
		for i, r := range runs {
			if r.c >= 3 {
				terms = append(terms, subleaveTerm{
					tiles: []tilemapping.MachineLetter{r.t, r.t, r.t},
					phi:   mod.phi3[(int(r.t)*A+int(r.t))*A+int(r.t)],
				})
			}
			for j := i + 1; j < len(runs); j++ {
				r2 := runs[j]
				if r.c >= 2 {
					terms = append(terms, subleaveTerm{
						tiles: []tilemapping.MachineLetter{r.t, r.t, r2.t},
						phi:   mod.phi3[(int(r.t)*A+int(r.t))*A+int(r2.t)],
					})
				}
				if r2.c >= 2 {
					terms = append(terms, subleaveTerm{
						tiles: []tilemapping.MachineLetter{r.t, r2.t, r2.t},
						phi:   mod.phi3[(int(r.t)*A+int(r2.t))*A+int(r2.t)],
					})
				}
				for l := j + 1; l < len(runs); l++ {
					terms = append(terms, subleaveTerm{
						tiles: []tilemapping.MachineLetter{r.t, r2.t, runs[l].t},
						phi:   mod.phi3[(int(r.t)*A+int(r2.t))*A+int(runs[l].t)],
					})
				}
			}
		}
	}
	return terms
}

// accStats returns the accumulator's raw containment stats (sample count and
// likelihood sum) for a sorted sub-multiset of order 1–3, for diagnostics.
func (acc *subleaveAccumulator) accStats(sub []tilemapping.MachineLetter) (cnt, wsum float64) {
	switch len(sub) {
	case 1:
		return acc.cnt1[sub[0]], acc.wsum1[sub[0]]
	case 2:
		j := acc.idx2(sub[0], sub[1])
		return acc.cnt2[j], acc.wsum2[j]
	case 3:
		j := acc.idx3(sub[0], sub[1], sub[2])
		return acc.cnt3[j], acc.wsum3[j]
	}
	return 0, 0
}

// measuredLeave accumulates repeated likelihood measurements of one distinct
// leave, so the posterior uses their mean instead of first-wins.
type measuredLeave struct {
	sumW  float64
	count int
}

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

// imputeFullPosterior walks every feasible leave of size k drawable from
// bagMap and assigns posterior weight prior(L) × likelihood(L), where the
// likelihood is the measured mean when L was evaluated and the calibrated
// imputed value otherwise. Weights are normalized so the max is 1.
func imputeFullPosterior(bagMap []uint8, k int, acc *subleaveAccumulator,
	measured map[string]*measuredLeave, threads int) *imputationResult {

	mod := buildImputationModel(acc, imputationLambda, maxAbsLogLift, maxAbsInteraction)

	// Calibration: measured likelihoods live on the raw softmax scale while
	// logImputed is a sum of lift terms. Moment-match the arithmetic means:
	// choose logCalib so that Σ c·exp(logCalib + Σφ) = Σ c·w over measured
	// leaves. (A mean-of-logs anchor would target the geometric mean, which
	// for heavily right-skewed likelihoods sits orders of magnitude below the
	// arithmetic scale the lifts are estimated on, starving every imputed
	// leaf of posterior mass relative to measured ones.)
	logCalib := 0.0
	{
		var sumCW float64      // Σ c·w
		var logCPhis []float64 // log c + Σφ per measured leave with w > 0
		var runBuf []tileRun
		tiles := make([]tilemapping.MachineLetter, 0, k)
		for key, ml := range measured {
			w := ml.mean()
			if w <= 0 {
				continue
			}
			tiles = tiles[:0]
			for i := 0; i < len(key); i++ {
				tiles = append(tiles, tilemapping.MachineLetter(key[i]))
			}
			runBuf = runsOf(tiles, runBuf)
			sumCW += float64(ml.count) * w
			logCPhis = append(logCPhis,
				math.Log(float64(ml.count))+mod.logImputed(runBuf))
		}
		if sumCW > 0 && len(logCPhis) > 0 {
			// log Σ c·e^Σφ via log-sum-exp for stability.
			maxLP := math.Inf(-1)
			for _, lp := range logCPhis {
				if lp > maxLP {
					maxLP = lp
				}
			}
			var expSum float64
			for _, lp := range logCPhis {
				expSum += math.Exp(lp - maxLP)
			}
			logCalib = math.Log(sumCW) - (maxLP + math.Log(expSum))
		}
	}

	N := 0
	for _, c := range bagMap {
		N += int(c)
	}
	if k <= 0 || N < k {
		return &imputationResult{marginalOrder: acc.maxOrder, model: mod, logCalib: logCalib}
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
		marginalOrder: acc.maxOrder,
		model:         mod,
		logCalib:      logCalib,
		maxLogW:       maxLogW,
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
