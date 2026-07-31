package rangefinder

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"

	"github.com/domino14/macondo/ai/simplesimmer"
	"github.com/domino14/macondo/move"
)

// This file implements the measure–impute–recalibrate loop described in
// docs/development/iterative_inference_plan.md. Round 0 (the prior-sampled MC
// pass in inference.go, or exhaustive enumeration) leaves us with a model
// fitted to whatever it happened to see. Each refine round then draws leaves
// from the *current* posterior — biased toward high weight, with a bonus for
// leaves the model is unsure about — evaluates them for real, and refits. The
// draws carry importance weights u = P/q so the lift estimator stays unbiased
// under a proposal that is nothing like the prior.

const (
	// DefaultMaxRefineRounds bounds the loop regardless of convergence.
	DefaultMaxRefineRounds = 6

	// refineExplorationLambda scales the uncertainty bonus in the proposal:
	// q ∝ W · (1 + λ·unc). 0 is pure exploitation.
	refineExplorationLambda = 1.0

	// refineStage0Frac is the share of the time budget spent on round 0
	// (blind, prior-sampled exploration) before refinement starts.
	refineStage0Frac = 0.4

	// refineConvergedTol is the scale error we are willing to live with in the
	// imputed set: the loop stops once the whole confidence interval of
	// log R̂ fits inside ±this. 10% is far below the model's shape error.
	refineConvergedTol = 0.10

	// refineMassCovered stops the loop once the unmeasured leaves hold less
	// than this share of posterior mass — whatever the model still gets wrong
	// there cannot move the posterior much.
	refineMassCovered = 0.02

	// refineMinBatch is the smallest worthwhile round; below this the ratio
	// statistic is too noisy to learn from.
	refineMinBatch = 16
)

// evaluateLeaves runs the mini-sim likelihood measurement for each leave in
// parallel over the worker threads, calling onResult for each. onResult may be
// called from any goroutine; callers synchronize. Returns early (without
// error) when ctx is done, so partial batches are usable.
func (r *RangeFinder) evaluateLeaves(ctx context.Context, leaves [][]tilemapping.MachineLetter,
	onResult func(leave []tilemapping.MachineLetter, likelihood float64)) error {

	if len(leaves) == 0 {
		return nil
	}
	var nextIdx atomic.Int64
	eg := errgroup.Group{}
	for t := 0; t < r.threads; t++ {
		t := t
		eg.Go(func() error {
			gc := r.gameCopies[t]
			opp := gc.PlayerOnTurn()
			simmer := r.aiplayers[t].(*simplesimmer.SimpleSimmer)

			// fullRack is reused across iterations to avoid per-leaf allocation.
			fullRack := make([]tilemapping.MachineLetter,
				len(r.lastOppMoveRackTiles)+r.inference.RackLength)
			copy(fullRack, r.lastOppMoveRackTiles)

			for {
				i := int(nextIdx.Add(1)) - 1
				if i >= len(leaves) {
					return nil
				}
				if ctx.Err() != nil {
					return nil
				}
				leave := leaves[i]

				// Build the full deterministic rack: played tiles + this leave.
				// Passing a full RackTileLimit-length knownRack to SetRandomRack
				// causes it to: (1) put back the old rack, (2) remove fullRack
				// from the bag, (3) draw 0 additional tiles — an exact,
				// deterministic rack assignment.
				copy(fullRack[len(r.lastOppMoveRackTiles):], leave)
				if _, err := gc.SetRandomRack(opp, fullRack); err != nil {
					return err
				}

				lastOppMove := &move.Move{}
				lastOppMove.CopyFrom(r.lastOppMove)
				lastOppMove.SetLeave(leave)

				if _, err := simmer.GenAndSim(context.Background(), 10, lastOppMove); err != nil {
					return err
				}
				r.simCount.Add(1)

				bestPlays := simmer.BestPlays().PlaysNoLock()
				// Zero-likelihood leaves are reported too — they are measured
				// evidence, not missing data, so they must not be imputed.
				likelihood, _ := softmaxLikelihood(bestPlays, lastOppMove, gc.Board(), r.Tau())
				onResult(leave, likelihood)
			}
		})
	}
	return eg.Wait()
}

// refineCandidate is one unmeasured leave with everything the sampler needs.
type refineCandidate struct {
	tiles  []tilemapping.MachineLetter
	weight float64 // current posterior weight (normalized by the max)
	q      float64 // proposal probability, normalized over the candidates
	u      float64 // importance weight P/q
	lhat   float64 // the model's current imputed likelihood, for the ratio test
}

// buildProposal forms the round's sampling distribution over leaves that have
// not been measured yet:
//
//	q(L) ∝ W(L) · (1 + λ·unc(L))
//
// exploiting posterior weight while spending extra draws where the model's own
// terms rest on thin support. Returns the candidates with normalized q and
// importance weights u = P/q, plus the share of total posterior mass sitting
// on unmeasured leaves.
func (r *RangeFinder) buildProposal(lambdaEx float64) (cands []refineCandidate, unmeasuredMass float64) {
	res := r.imputeRes
	if res == nil || res.model == nil {
		return nil, 0
	}
	var totalW, unmeasuredW, qTotal float64
	var runBuf []tileRun
	for _, ir := range r.inference.InferredRacks {
		totalW += ir.Weight
		key := leaveKey(ir.Leave)
		if ml, ok := r.measured[key]; ok && ml.count > 0 {
			continue
		}
		unmeasuredW += ir.Weight
		runBuf = runsOf(ir.Leave, runBuf)
		q := ir.Weight * (1 + lambdaEx*res.model.uncertainty(runBuf))
		if q <= 0 {
			continue
		}
		qTotal += q
		cands = append(cands, refineCandidate{
			tiles:  ir.Leave,
			weight: ir.Weight,
			q:      q,
			lhat:   math.Exp(res.logCalib + res.model.logImputed(runBuf)),
		})
	}
	if totalW > 0 {
		unmeasuredMass = unmeasuredW / totalW
	}
	if qTotal <= 0 {
		return nil, unmeasuredMass
	}
	// Normalize q and derive u = P/q. An unmeasured leave's stored weight is
	// P·ℓ̂/maxW, so its prior is Weight·maxW/ℓ̂ — exact, and cheaper than
	// recomputing combinatorialPrior for tens of thousands of leaves.
	maxW := math.Exp(res.maxLogW)
	for i := range cands {
		cands[i].q /= qTotal
		prior := 0.0
		if cands[i].lhat > 0 {
			prior = cands[i].weight * maxW / cands[i].lhat
		}
		cands[i].u = prior / cands[i].q
	}
	return cands, unmeasuredMass
}

// truncateWeights caps importance weights at sqrt(n)·mean (Ionides), bounding
// the variance a single lucky draw from the model's tail can inject into the
// lifts. Returns the number of weights that were capped.
func truncateWeights(cands []refineCandidate) int {
	if len(cands) == 0 {
		return 0
	}
	var sum float64
	for _, c := range cands {
		sum += c.u
	}
	limit := math.Sqrt(float64(len(cands))) * sum / float64(len(cands))
	capped := 0
	for i := range cands {
		if cands[i].u > limit {
			cands[i].u = limit
			capped++
		}
	}
	return capped
}

// systematicSample draws m indices proportional to size from the (normalized)
// probabilities q, using one uniform offset and m equally spaced positions on
// the CDF. Every item with q ≥ 1/m is drawn at least once, and E[draws of i] =
// m·q(i) exactly, so the importance weights stay valid while the variance is
// far below i.i.d. sampling. Returns indices, with repeats where q is large.
func systematicSample(q []float64, m int, rng *rand.Rand) []int {
	if m <= 0 || len(q) == 0 {
		return nil
	}
	step := 1.0 / float64(m)
	pos := rng.Float64() * step
	out := make([]int, 0, m)
	cum := 0.0
	i := 0
	for len(out) < m && i < len(q) {
		cum += q[i]
		for len(out) < m && pos <= cum {
			out = append(out, i)
			pos += step
		}
		i++
	}
	// Floating-point slack can leave the last position just past the final
	// cumulative total; assign any remainder to the last positive-q item.
	for len(out) < m {
		last := len(q) - 1
		for last > 0 && q[last] <= 0 {
			last--
		}
		out = append(out, last)
	}
	return out
}

// roundStats records one refine round for logging and the stopping rule.
type roundStats struct {
	round int
	// drawn is how many PPS draws the round asked for; distinct is how many
	// distinct leaves those draws landed on (a leave holding ≥ 1/drawn of the
	// proposal can come up more than once); evaluated is how many of those
	// actually finished before the deadline and were recorded.
	drawn      int
	distinct   int
	evaluated  int
	logRatio   float64 // log R̂: measured mass ÷ predicted mass on this batch
	seLogRatio float64
	unmeasured float64 // share of posterior mass still unmeasured
	converged  bool
}

// ratioStatistic computes R̂ = Σ u·w / Σ u·ℓ̂ over a freshly evaluated batch —
// all leaves previously unmeasured, so the predictions are genuinely
// out-of-sample — together with the linearized standard error of the ratio
// estimator. This is the loop's convergence test: it asks whether the model
// was right about the mass it just pointed at.
func ratioStatistic(us, ws, lhats []float64) (ratio, se float64) {
	var num, den float64
	for i := range us {
		num += us[i] * ws[i]
		den += us[i] * lhats[i]
	}
	if den <= 0 {
		return 0, 0
	}
	ratio = num / den
	var v float64
	for i := range us {
		resid := ws[i] - ratio*lhats[i]
		v += us[i] * us[i] * resid * resid
	}
	se = math.Sqrt(v) / den
	return ratio, se
}

// refineRounds runs the measure–impute–recalibrate loop until it converges,
// exhausts the leave space, covers the posterior mass, runs out of rounds, or
// runs out of time. It assumes finalizePlacementPosterior has already produced
// a first posterior from round 0.
func (r *RangeFinder) refineRounds(ctx context.Context, maxRounds int) {
	if maxRounds <= 0 || r.acc == nil {
		return
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	for round := 1; round <= maxRounds; round++ {
		if ctx.Err() != nil {
			return
		}
		r.currentRound = round
		cands, unmeasuredMass := r.buildProposal(refineExplorationLambda)
		if len(cands) == 0 {
			log.Info().Int("round", round).Msg("refine-space-exhausted")
			return
		}
		if unmeasuredMass < refineMassCovered {
			log.Info().Int("round", round).Float64("unmeasured-mass", unmeasuredMass).
				Msg("refine-mass-covered")
			return
		}

		batch := r.roundBatchSize(ctx, round, maxRounds, len(cands))
		if batch <= 0 {
			log.Info().Int("round", round).Msg("refine-budget-exhausted")
			return
		}
		truncated := truncateWeights(cands)

		qs := make([]float64, len(cands))
		for i, c := range cands {
			qs[i] = c.q
		}
		picks := systematicSample(qs, batch, rng)

		// De-duplicate for evaluation but keep multiplicity in the weights: a
		// leave drawn twice contributes 2u, and its measurements average into
		// one w̄. (PPS hands repeats only to leaves with q ≥ 1/m, where the
		// extra precision is worth the most.)
		type batchEntry struct {
			cand int // index into cands
			mult int
			w    float64
			got  bool
		}
		entries := make([]batchEntry, 0, len(picks))
		entryOf := make(map[int]int, len(picks))
		for _, p := range picks {
			if at, ok := entryOf[p]; ok {
				entries[at].mult++
				continue
			}
			entryOf[p] = len(entries)
			entries = append(entries, batchEntry{cand: p, mult: 1})
		}
		toEval := make([][]tilemapping.MachineLetter, len(entries))
		keyToEntry := make(map[string]int, len(entries))
		for i, e := range entries {
			toEval[i] = cands[e.cand].tiles
			keyToEntry[leaveKey(cands[e.cand].tiles)] = i
		}

		var mu sync.Mutex
		err := r.evaluateLeaves(ctx, toEval, func(leave []tilemapping.MachineLetter, lik float64) {
			mu.Lock()
			defer mu.Unlock()
			if i, ok := keyToEntry[leaveKey(leave)]; ok {
				entries[i].w = lik
				entries[i].got = true
			}
		})
		if err != nil {
			log.Err(err).Int("round", round).Msg("refine-evaluate-failed")
			return
		}

		// Record the evaluated draws and collect the ratio statistic.
		var us, ws, lhats []float64
		evaluated := 0
		for _, e := range entries {
			if !e.got {
				continue // context expired before this one ran
			}
			c := cands[e.cand]
			u := c.u * float64(e.mult)
			r.recordPlacementSample(c.tiles, e.w, u)
			// Keep what the model predicted before this measurement: once it
			// refits below, no out-of-sample prediction for this leave exists
			// anywhere else.
			if ml, ok := r.measured[leaveKey(c.tiles)]; ok {
				ml.predicted = c.lhat
			}
			r.refinedCount++
			evaluated++
			us = append(us, u)
			ws = append(ws, e.w)
			lhats = append(lhats, c.lhat)
		}
		if evaluated == 0 {
			return
		}

		ratio, se := ratioStatistic(us, ws, lhats)
		st := roundStats{round: round, drawn: batch, distinct: len(entries),
			evaluated: evaluated, unmeasured: unmeasuredMass}
		if ratio > 0 {
			st.logRatio = math.Log(ratio)
			st.seLogRatio = se / ratio
			// Equivalence test, not a significance test: stop only when the
			// entire interval sits inside the tolerance band. A wide interval
			// means the batch could not detect miscalibration — that is a
			// reason to keep measuring, not to declare victory.
			st.converged = math.Abs(st.logRatio)+2*st.seLogRatio < refineConvergedTol
		}
		r.roundLog = append(r.roundLog, st)

		// Refit and re-impute with the new measurements before the next round.
		r.finalizePlacementPosterior()

		log.Info().Int("round", round).
			Int("drawn", st.drawn).Int("distinct-leaves", st.distinct).
			Int("evaluated", st.evaluated).
			Int("truncated-weights", truncated).
			Float64("log-ratio", st.logRatio).Float64("se", st.seLogRatio).
			Float64("unmeasured-mass", unmeasuredMass).
			Bool("converged", st.converged).
			Msg("refine-round")

		if st.converged {
			return
		}
	}
}

// roundBatchSize divides the time left among the rounds left, converting to a
// leaf count via the evaluation rate observed in round 0. A round smaller than
// refineMinBatch is not worth running — the ratio statistic would be pure
// noise — so a thin share is rounded up to the minimum and the loop simply
// runs fewer, larger rounds. Returns 0 when even one minimum batch will not
// fit in the time that remains.
func (r *RangeFinder) roundBatchSize(ctx context.Context, round, maxRounds, candidates int) int {
	deadline, ok := ctx.Deadline()
	if !ok {
		// No deadline: take an even slice of the candidate space per round.
		return min(candidates/maxRounds+1, candidates)
	}
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return 0
	}
	roundsLeft := maxRounds - round + 1

	// Evaluations per second, measured on round 0 across all threads.
	rate := 0.0
	if r.stage0Elapsed > 0 {
		rate = float64(r.simCount.Load()) / r.stage0Elapsed.Seconds()
	}
	if rate <= 0 {
		rate = float64(r.threads) * 10 // ~100ms per mini-sim, conservative
	}

	b := int(rate * remaining.Seconds() / float64(roundsLeft))
	if b < refineMinBatch {
		if int(rate*remaining.Seconds()) < refineMinBatch {
			return 0 // no time for a worthwhile round
		}
		b = refineMinBatch
	}
	return min(b, candidates)
}
