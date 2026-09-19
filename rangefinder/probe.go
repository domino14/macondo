package rangefinder

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/domino14/word-golib/tilemapping"
)

// Probing whether the imputation ranks leaves in the right order.
//
// The posterior a bot samples from is mostly imputed: a few hundred leaves have
// a measured likelihood and tens of thousands have one estimated from marginal
// lifts. What matters about the estimates is not their scale but their order --
// a leave the model ranks above the truth gets sampled instead of it, and one
// ranked below never gets sampled at all. Neither error shows in the outcome
// alone, since a bad read looks the same whether the model promoted junk over
// the answer or demoted the answer under junk.
//
// A probe takes one leave of interest, splits the unmeasured posterior at its
// rank, draws leaves from either side, and measures them for real without
// feeding anything back. Each probed leaf then carries both what the model
// believed and what the mini-sim found, and the comparison with the reference
// leaf says which way the ordering went wrong.

// ProbedLeaf is one leave the probe measured.
type ProbedLeaf struct {
	Leave string
	// Rank is the leave's 1-based position by posterior weight among every
	// leave in the posterior, measured or not.
	Rank int
	// Above is true when the model ranked this leave above the reference.
	Above bool
	// Stratum says how the leave was chosen: "top" for the highest imputed
	// weights, "above" or "below" for a uniform draw from that side.
	Stratum string
	// Imputed is the calibrated likelihood the model assigned, and Measured
	// is what the mini-sim found. Prior is the tile-count probability.
	Imputed, Measured, Prior float64
}

// OrderingProbe is the result of probing around one reference leave.
type OrderingProbe struct {
	// Reference is the leave of interest. When it was measured during the
	// inference its Imputed is the model's prediction from before that
	// measurement, or zero if it was measured in round 0 with no model yet.
	Reference   ProbedLeaf
	RefMeasured bool
	// RefInPosterior is false when the reference carries no weight at all,
	// in which case every probed leaf counts as above it.
	RefInPosterior bool
	// Unmeasured counts the imputed leaves on each side of the reference.
	UnmeasuredAbove, UnmeasuredBelow int
	Probes                           []ProbedLeaf
}

// ProbeOrdering measures leaves the model ranked above and below ref, without
// changing the posterior. topN takes the highest-weighted unmeasured leaves
// above ref, and above and below draw uniformly from the rest of each side.
// The reference itself is measured too if the inference never did.
func (r *RangeFinder) ProbeOrdering(ctx context.Context, ref []tilemapping.MachineLetter,
	topN, above, below int, seed int64) (*OrderingProbe, error) {

	res := r.imputeRes
	if r.inference == nil || res == nil || res.model == nil {
		return nil, nil
	}
	alph := r.origGame.Alphabet()
	show := func(l []tilemapping.MachineLetter) string {
		return tilemapping.MachineWord(l).UserVisible(alph)
	}

	// Rank everything by weight, the same order ScoreLeave reports.
	ranked := make([]int, len(r.inference.InferredRacks))
	for i := range ranked {
		ranked[i] = i
	}
	racks := r.inference.InferredRacks
	sort.SliceStable(ranked, func(a, b int) bool {
		return racks[ranked[a]].Weight > racks[ranked[b]].Weight
	})

	refKey := leaveKey(ref)
	refPos := len(ranked) // past the end: below everything
	for pos, i := range ranked {
		if leaveKey(racks[i].Leave) == refKey {
			refPos = pos
			break
		}
	}

	out := &OrderingProbe{RefInPosterior: refPos < len(ranked)}
	maxW := math.Exp(res.maxLogW)
	var runBuf []tileRun
	imputedOf := func(leave []tilemapping.MachineLetter, weight float64) (lhat, prior float64) {
		runBuf = runsOf(leave, runBuf)
		lhat = math.Exp(res.logCalib + res.model.logImputed(runBuf))
		if lhat > 0 {
			prior = weight * maxW / lhat
		}
		return lhat, prior
	}
	isMeasured := func(leave []tilemapping.MachineLetter) bool {
		ml, ok := r.measured[leaveKey(leave)]
		return ok && ml.count > 0
	}

	// Split the unmeasured leaves at the reference.
	var aboveIdx, belowIdx []int // positions in ranked
	for pos, i := range ranked {
		if pos == refPos || isMeasured(racks[i].Leave) {
			continue
		}
		if pos < refPos {
			aboveIdx = append(aboveIdx, pos)
		} else {
			belowIdx = append(belowIdx, pos)
		}
	}
	out.UnmeasuredAbove, out.UnmeasuredBelow = len(aboveIdx), len(belowIdx)

	rng := rand.New(rand.NewSource(seed))
	pick := func(from []int, n int) []int {
		if n >= len(from) {
			return append([]int(nil), from...)
		}
		perm := rng.Perm(len(from))[:n]
		sort.Ints(perm)
		chosen := make([]int, n)
		for i, p := range perm {
			chosen[i] = from[p]
		}
		return chosen
	}
	chosen := map[int]string{} // ranked position -> stratum
	nTop := min(topN, len(aboveIdx))
	for _, pos := range aboveIdx[:nTop] {
		chosen[pos] = "top"
	}
	for _, pos := range pick(aboveIdx[nTop:], above) {
		chosen[pos] = "above"
	}
	for _, pos := range pick(belowIdx, below) {
		chosen[pos] = "below"
	}

	// What to measure: the chosen leaves, plus the reference unless the
	// inference already did.
	var toEval [][]tilemapping.MachineLetter
	byKey := map[string]int{}
	for pos := range chosen {
		leave := racks[ranked[pos]].Leave
		byKey[leaveKey(leave)] = len(toEval)
		toEval = append(toEval, leave)
	}
	refML, refWasMeasured := r.measured[refKey]
	refWasMeasured = refWasMeasured && refML.count > 0
	if !refWasMeasured {
		byKey[refKey] = len(toEval)
		toEval = append(toEval, ref)
	}

	found := make([]float64, len(toEval))
	got := make([]bool, len(toEval))
	var mu sync.Mutex
	err := r.evaluateLeaves(ctx, toEval, func(leave []tilemapping.MachineLetter, lik float64) {
		mu.Lock()
		defer mu.Unlock()
		if i, ok := byKey[leaveKey(leave)]; ok {
			found[i], got[i] = lik, true
		}
	})
	if err != nil {
		return nil, err
	}

	// The reference.
	out.RefMeasured = refWasMeasured
	out.Reference = ProbedLeaf{Leave: show(ref), Rank: refPos + 1, Stratum: "reference"}
	if refPos < len(ranked) {
		w := racks[ranked[refPos]].Weight
		lhat, prior := imputedOf(ref, w)
		out.Reference.Prior = prior
		if refWasMeasured {
			out.Reference.Measured = refML.mean()
			out.Reference.Imputed = refML.predicted
		} else {
			out.Reference.Imputed = lhat
		}
	} else {
		out.Reference.Rank = 0
		out.Reference.Prior = combinatorialPrior(ref, r.inferenceBagMap)
	}
	if !refWasMeasured {
		if i := byKey[refKey]; got[i] {
			out.Reference.Measured = found[i]
		}
	}

	// The probes, in rank order.
	positions := make([]int, 0, len(chosen))
	for pos := range chosen {
		positions = append(positions, pos)
	}
	sort.Ints(positions)
	for _, pos := range positions {
		leave := racks[ranked[pos]].Leave
		i := byKey[leaveKey(leave)]
		if !got[i] {
			continue
		}
		lhat, prior := imputedOf(leave, racks[ranked[pos]].Weight)
		out.Probes = append(out.Probes, ProbedLeaf{
			Leave: show(leave), Rank: pos + 1, Above: pos < refPos,
			Stratum: chosen[pos], Imputed: lhat, Measured: found[i], Prior: prior,
		})
	}
	return out, nil
}
