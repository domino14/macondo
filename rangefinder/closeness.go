package rangefinder

import (
	"math"

	"github.com/domino14/word-golib/tilemapping"
)

// Grading a read with partial credit.
//
// LiftBits (score.go) looks only at the probability of the exact true leave, so
// a read that puts its weight on AENRS when the opponent kept AENST scores the
// same as one that puts it on QVWXZ. For the sim the first read is nearly as
// good as a perfect one. Closeness generalizes the lift with a similarity
// kernel K(L, T) in [0, 1], K(T, T) = 1:
//
//	soft lift = log2( sum_L post(L) K(L,T) / sum_L prior(L) K(L,T) )
//
// With K = [L == T] it is the lift exactly. Two kernels:
//
//	overlap  2^-(tiles of T not in L): each missed tile halves the credit
//	value    exp(-d/ValueKernelScale), d = |V(L) - V(T)| + SwapCost per tile
//	         of T not in L. Swapping T for R in AENST costs about a point;
//	         swapping S for Q costs the difference in leave value. The swap
//	         cost keeps unrelated leaves of equal value apart.
//
// Expectations under the prior use the posterior's own support with prior
// weights, which is the whole leave space when the posterior is complete.

// ValueKernelScale is the leave-value distance, in points, at which the value
// kernel gives 1/e credit.
const ValueKernelScale = 3.0

// SwapCost is the leave-value distance, in points, charged per tile of the
// true leave the other leave lacks, on top of the difference in value.
const SwapCost = 1.0

// Closeness is a read graded against the true leave with partial credit.
type Closeness struct {
	// Overlap is the expected share of the true leave's tiles a leave drawn
	// from the posterior holds; OverlapPrior the same under the prior.
	Overlap, OverlapPrior float64
	// ValueDist is the expected leave-value path distance to the truth, in
	// points; ValueDistPrior the same under the prior.
	ValueDist, ValueDistPrior float64
	// SoftLiftOverlap and SoftLiftValue are the soft lifts, in bits, under the
	// overlap and value kernels.
	SoftLiftOverlap, SoftLiftValue float64
	// TrueValue is V(true leave); MeanValue and MeanValuePrior are E[V(L)].
	TrueValue, MeanValue, MeanValuePrior float64
}

// Posterior returns the current read: each leave with its normalized weight,
// and whether it covers every feasible leave.
func (r *RangeFinder) Posterior() ([][]tilemapping.MachineLetter, []float64, bool) {
	if r.inference == nil || len(r.inference.InferredRacks) == 0 {
		return nil, nil, false
	}
	total := 0.0
	for _, ir := range r.inference.InferredRacks {
		total += ir.Weight
	}
	if total <= 0 {
		return nil, nil, false
	}
	leaves := make([][]tilemapping.MachineLetter, len(r.inference.InferredRacks))
	weights := make([]float64, len(leaves))
	for i, ir := range r.inference.InferredRacks {
		leaves[i] = ir.Leave
		weights[i] = ir.Weight / total
	}
	return leaves, weights, r.inference.Complete
}

// LeavePrior is a leave's hypergeometric probability given the unseen pool.
func (r *RangeFinder) LeavePrior(leave []tilemapping.MachineLetter) float64 {
	return combinatorialPrior(leave, r.inferenceBagMap)
}

// LeaveValuer returns the static leave value function inference uses, or nil.
func (r *RangeFinder) LeaveValuer() func(tilemapping.MachineWord) float64 {
	type valuer interface {
		LeaveValue(leave tilemapping.MachineWord) float64
	}
	for _, c := range r.equityCalculators {
		if lv, ok := c.(valuer); ok {
			return lv.LeaveValue
		}
	}
	return nil
}

// ReadCloseness grades the posterior against the true leave. ok is false when
// there is no complete posterior or no leave valuer.
func (r *RangeFinder) ReadCloseness(trueLeave []tilemapping.MachineLetter) (Closeness, bool) {
	leaves, post, complete := r.Posterior()
	valueOf := r.LeaveValuer()
	if !complete || valueOf == nil || len(trueLeave) == 0 {
		return Closeness{}, false
	}
	prior := make([]float64, len(leaves))
	ptot := 0.0
	for i, l := range leaves {
		prior[i] = combinatorialPrior(l, r.inferenceBagMap)
		ptot += prior[i]
	}
	if ptot <= 0 {
		return Closeness{}, false
	}
	return closeness(trueLeave, leaves, post, prior, ptot, valueOf), true
}

func closeness(trueLeave []tilemapping.MachineLetter, leaves [][]tilemapping.MachineLetter,
	post, prior []float64, ptot float64, valueOf func(tilemapping.MachineWord) float64) Closeness {

	var tc [256]int
	for _, t := range trueLeave {
		tc[t]++
	}
	n := float64(len(trueLeave))
	vT := valueOf(tilemapping.MachineWord(trueLeave))
	c := Closeness{TrueValue: vT}
	var kOp, kOq, kVp, kVq float64
	for i, l := range leaves {
		var used [256]int
		common := 0
		for _, t := range l {
			if used[t] < tc[t] {
				used[t]++
				common++
			}
		}
		shared := float64(common)
		vL := valueOf(tilemapping.MachineWord(l))
		d := math.Abs(vL-vT) + SwapCost*(n-shared)
		kO := math.Exp2(shared - n)
		kV := math.Exp(-d / ValueKernelScale)
		p, q := post[i], prior[i]/ptot
		c.Overlap += p * shared / n
		c.OverlapPrior += q * shared / n
		c.ValueDist += p * d
		c.ValueDistPrior += q * d
		c.MeanValue += p * vL
		c.MeanValuePrior += q * vL
		kOp += p * kO
		kOq += q * kO
		kVp += p * kV
		kVq += q * kV
	}
	c.SoftLiftOverlap = math.Log2(kOp / kOq)
	c.SoftLiftValue = math.Log2(kVp / kVq)
	return c
}
