package rangefinder

import (
	"math"

	"github.com/domino14/word-golib/tilemapping"
)

// Scoring inference against the truth.
//
// In a self-play experiment the opponent's real leave is known, so inference can
// be graded rather than merely inspected: look up what probability the posterior
// placed on the leave the opponent actually held. On its own that number says
// little, because a two-tile leave has a few hundred candidates and a five-tile
// one has thousands, so the probabilities shrink with the size of the space. The
// comparison that carries meaning is against the hypergeometric prior -- what the
// tile counts alone would have said, with no reasoning about the opponent's play.
//
// LiftBits is that comparison, log2(posterior/prior): the bits of information
// inference added about the true leave. Zero means it did no better than counting
// tiles, positive means it learned something, and negative means it was actively
// misled. Averaged over the turns of a run it is a single number that can be
// compared across tau values or budgets, and being a log score it cannot be
// gamed by hedging.

// LeaveScore grades the posterior against one particular leave.
type LeaveScore struct {
	// Weight is the leave's raw posterior weight, before normalizing.
	Weight float64
	// Posterior is the leave's share of the posterior, normalized to sum to 1
	// over every inferred leave. Zero when the leave is not in the posterior.
	Posterior float64
	// Prior is the leave's multivariate hypergeometric probability given the
	// unseen pool -- the answer with no inference at all.
	Prior float64
	// LiftBits is log2(Posterior/Prior), the information inference added about
	// this leave. NaN when either probability is zero, which leaves nothing to
	// take a ratio of.
	LiftBits float64
	// Rank is the leave's 1-based position among inferred leaves by weight, or 0
	// when it is not in the posterior.
	Rank int
	// Leaves is how many leaves the posterior covers, for scale.
	Leaves int
	// Measured is true when a mini-sim actually evaluated this leave, false when
	// its weight came from the imputation model.
	Measured bool
	// InPosterior is false when the leave carries no weight at all: either it is
	// impossible from the unseen pool, or it was measured as never producing the
	// play the opponent made.
	InPosterior bool
}

// ScoreLeave grades the current posterior against the given leave. It reports
// zero-valued fields, rather than an error, for a leave the posterior does not
// cover, since "inference ruled the truth out" is itself a result worth logging.
func (r *RangeFinder) ScoreLeave(leave []tilemapping.MachineLetter) LeaveScore {
	score := LeaveScore{LiftBits: math.NaN()}
	if r.inference == nil || len(r.inference.InferredRacks) == 0 {
		return score
	}
	score.Leaves = len(r.inference.InferredRacks)
	for _, ml := range leave {
		if int(ml) >= len(r.inferenceBagMap) {
			return score
		}
	}
	key := leaveKey(leave)

	// One pass for the normalizing total, the leave's weight, and its rank.
	sumW, weight := 0.0, -1.0
	for _, ir := range r.inference.InferredRacks {
		sumW += ir.Weight
		if weight < 0 && leaveKey(ir.Leave) == key {
			weight = ir.Weight
		}
	}
	score.Prior = combinatorialPrior(leave, r.inferenceBagMap)
	if ml, ok := r.measured[key]; ok && ml.count > 0 {
		score.Measured = true
	}
	if weight < 0 || sumW <= 0 {
		return score
	}
	score.InPosterior = true
	score.Weight = weight
	score.Posterior = weight / sumW
	rank := 1
	for _, ir := range r.inference.InferredRacks {
		if ir.Weight > weight {
			rank++
		}
	}
	score.Rank = rank
	if score.Posterior > 0 && score.Prior > 0 {
		score.LiftBits = math.Log2(score.Posterior / score.Prior)
	}
	return score
}
