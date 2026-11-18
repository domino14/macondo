package montecarlo

import (
	"fmt"
	"os"
	"math"
	"sync/atomic"
)

// RobustLambda, when positive, picks the winning play by an entropic risk
// measure over the opponent's posterior leaves instead of by mean utility:
// R = -1/λ · log Σ_L w_L · exp(-λ·u_L), with u_L a play's mean utility over
// the samples that drew leave L and w_L their share of its samples. It is the
// worst case over posteriors within a KL ball of the read (larger λ, larger
// ball), so it favours plays that hold up across the read's leaves over plays
// whose mean rests on a few of them. λ → 0 is the mean, λ → ∞ the worst leave.
// Without a complete posterior every sample shares one group and R is the
// mean. Only plays with at least half the most-sampled play's samples are
// eligible, so the stopping rule's early cuts stand. A lab setting.
var RobustLambda float64

// RobustChanged counts the sims where the robust pick differed from the
// mean's.
var RobustChanged, RobustPicks atomic.Int64

type leaveStat struct {
	sum float64
	n   int
}

func (sp *SimmedPlay) addLeaveSample(idx int, u float64) {
	sp.Lock()
	defer sp.Unlock()
	if sp.byLeave == nil {
		sp.byLeave = map[int]*leaveStat{}
	}
	ls := sp.byLeave[idx]
	if ls == nil {
		ls = &leaveStat{}
		sp.byLeave[idx] = ls
	}
	ls.sum += u
	ls.n++
}

// robustUtility is the play's R at lambda.
func (sp *SimmedPlay) robustUtility(lambda float64) float64 {
	sp.RLock()
	defer sp.RUnlock()
	total := 0
	for _, ls := range sp.byLeave {
		total += ls.n
	}
	if total == 0 {
		return math.Inf(-1)
	}
	// Shift by the smallest leave mean so the exponentials stay in range.
	lo := math.Inf(1)
	for _, ls := range sp.byLeave {
		lo = min(lo, ls.sum/float64(ls.n))
	}
	acc := 0.0
	for _, ls := range sp.byLeave {
		acc += float64(ls.n) / float64(total) * math.Exp(-lambda*(ls.sum/float64(ls.n)-lo))
	}
	return lo - math.Log(acc)/lambda
}

// robustWinner picks from plays, sorted with the active ones first, the
// eligible play of highest R.
func robustWinner(plays []*SimmedPlay) *SimmedPlay {
	maxN := 0
	for _, p := range plays {
		if !p.ignore.Load() {
			maxN = max(maxN, int(p.utilityStats.Iterations()))
		}
	}
	best, bestR := plays[0], math.Inf(-1)
	for _, p := range plays {
		if p.ignore.Load() || 2*int(p.utilityStats.Iterations()) < maxN {
			continue
		}
		if r := p.robustUtility(RobustLambda); r > bestR {
			best, bestR = p, r
		}
	}
	if os.Getenv("ROBUST_DEBUG") != "" {
		for _, p := range plays {
			fmt.Fprintf(os.Stderr, "  %-14s ign %v n %5d groups %4d mean %.4f R %.4f\n", p.play.ShortDescription(), p.ignore.Load(), p.utilityStats.Iterations(), len(p.byLeave), p.utilityStats.Mean(), p.robustUtility(RobustLambda))
		}
		fmt.Fprintln(os.Stderr, "  ->", best.play.ShortDescription())
	}
	RobustPicks.Add(1)
	if best != plays[0] {
		RobustChanged.Add(1)
	}
	return best
}
