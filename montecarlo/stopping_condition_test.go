package montecarlo

import (
	"math"
	"testing"

	"github.com/domino14/macondo/stats"
	"github.com/matryer/is"
)

// pushAlternating pushes [mean-d, mean+d] n times (2n values total), giving
// a Statistic with the specified mean and a non-trivial standard error.
// With 2n pushes, SE ≈ d / sqrt(2n-1) for large n.
func pushAlternating(s *stats.Statistic, mean, d float64, n int) {
	for i := 0; i < n; i++ {
		s.Push(mean - d)
		s.Push(mean + d)
	}
}

func newTestStopper(sc StoppingCondition) *AutoStopper {
	a := newAutostopper()
	a.stoppingCondition = sc
	return a
}

// ucbLCBc computes the confidence multiplier for the given parameters,
// mirroring the formula in shouldStop. Exported for test assertions.
func ucbLCBc(K, round, delta float64) float64 {
	if K < 2 {
		K = 2
	}
	if round < 1 {
		round = 1
	}
	c := math.Sqrt(2 * math.Log(2*K*round*round/delta))
	if c < 1.0 {
		c = 1.0
	}
	return c
}

// TestShouldStop_EarlyExits checks the fast-exit conditions that don't involve statistics.
func TestShouldStop_EarlyExits(t *testing.T) {
	is := is.New(t)
	a := newTestStopper(Stop95)

	// Fewer than 2 plays → always stop.
	sp1 := &SimmedPlays{plays: []*SimmedPlay{{}}}
	is.True(a.shouldStop(0, sp1, 2))

	// Iteration count beyond the cutoff → stop.
	// defaultIterationsCutoff=2000, maxPlies=2, perPlyStopScaling=625 → ceiling=3250
	sp2 := &SimmedPlays{plays: []*SimmedPlay{{}, {}}}
	is.True(a.shouldStop(4000, sp2, 2))
}

// TestShouldStop_PrunesClearlyInferior checks that a play with a much lower
// win probability is pruned when samples are plentiful (tight SE).
// With 200 pushes each, SE ≈ 0.001, c ≈ 2.96:
//
//	LCB(leader=0.80) ≈ 0.797 >> UCB(arm=0.20) ≈ 0.203  →  pruned
func TestShouldStop_PrunesClearlyInferior(t *testing.T) {
	is := is.New(t)
	a := newTestStopper(Stop95)

	leader := &SimmedPlay{}
	arm := &SimmedPlay{}

	pushAlternating(&leader.winPctStats, 0.80, 0.01, 100) // 200 pushes, SE≈0.001
	pushAlternating(&leader.equityStats, 15.0, 1.0, 100)
	pushAlternating(&arm.winPctStats, 0.20, 0.01, 100)
	pushAlternating(&arm.equityStats, 5.0, 1.0, 100)

	sp := &SimmedPlays{plays: []*SimmedPlay{leader, arm}}
	result := a.shouldStop(0, sp, 2)

	is.True(result)         // only leader left, stop
	is.True(arm.ignore)     // arm was pruned
	is.True(!leader.ignore) // leader untouched
}

// TestShouldStop_DoesNotPruneClosePlay checks that plays with similar win
// probabilities are NOT pruned when samples are sparse (wide SE).
// With 4 pushes each and d=0.10, SE ≈ 0.058, c ≈ 2.96:
//
//	LCB(leader=0.55) ≈ 0.38  <  UCB(arm=0.45) ≈ 0.62  →  not pruned
func TestShouldStop_DoesNotPruneClosePlay(t *testing.T) {
	is := is.New(t)
	a := newTestStopper(Stop95)

	leader := &SimmedPlay{}
	arm := &SimmedPlay{}

	pushAlternating(&leader.winPctStats, 0.55, 0.10, 2) // 4 pushes, SE≈0.058
	pushAlternating(&leader.equityStats, 12.0, 1.0, 2)
	pushAlternating(&arm.winPctStats, 0.45, 0.10, 2)
	pushAlternating(&arm.equityStats, 8.0, 1.0, 2)

	sp := &SimmedPlays{plays: []*SimmedPlay{leader, arm}}
	result := a.shouldStop(0, sp, 2)

	is.True(!result)     // can't distinguish yet, keep going
	is.True(!arm.ignore) // arm was NOT pruned
}

// TestShouldStop_EquityTiebreak checks that when all plays have near-zero
// win probability, the autostopper switches to equity-based comparison and
// prunes the clearly lower-equity arm.
// win pct ≈ 0.001 << minReasonableWProb (0.005) → tiebreakByEquity=true
// equity: leader=20.0 vs arm=5.0, tight SE → arm pruned by LCB(leader) > UCB(arm)
func TestShouldStop_EquityTiebreak(t *testing.T) {
	is := is.New(t)
	a := newTestStopper(Stop95)

	leader := &SimmedPlay{}
	arm := &SimmedPlay{}

	// Both near-zero win probability so the equity tiebreak triggers.
	// Leader must be first in the simmedPlays slice to avoid the log path
	// that calls play.ShortDescription() (play is nil in tests).
	pushAlternating(&leader.winPctStats, 0.001, 0.0005, 100)
	pushAlternating(&leader.equityStats, 20.0, 0.5, 100) // clear equity leader
	pushAlternating(&arm.winPctStats, 0.001, 0.0005, 100)
	pushAlternating(&arm.equityStats, 5.0, 0.5, 100)

	sp := &SimmedPlays{plays: []*SimmedPlay{leader, arm}}
	result := a.shouldStop(0, sp, 2)

	is.True(result)         // arm pruned by equity difference
	is.True(arm.ignore)
	is.True(!leader.ignore)
}

// TestShouldStop_UnignorablePlayNeverPruned checks that a play marked
// unignorable is never pruned regardless of its win probability.
func TestShouldStop_UnignorablePlayNeverPruned(t *testing.T) {
	is := is.New(t)
	a := newTestStopper(Stop95)

	leader := &SimmedPlay{}
	arm := &SimmedPlay{}
	arm.unignorable = true

	pushAlternating(&leader.winPctStats, 0.80, 0.01, 100)
	pushAlternating(&leader.equityStats, 15.0, 1.0, 100)
	pushAlternating(&arm.winPctStats, 0.20, 0.01, 100)
	pushAlternating(&arm.equityStats, 5.0, 1.0, 100)

	sp := &SimmedPlays{plays: []*SimmedPlay{leader, arm}}
	result := a.shouldStop(0, sp, 2)

	is.True(!result)     // unignorable arm keeps sim alive
	is.True(!arm.ignore) // was not pruned
}

// TestShouldStop_RoundCountGrows checks that the round counter increments on
// each call and that c grows accordingly, making later rounds more conservative.
func TestShouldStop_RoundCountGrows(t *testing.T) {
	is := is.New(t)
	a := newTestStopper(Stop95)
	is.Equal(a.roundCount, 0)

	leader := &SimmedPlay{}
	arm := &SimmedPlay{}
	pushAlternating(&leader.winPctStats, 0.80, 0.01, 100)
	pushAlternating(&leader.equityStats, 15.0, 1.0, 100)
	pushAlternating(&arm.winPctStats, 0.20, 0.01, 100)
	pushAlternating(&arm.equityStats, 5.0, 1.0, 100)

	sp := &SimmedPlays{plays: []*SimmedPlay{leader, arm}}
	a.shouldStop(0, sp, 2)
	is.Equal(a.roundCount, 1)

	// c grows with round, so bounds widen (more conservative) over time.
	// Even though we're pruning with lots of samples, the formula itself
	// must produce a larger c for round=2 than round=1.
	c1 := ucbLCBc(2, 1, 0.05)
	c2 := ucbLCBc(2, 2, 0.05)
	is.True(c2 > c1)
}

// TestShouldStop_ThreePlaysTwoPruned checks that shouldStop returns true when
// two out of three arms get pruned, leaving only the leader.
func TestShouldStop_ThreePlaysTwoPruned(t *testing.T) {
	is := is.New(t)
	a := newTestStopper(Stop95)

	leader := &SimmedPlay{}
	arm1 := &SimmedPlay{}
	arm2 := &SimmedPlay{}

	pushAlternating(&leader.winPctStats, 0.80, 0.01, 100)
	pushAlternating(&leader.equityStats, 15.0, 1.0, 100)
	pushAlternating(&arm1.winPctStats, 0.15, 0.01, 100)
	pushAlternating(&arm1.equityStats, 3.0, 1.0, 100)
	pushAlternating(&arm2.winPctStats, 0.05, 0.01, 100)
	pushAlternating(&arm2.equityStats, 1.0, 1.0, 100)

	sp := &SimmedPlays{plays: []*SimmedPlay{leader, arm1, arm2}}
	result := a.shouldStop(0, sp, 2)

	is.True(result)          // both arms pruned
	is.True(!leader.ignore)
	is.True(arm1.ignore)
	is.True(arm2.ignore)
}

// TestZTest checks the one-shot z-test used for equity tiebreak detection.
func TestZTest(t *testing.T) {
	is := is.New(t)
	is.True(!zTest(100, 96, 1.62, -stats.Z98, false))
	// actual significance is about 98.6%
	is.True(zTest(100, 96, 1.62, -stats.Z99, false))

	// Are we 99% sure that 0.002 < 0.005 with the given stderr? Yes
	is.True(zTest(0.005, 0.002, 0.001, -stats.Z99, true))
	// Are we 99% sure that 0.998 > 0.995 with the given stderr? Yes
	is.True(zTest(0.995, 0.998, 0.001, stats.Z99, false))
	// We are NOT 99% sure that 0.997 > 0.995 with the given stderr.
	is.True(!zTest(0.995, 0.997, 0.001, stats.Z99, false))
	is.True(!zTest(0.995, 0.990, 0.001, stats.Z99, false))
}
