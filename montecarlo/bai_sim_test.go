package montecarlo

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/rs/zerolog"
	"lukechampine.com/frand"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/montecarlo/bai"
	"github.com/domino14/macondo/stats"
)

// baiTestSimmer is a seeded sim of a midgame position, set up for best-arm
// identification.
func baiTestSimmer(t *testing.T, seedByte byte, threads, plays int) *Simmer {
	t.Helper()
	const pos = "15/15/15/15/15/4FENCERS4/6EALE5/5CWM7/5A9/2NONRATED5/5B9/5I4FAX2/3PUNITORY4/5E9/3PISHEOG5 AANORST/ 187/210 0 lex NWL18;"
	parsed, err := cgp.ParseCGP(DefaultConfig, pos)
	if err != nil {
		t.Fatal(err)
	}
	parsed.RecalculateBoard()
	g := parsed.Game
	var seed [32]byte
	seed[0] = seedByte
	g.SeedBag(seed)
	calcs, leaves := defaultSimCalculators("NWL18")
	p, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(g, DefaultConfig, calcs)
	if err != nil {
		t.Fatal(err)
	}
	s := &Simmer{}
	s.Init(g, calcs, leaves.(*equity.CombinedStaticCalculator), DefaultConfig)
	s.SetThreads(threads)
	s.SetBAI(true)
	s.SetStoppingCondition(Stop99)
	// Short sims, for the tests' sake.
	s.SetBAICapPerPlay(700)
	if err := s.PrepareSim(2, p.GenerateMoves(plays)); err != nil {
		t.Fatal(err)
	}
	return s
}

type baiTestResult struct {
	best           string
	rollouts       int
	status         string
	counts         []int
	wins, equities []float64
}

func runBAITest(t *testing.T, s *Simmer) baiTestResult {
	t.Helper()
	if err := s.Simulate(context.Background()); err != nil {
		t.Fatal(err)
	}
	r := baiTestResult{
		best:     s.WinningPlay().play.ShortDescription(),
		rollouts: s.bai.runner.Samples(),
		status:   s.BAIStatus(),
	}
	for _, a := range s.bai.arms {
		r.counts = append(r.counts, a.winPctStats.Iterations())
		r.wins = append(r.wins, a.winPctStats.Mean())
		r.equities = append(r.equities, a.equityStats.Mean())
	}
	return r
}

// A seeded best-arm identification sim gives the same answer, from the
// same rollouts, on one thread and on several: a sample depends only on the
// play and its number, and rounds are folded in order.
func TestBAISimIsReproducible(t *testing.T) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	defer zerolog.SetGlobalLevel(zerolog.DebugLevel)
	var first baiTestResult
	for i, threads := range []int{1, 1, 3, 8} {
		r := runBAITest(t, baiTestSimmer(t, 9, threads, 12))
		if i == 0 {
			first = r
			if r.rollouts == 0 || r.status == "none" {
				t.Fatalf("the sim did not run: %+v", r)
			}
			continue
		}
		if r.best != first.best || r.rollouts != first.rollouts || r.status != first.status {
			t.Fatalf("%d threads: %s after %d rollouts (%s), one thread: %s after %d (%s)",
				threads, r.best, r.rollouts, r.status, first.best, first.rollouts, first.status)
		}
		for a := range r.counts {
			if r.counts[a] != first.counts[a] {
				t.Fatalf("%d threads: play %d was rolled out %d times, one thread %d", threads, a, r.counts[a], first.counts[a])
			}
			// The same rollouts in another order: means agree to rounding.
			if math.Abs(r.wins[a]-first.wins[a]) > 1e-9 || math.Abs(r.equities[a]-first.equities[a]) > 1e-6 {
				t.Fatalf("%d threads: play %d's means %v, %v; one thread %v, %v", threads, a,
					r.wins[a], r.equities[a], first.wins[a], first.equities[a])
			}
		}
	}
	// Another seed is another sim.
	other := runBAITest(t, baiTestSimmer(t, 10, 1, 12))
	same := other.rollouts == first.rollouts
	for a := range other.wins {
		same = same && other.wins[a] == first.wins[a]
	}
	if same {
		t.Fatal("two seeds gave the same sim")
	}
}

// Sample number n of a play is the same rollout whichever thread draws it
// and whatever was drawn before it, and deals the opponent the same rack as
// sample n of any other play.
func TestBAISamplesShareTheirLuck(t *testing.T) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	defer zerolog.SetGlobalLevel(zerolog.DebugLevel)
	s := baiTestSimmer(t, 3, 4, 6)
	s.startBAI()
	s.rngs = make([]*frand.RNG, 4)
	opp := 1 - s.initialPlayer
	type draw struct {
		reward float64
		rack   string
	}
	sample := func(thread, arm int, ordinal uint64) draw {
		// The rack the sample deals is the one on the copy as the rollout
		// starts; the hook sees the rollout's end, so read it before.
		reward, err := s.baiSample(thread, s.bai.arms[arm], ordinal, nil)
		if err != nil {
			t.Fatal(err)
		}
		// baiSample resets the copy to the dealt position.
		return draw{reward, s.gameCopies[thread].RackLettersFor(opp)}
	}
	want := map[[2]uint64]draw{}
	for arm := 0; arm < 6; arm++ {
		for ordinal := uint64(0); ordinal < 20; ordinal++ {
			want[[2]uint64{uint64(arm), ordinal}] = sample(0, arm, ordinal)
		}
	}
	for ordinal := uint64(0); ordinal < 20; ordinal++ {
		for arm := 1; arm < 6; arm++ {
			if a, b := want[[2]uint64{0, ordinal}], want[[2]uint64{uint64(arm), ordinal}]; a.rack != b.rack {
				t.Fatalf("sample %d deals %s to play 0's rollout and %s to play %d's", ordinal, a.rack, b.rack, arm)
			}
		}
	}
	racks := map[string]bool{}
	for ordinal := uint64(0); ordinal < 20; ordinal++ {
		racks[want[[2]uint64{0, ordinal}].rack] = true
	}
	if len(racks) < 15 {
		t.Fatalf("20 samples dealt only %d different racks", len(racks))
	}
	// Other threads, in another order.
	for thread := 3; thread >= 1; thread-- {
		for ordinal := uint64(19); ; ordinal-- {
			for arm := 5; arm >= 0; arm-- {
				got := sample(thread, arm, ordinal)
				if got != want[[2]uint64{uint64(arm), ordinal}] {
					t.Fatalf("thread %d: sample %d of play %d is %+v, on thread 0 it was %+v",
						thread, ordinal, arm, got, want[[2]uint64{uint64(arm), ordinal}])
				}
			}
			if ordinal == 0 {
				break
			}
		}
	}
}

// With no stopping condition, or switched off, a sim is not a best-arm
// identification run; with one it is, it stops short of its budget when it
// has told the best play apart, and its pick comes first.
func TestBAISimStops(t *testing.T) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	defer zerolog.SetGlobalLevel(zerolog.DebugLevel)
	s := baiTestSimmer(t, 5, 4, 15)
	// A close position: it takes more than the default budget to settle.
	s.SetBAICapPerPlay(5000)
	r := runBAITest(t, s)
	budget := 5000 * 15
	if r.status != "threshold" || r.rollouts >= budget {
		t.Fatalf("stopped by %s after %d rollouts of a budget of %d", r.status, r.rollouts, budget)
	}
	if s.bai.winner == nil || s.WinningPlay() != s.bai.winner {
		t.Fatal("the sim's pick is not the winning play")
	}
	// Every play got its minimum, and the rollouts went mostly to a few.
	most := 0
	for _, c := range r.counts {
		if c < DefaultBAIMinSamples {
			t.Fatalf("a play was rolled out %d times, under the minimum", c)
		}
		most = max(most, c)
	}
	if most <= DefaultBAIMinSamples {
		t.Fatal("no play was rolled out more than the minimum")
	}

	off := baiTestSimmer(t, 5, 1, 15)
	off.SetBAI(false)
	if off.usesBAI() {
		t.Fatal("best-arm identification runs when switched off")
	}
	none := baiTestSimmer(t, 5, 1, 15)
	none.SetStoppingCondition(StopNone)
	if none.usesBAI() {
		t.Fatal("best-arm identification runs without a stopping condition")
	}
}

// A time limit ends a sim, of either kind, that would otherwise run on; the
// best-arm sim still names its play.
func TestSimTimeLimit(t *testing.T) {
	for _, useBAI := range []bool{true, false} {
		s := baiTestSimmer(t, 1, 4, 20)
		s.SetBAI(useBAI)
		s.SetBAICapPerPlay(1000000)
		s.SetAutostopIterationsCutoff(1000000000)
		s.SetStoppingCondition(Stop999)
		s.SetTimeLimit(300 * time.Millisecond)
		start := time.Now()
		if err := s.Simulate(context.Background()); err != nil {
			t.Fatal(err)
		}
		took := time.Since(start)
		if took < 250*time.Millisecond || took > 5*time.Second {
			t.Fatalf("bai=%v: a 300 ms sim took %v", useBAI, took)
		}
		if useBAI && (s.BAIStatus() != bai.StatusTimeout.String() || s.bai.winner == nil) {
			t.Fatalf("status %v, winner %v", s.BAIStatus(), s.bai.winner)
		}
	}
}

// A bag-cycled sim rolls every play out in whole cycles, keeps to its
// budget, and is the same sim on any number of threads.
func TestBAISimBagCycle(t *testing.T) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	defer zerolog.SetGlobalLevel(zerolog.DebugLevel)
	var first baiTestResult
	for i, threads := range []int{1, 3, 8, 16} {
		s := baiTestSimmer(t, 9, threads, 12)
		s.SetBAIBagCycle(true)
		r := runBAITest(t, s)
		rot := s.bai.rotations
		if rot < 2 {
			t.Fatalf("%d rotations in a position with a full bag", rot)
		}
		budget := 700 * 12
		if r.rollouts == 0 || r.rollouts > budget+rot {
			t.Fatalf("%d rollouts of a budget of %d", r.rollouts, budget)
		}
		for a, c := range r.counts {
			if c%rot != 0 || c < DefaultBAIMinSamples {
				t.Fatalf("play %d was rolled out %d times: cycles are %d rollouts, the minimum %d",
					a, c, rot, DefaultBAIMinSamples)
			}
		}
		if i == 0 {
			first = r
			continue
		}
		if r.best != first.best || r.rollouts != first.rollouts || r.status != first.status {
			t.Fatalf("%d threads: %s after %d rollouts (%s), one thread: %s after %d (%s)",
				threads, r.best, r.rollouts, r.status, first.best, first.rollouts, first.status)
		}
		for a := range r.counts {
			if r.counts[a] != first.counts[a] || math.Abs(r.wins[a]-first.wins[a]) > 1e-9 {
				t.Fatalf("%d threads: play %d differs from one thread's", threads, a)
			}
		}
	}
	plain := runBAITest(t, baiTestSimmer(t, 9, 1, 12))
	if plain.rollouts == first.rollouts && plain.wins[0] == first.wins[0] {
		t.Fatal("bag cycling changed nothing")
	}
}

// A sample drawn ahead of the schedule is the sample drawn on schedule: it
// leaves its play's statistics alone until it is committed, and then they
// are what drawing it on schedule would have made them.
func TestBAIHeldSample(t *testing.T) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	defer zerolog.SetGlobalLevel(zerolog.DebugLevel)
	for _, cycle := range []bool{false, true} {
		direct, ahead := baiTestSimmer(t, 5, 1, 6), baiTestSimmer(t, 5, 1, 6)
		for _, s := range []*Simmer{direct, ahead} {
			s.SetBAIBagCycle(cycle)
			s.startBAI()
			s.rngs = make([]*frand.RNG, 1)
		}
		const arm, samples = 2, 12
		var held []*baiHeld
		for ordinal := uint64(0); ordinal < samples; ordinal++ {
			want, err := direct.baiSample(0, direct.bai.arms[arm], ordinal, nil)
			if err != nil {
				t.Fatal(err)
			}
			h := &baiHeld{tally: rolloutTally{play: ahead.bai.arms[arm]}}
			got, err := ahead.baiSample(0, ahead.bai.arms[arm], ordinal, h)
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("cycle=%v sample %d: %v ahead of the schedule, %v on it", cycle, ordinal, got, want)
			}
			held = append(held, h)
		}
		d, a := direct.bai.arms[arm], ahead.bai.arms[arm]
		if a.winPctStats.Iterations() != 0 || a.equityStats.Iterations() != 0 || a.scoreStats[0].Iterations() != 0 {
			t.Fatalf("cycle=%v: samples not yet used are in the play's statistics", cycle)
		}
		for _, h := range held {
			ahead.baiCommit(h)
		}
		if n := d.winPctStats.Iterations(); n != samples*direct.bai.rotations {
			t.Fatalf("cycle=%v: %d rollouts tallied for %d samples of %d", cycle, n, samples, direct.bai.rotations)
		}
		same := func(what string, x, y *stats.Statistic) {
			if x.Iterations() != y.Iterations() || x.Mean() != y.Mean() || x.Variance() != y.Variance() {
				t.Fatalf("cycle=%v %s: %d rollouts mean %v ahead, %d mean %v on schedule", cycle, what,
					x.Iterations(), x.Mean(), y.Iterations(), y.Mean())
			}
		}
		same("win", &a.winPctStats, &d.winPctStats)
		same("equity", &a.equityStats, &d.equityStats)
		same("leftover", &a.leftoverStats, &d.leftoverStats)
		for ply := range d.scoreStats {
			same("score", &a.scoreStats[ply], &d.scoreStats[ply])
			same("bingo", &a.bingoStats[ply], &d.bingoStats[ply])
		}
	}
}
