package montecarlo

import (
	"context"
	"math"
	"testing"

	"github.com/rs/zerolog"
	"lukechampine.com/frand"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
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
		reward, err := s.baiSample(thread, s.bai.arms[arm], ordinal)
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
	s.SetBAICapPerPlay(0)
	r := runBAITest(t, s)
	budget := (s.autostopper.iterationsCutoff + 2*s.autostopper.perPlyStopScaling) * 15
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
