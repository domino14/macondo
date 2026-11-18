package montecarlo

import (
	"context"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"github.com/domino14/macondo/move"
)

func pairedTestSimmer(t *testing.T, seed byte, threads, plays int) *Simmer {
	s := baiTestSimmer(t, seed, threads, plays)
	s.SetBAI(false)
	s.SetPairedStop(true)
	return s
}

// pairedSpacedSimmer is for the tests that need the sim to reach its
// threshold: its plays are ten ranks apart, so that never takes long.
func pairedSpacedSimmer(t *testing.T, seed byte, threads, plays int) *Simmer {
	s := spacedTestSimmer(t, seed, threads, plays, 10)
	s.SetBAI(false)
	s.SetPairedStop(true)
	return s
}

// A paired sim stops on its own with one play left, its pick first, having
// dropped the others.
func TestPairedSimStops(t *testing.T) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	defer zerolog.SetGlobalLevel(zerolog.DebugLevel)
	for run := 0; run < 1; run++ {
		s := pairedSpacedSimmer(t, 5, 1, 15)
		const capPerPlay = 20000
		s.SetPairedCapPerPlay(capPerPlay)
		if err := s.Simulate(context.Background()); err != nil {
			t.Fatal(err)
		}
		if s.PairedStatus() != "threshold" {
			t.Fatalf("stopped by %q after %d rollouts", s.PairedStatus(), s.PairedRollouts())
		}
		best := s.WinningPlay()
		if best != s.paired.winner {
			t.Fatal("the sim's pick is not the winning play")
		}
		alive := 0
		for _, p := range s.simmedPlays.plays {
			if !p.ignore.Load() {
				alive++
			}
			if p.winPctStats.Iterations() < DefaultPairedMinIterations {
				t.Fatalf("%s has %d rollouts, under the minimum", p.play.ShortDescription(), p.winPctStats.Iterations())
			}
		}
		if alive != 1 {
			t.Fatalf("%d plays left", alive)
		}
		if s.PairedRollouts() >= 15*capPerPlay {
			t.Fatalf("%d rollouts: the whole budget", s.PairedRollouts())
		}
	}
}

// The budget, the clock and many threads each end a paired sim with a pick.
func TestPairedSimLimits(t *testing.T) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	defer zerolog.SetGlobalLevel(zerolog.DebugLevel)
	s := pairedTestSimmer(t, 5, 4, 15)
	s.SetPairedCapPerPlay(150)
	if err := s.Simulate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s.PairedStatus() != "sample-limit" || s.paired.winner == nil {
		t.Fatalf("status %q, winner %v", s.PairedStatus(), s.paired.winner)
	}
	// Threads in flight when the budget ran out are not counted.
	if r := s.PairedRollouts(); r < 15*150 || r >= 15*150+15 {
		t.Fatalf("%d rollouts for a budget of %d", r, 15*150)
	}

	s = pairedTestSimmer(t, 5, 4, 15)
	s.SetPairedCapPerPlay(100000000)
	s.SetStoppingCondition(Stop999)
	s.SetPairedMinIterations(100000000)
	s.SetTimeLimit(300 * time.Millisecond)
	start := time.Now()
	if err := s.Simulate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if took := time.Since(start); took < 250*time.Millisecond || took > 5*time.Second {
		t.Fatalf("a 300 ms sim took %v", took)
	}
	if s.paired.winner == nil || s.WinningPlay() != s.paired.winner {
		t.Fatal("a timed-out sim has no pick")
	}
}

// A play the caller asked to keep is never dropped, and does not keep the
// sim from stopping.
func TestPairedSimKeepsPlays(t *testing.T) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	defer zerolog.SetGlobalLevel(zerolog.DebugLevel)
	s := pairedSpacedSimmer(t, 5, 1, 15)
	s.SetPairedCapPerPlay(20000)
	worst := s.simmedPlays.plays[len(s.simmedPlays.plays)-1]
	s.AvoidPruningMoves([]*move.Move{worst.play})
	if err := s.Simulate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s.PairedStatus() != "threshold" {
		t.Fatalf("stopped by %q", s.PairedStatus())
	}
	if worst.ignore.Load() {
		t.Fatal("the kept play was dropped")
	}
	if worst.winPctStats.Iterations() != s.paired.iters {
		t.Fatalf("the kept play has %d rollouts of %d iterations", worst.winPctStats.Iterations(), s.paired.iters)
	}
}
