package montecarlo

import (
	"bytes"
	"context"
	"runtime"
	"strings"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/move"
)

// placementsByScore is every tile placement for the player on turn, highest
// scoring first.
func placementsByScore(t testing.TB, g *game.Game) []*move.Move {
	t.Helper()
	moves, err := g.MagpieMoves(100000, magpie.SortScore)
	if err != nil {
		t.Fatal(err)
	}
	var plays []*move.Move
	for _, m := range moves {
		if m.Action() == move.MoveTypePlay {
			plays = append(plays, m)
		}
	}
	return plays
}

// seededSimmer builds the same position every time: a seeded bag means the deal
// and every rack the sim draws come out of one reproducible stream.
func seededSimmer(t *testing.T, seedByte byte, plies, threads int) *Simmer {
	return seededSimmerWith(t, seedByte, plies, threads, func(plays []*move.Move) []*move.Move {
		return plays[:10]
	})
}

// seededSimmerWith is seededSimmer simming the plays pick chooses from the
// position's plays, best-scoring first.
func seededSimmerWith(t *testing.T, seedByte byte, plies, threads int,
	pick func([]*move.Move) []*move.Move) *Simmer {
	t.Helper()
	is := is.New(t)

	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18",
		board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	g, err := game.NewGame(rules, players)
	is.NoErr(err)

	var seed [32]byte
	seed[0] = seedByte
	g.SeedBag(seed)
	g.StartGame()
	g.SetPlayerOnTurn(0)
	g.SetRackFor(0, tilemapping.RackFromString("AAAENSW", g.Alphabet()))

	plays := placementsByScore(t, g)

	calcs, leaves := defaultSimCalculators("NWL18")
	s := &Simmer{}
	s.Init(g, calcs, leaves.(*equity.CombinedStaticCalculator), DefaultConfig)
	s.SetThreads(threads)
	is.NoErr(s.PrepareSim(plies, pick(plays)))
	return s
}

// The single-goroutine sim has nothing racing it, so the same seed must give
// the same answer down to the iteration it stopped on. This is what makes
// paired autoplay games comparable.
func TestInlineSimIsReproducible(t *testing.T) {
	is := is.New(t)

	run := func() (int, string, float64) {
		s := seededSimmer(t, 42, 2, 1)
		s.SetStoppingCondition(Stop99)
		is.NoErr(s.Simulate(context.Background()))
		top := s.WinningPlay()
		return s.Iterations(),
			top.play.ShortDescription(),
			top.winPctStats.Mean()
	}

	iters1, play1, win1 := run()
	iters2, play2, win2 := run()

	is.Equal(iters1, iters2)
	is.Equal(play1, play2)
	is.Equal(win1, win2)
	is.True(iters1 > 0)
}

// A clear-cut position stops on the stopping rule, well short of the cap.
func TestInlineSimStopsOnTheThreshold(t *testing.T) {
	is := is.New(t)

	// The best-scoring play against the five worst.
	s := seededSimmerWith(t, 7, 2, 1, func(plays []*move.Move) []*move.Move {
		return append([]*move.Move{plays[0]}, plays[len(plays)-5:]...)
	})
	s.SetStoppingCondition(Stop99)
	is.NoErr(s.Simulate(context.Background()))

	cap := s.autostopper.iterationsCutoff + 2*s.autostopper.perPlyStopScaling
	t.Logf("stopped after %d iterations (cap %d)", s.Iterations(), cap)
	is.True(s.Iterations() >= DefaultMinSamplesPerPlay)
	is.True(s.Iterations() < cap/2)
	is.Equal(s.WinningPlay().Move(), s.arms[0].Move())
}

// liveGoroutines returns the id of every goroutine that exists right now.
func liveGoroutines() map[string]bool {
	buf := make([]byte, 1<<20)
	n := runtime.Stack(buf, true)
	ids := map[string]bool{}
	for _, line := range strings.Split(string(buf[:n]), "\n") {
		if strings.HasPrefix(line, "goroutine ") {
			ids[strings.Fields(line)[1]] = true
		}
	}
	return ids
}

// "As few threads as possible" means none: the inline sim must not spawn a
// controller or a log writer, even with the log stream turned on. The check
// is by identity, not by count: a goroutine left over from an earlier test
// can finish while the sim runs, and that must not read as anything.
func TestInlineSimSpawnsNoGoroutines(t *testing.T) {
	is := is.New(t)

	s := seededSimmer(t, 9, 2, 1)
	var buf bytes.Buffer
	s.SetLogStream(&buf)

	before := liveGoroutines()
	s.SimSingleThread(20, 2)
	for id := range liveGoroutines() {
		if !before[id] {
			t.Fatalf("goroutine %s was started by the inline sim", id)
		}
	}
	// The log still gets written -- by the sim itself, since it is the only
	// writer there is.
	is.True(buf.Len() > 0)
}

// The iteration counter reports what actually ran, and keeps counting when a
// caller sims the same prepared position again (which is how the wasm analyzer
// drives it).
func TestSimSingleThreadCountsWhatItRan(t *testing.T) {
	is := is.New(t)

	s := seededSimmer(t, 11, 2, 1)
	s.SetStoppingCondition(StopNone)

	s.SimSingleThread(50, 2)
	is.Equal(s.Iterations(), 50)

	s.SimSingleThread(30, 2)
	is.Equal(s.Iterations(), 80)
}

// A cancelled context stops the sim without it counting as a failure, and the
// stats collected up to that point survive.
func TestInlineSimHonorsContext(t *testing.T) {
	is := is.New(t)

	s := seededSimmer(t, 13, 2, 1)
	s.SetStoppingCondition(StopNone)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	is.NoErr(s.Simulate(ctx))
	// The context is consulted before every round, so nothing runs.
	is.Equal(s.Iterations(), 0)
}
