package montecarlo

import (
	"bytes"
	"context"
	"runtime"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/movegen"
)

// seededSimmer builds the same position every time: a seeded bag means the deal
// and every rack the sim draws come out of one reproducible stream.
func seededSimmer(t *testing.T, seedByte byte, plies, threads int) *Simmer {
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

	gd, err := kwg.GetKWG(g.Config().WGLConfig(), g.LexiconName())
	is.NoErr(err)
	gen := movegen.NewGordonGenerator(gd, g.Board(), rules.LetterDistribution())
	gen.GenAll(g.RackFor(0), false)

	calcs, leaves := defaultSimCalculators("NWL18")
	s := &Simmer{}
	s.Init(g, calcs, leaves.(*equity.CombinedStaticCalculator), DefaultConfig)
	s.SetThreads(threads)
	is.NoErr(s.PrepareSim(plies, gen.Plays()[:10]))
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

// The loop leaves on the very iteration the stopping condition trips, so the
// count always lands on a check boundary. The threaded loop cannot promise this:
// its workers only find out through the context.
func TestInlineSimStopsOnACheckBoundary(t *testing.T) {
	is := is.New(t)

	s := seededSimmer(t, 7, 2, 1)
	s.SetStoppingCondition(Stop99)
	is.NoErr(s.Simulate(context.Background()))

	interval := s.autostopper.stopConditionCheckInterval
	is.True(interval > 0)
	is.Equal(uint64(s.Iterations())%interval, uint64(0))
}

// "As few threads as possible" means none: the inline sim must not spawn a
// controller or a log writer, even with the log stream turned on.
func TestInlineSimSpawnsNoGoroutines(t *testing.T) {
	is := is.New(t)

	s := seededSimmer(t, 9, 2, 1)
	var buf bytes.Buffer
	s.SetLogStream(&buf)

	before := runtime.NumGoroutine()
	s.SimSingleThread(20, 2)
	after := runtime.NumGoroutine()

	is.Equal(after, before)
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
	// One iteration runs before the context is consulted; the point is that it
	// stops immediately rather than simming forever.
	is.Equal(s.Iterations(), 1)
}
