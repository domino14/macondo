package rangefinder

import (
	"context"
	"testing"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/matryer/is"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/movegen"
)

// seededMidGame builds the same mid-game position every time, from a seeded
// bag, so that inference has a game seed to derive its own randomness from.
func seededMidGame(t *testing.T, turns int) *game.Game {
	t.Helper()
	zerolog.SetGlobalLevel(zerolog.Disabled)
	lex := "NWL18"
	players := []*macondo.PlayerInfo{
		{Nickname: "p1", RealName: "p1"},
		{Nickname: "p2", RealName: "p2"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex,
		board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		t.Fatal(err)
	}
	g, err := game.NewGame(rules, players)
	if err != nil {
		t.Fatal(err)
	}
	var seed [32]byte
	seed[0] = 5
	g.SeedBag(seed)
	g.StartGame()

	gd, err := kwg.GetKWG(g.Config().WGLConfig(), lex)
	if err != nil {
		t.Fatal(err)
	}
	gen := movegen.NewGordonGenerator(gd, g.Board(), rules.LetterDistribution())
	for i := 0; i < turns; i++ {
		plays := gen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
		if len(plays) == 0 {
			break
		}
		if err := g.PlayMove(plays[0], true, 0); err != nil {
			t.Fatal(err)
		}
	}
	return g
}

// inferOnce runs one inference over the position and returns the posterior in a
// comparable form.
func inferOnce(t *testing.T, g *game.Game, budget int) ([]string, []float64, uint64) {
	t.Helper()
	rf := &RangeFinder{}
	rf.Init(g, defaultSimCalculators("NWL18"), DefaultConfig)
	rf.SetThreads(1)
	rf.SetBudget(budget)
	if err := rf.PrepareFinder(g.RackFor(g.PlayerOnTurn()).TilesOn()); err != nil {
		t.Fatal(err)
	}
	// A generous deadline that must not be what stops it: the budget should.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	if err := rf.Infer(ctx); err != nil {
		t.Fatal(err)
	}
	inf := rf.Inferences()
	leaves := make([]string, len(inf.InferredRacks))
	weights := make([]float64, len(inf.InferredRacks))
	for i, ir := range inf.InferredRacks {
		leaves[i] = leaveKey(ir.Leave)
		weights[i] = ir.Weight
	}
	return leaves, weights, rf.SimCount()
}

// The whole point: the same position must infer the same way twice. Before the
// budget and the seeding, the mini-sims drew from the global source and every
// run produced a different posterior.
func TestInferenceIsReproducible(t *testing.T) {
	is := is.New(t)
	g := seededMidGame(t, 6)

	l1, w1, n1 := inferOnce(t, g, 40)
	l2, w2, n2 := inferOnce(t, g, 40)

	is.Equal(n1, n2) // same amount of work
	is.Equal(len(l1), len(l2))
	is.True(len(l1) > 0)
	is.Equal(l1, l2) // same leaves, in the same order
	for i := range w1 {
		// Bit-for-bit: nothing in the path is allowed to vary.
		is.Equal(w1[i], w2[i])
	}
}

// The budget, not the clock, decides how much work happens.
func TestBudgetBoundsTheWork(t *testing.T) {
	is := is.New(t)
	g := seededMidGame(t, 6)

	_, _, small := inferOnce(t, g, 20)
	_, _, large := inferOnce(t, g, 60)

	is.True(small <= 20)
	is.True(large > small)
	is.True(large <= 60)
}

// An unseeded game -- live play -- keeps working, on the global source.
func TestUnseededGameStillInfers(t *testing.T) {
	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.Disabled)
	lex := "NWL18"
	players := []*macondo.PlayerInfo{
		{Nickname: "p1", RealName: "p1"},
		{Nickname: "p2", RealName: "p2"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex,
		board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	g, err := game.NewGame(rules, players)
	is.NoErr(err)
	g.StartGame()

	gd, err := kwg.GetKWG(g.Config().WGLConfig(), lex)
	is.NoErr(err)
	gen := movegen.NewGordonGenerator(gd, g.Board(), rules.LetterDistribution())
	plays := gen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
	is.NoErr(g.PlayMove(plays[0], true, 0))

	rf := &RangeFinder{}
	rf.Init(g, defaultSimCalculators(lex), DefaultConfig)
	rf.SetThreads(1)
	rf.SetBudget(10)
	is.NoErr(rf.PrepareFinder(g.RackFor(g.PlayerOnTurn()).TilesOn()))
	is.Equal(rf.seed, [32]byte{}) // nothing to derive from, and that is fine

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	is.NoErr(rf.Infer(ctx))
	is.True(len(rf.Inferences().InferredRacks) > 0)
}
