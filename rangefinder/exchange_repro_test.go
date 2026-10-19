package rangefinder

import (
	"context"
	"testing"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// Inference after an exchange redraws racks from the bag, which makes it the
// one path that depends on the bag's tile order. That order comes from MakeBag
// and is shuffled off the global source, so it has to be re-established from
// the seed or this path alone stays unrepeatable -- which is exactly what a
// paired run then trips over, on the turn after any exchange.
func TestExchangeInferenceIsReproducible(t *testing.T) {
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
	seed[0] = 11
	g.SeedBag(seed)
	g.StartGame()

	gd, err := kwg.GetKWG(g.Config().WGLConfig(), lex)
	if err != nil {
		t.Fatal(err)
	}
	gen := movegen.NewGordonGenerator(gd, g.Board(), rules.LetterDistribution())
	// One normal move, then make the opponent exchange, so inference has an
	// exchange to reason about.
	plays := gen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
	if err := g.PlayMove(plays[0], true, 0); err != nil {
		t.Fatal(err)
	}
	rack := g.RackFor(g.PlayerOnTurn())
	exch := move.NewExchangeMove(rack.TilesOn()[:3], rack.TilesOn()[3:], g.Alphabet())
	if err := g.PlayMove(exch, true, 0); err != nil {
		t.Fatalf("exchange failed: %v", err)
	}
	t.Logf("opponent exchanged; now inferring as player %d", g.PlayerOnTurn())

	run := func() (int, []float64) {
		rf := &RangeFinder{}
		rf.Init(g, defaultSimCalculators(lex), DefaultConfig)
		rf.SetThreads(1)
		rf.SetBudget(50)
		if err := rf.PrepareFinder(g.RackFor(g.PlayerOnTurn()).TilesOn()); err != nil {
			t.Fatalf("prepare: %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		if err := rf.Infer(ctx); err != nil {
			t.Fatal(err)
		}
		inf := rf.Inferences()
		w := make([]float64, len(inf.InferredRacks))
		for i, ir := range inf.InferredRacks {
			w[i] = ir.Weight
		}
		return len(inf.InferredRacks), w
	}

	n1, w1 := run()
	n2, w2 := run()
	if n1 == 0 {
		t.Skip("position produced no exchange inferences")
	}
	if n1 != n2 {
		t.Fatalf("inferred rack count differs between runs: %d vs %d", n1, n2)
	}
	for i := range w1 {
		if w1[i] != w2[i] {
			t.Fatalf("weight %d differs between runs: %v vs %v", i, w1[i], w2[i])
		}
	}
}
