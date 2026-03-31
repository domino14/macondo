package movegen_test

import (
	"testing"

	"github.com/domino14/word-golib/kwg"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/movegen"
)

func setupBenchGame(b *testing.B) (*game.Game, *kwg.KWG) {
	b.Helper()
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL23",
		board.CrosswordGameLayout, "English",
		game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		b.Fatal(err)
	}
	playerInfos := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "Player1"},
		{Nickname: "p2", RealName: "Player2"},
	}
	g, err := game.NewGame(rules, playerInfos)
	if err != nil {
		b.Fatal(err)
	}
	seed := [32]byte{42}
	g.SeedBag(seed)
	g.StartGame()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL23")
	if err != nil {
		b.Fatal(err)
	}

	// Play a few turns to get an interesting board
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
	for i := 0; i < 8 && g.Playing() == pb.PlayState_PLAYING; i++ {
		rack := g.RackFor(g.PlayerOnTurn())
		plays := gen.GenAll(rack, false)
		if len(plays) > 0 {
			g.PlayMove(plays[0], false, 0)
		}
	}
	return g, gd
}

func BenchmarkGenAllNoShadow(b *testing.B) {
	g, gd := setupBenchGame(b)
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
	gen.SetPlayRecorder(movegen.AllPlaysRecorder)
	rack := g.RackFor(g.PlayerOnTurn())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.GenAll(rack, false)
	}
}

func BenchmarkTopPlayNoShadow(b *testing.B) {
	g, gd := setupBenchGame(b)
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
	gen.SetPlayRecorder(movegen.TopPlayOnlyRecorder)
	gen.SetGame(g)
	rack := g.RackFor(g.PlayerOnTurn())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.GenAll(rack, false)
	}
}

func BenchmarkTopPlayShadow(b *testing.B) {
	g, gd := setupBenchGame(b)
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
	gen.SetShadowEnabled(true)
	gen.SetPlayRecorder(movegen.TopPlayOnlyRecorder)
	gen.SetGame(g)
	rack := g.RackFor(g.PlayerOnTurn())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.GenAll(rack, false)
	}
}

func BenchmarkGenAllShadow(b *testing.B) {
	g, gd := setupBenchGame(b)
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
	gen.SetShadowEnabled(true)
	gen.SetPlayRecorder(movegen.AllPlaysRecorder)
	rack := g.RackFor(g.PlayerOnTurn())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.GenAll(rack, false)
	}
}
