package montecarlo

import (
	"context"
	"runtime"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/movegen"
	"github.com/rs/zerolog"
)

const (
	numPositions = 10
	itersPerPos  = 100
	simPlies     = 5
	numSimPlays  = 10
	benchLexicon = "CSW24"
)

type stopCondition int

const (
	stopAfterTurns stopCondition = iota
	stopWhenBagLE
)

type positionSpec struct {
	cond      stopCondition
	threshold int // turns played, or max tiles remaining in bag
}

var (
	earlyGame = positionSpec{stopAfterTurns, 2}
	midGame   = positionSpec{stopAfterTurns, 8}
	lateGame  = positionSpec{stopWhenBagLE, 14}
)

func buildSimmers(b *testing.B, spec positionSpec) []*Simmer {
	b.Helper()
	calcs, leaves := defaultSimCalculators(benchLexicon)

	rules, err := game.NewBasicGameRules(DefaultConfig, benchLexicon,
		board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		b.Fatal(err)
	}
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), benchLexicon)
	if err != nil {
		b.Fatal(err)
	}
	playerInfos := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "Player1"},
		{Nickname: "p2", RealName: "Player2"},
	}

	cgpStrings := make([]string, 0, numPositions)
	for seedIdx := 0; len(cgpStrings) < numPositions; seedIdx++ {
		g, err := game.NewGame(rules, playerInfos)
		if err != nil {
			b.Fatal(err)
		}
		var seed [32]byte
		seed[0] = byte(seedIdx)
		seed[1] = byte(seedIdx >> 8)
		g.SeedBag(seed)
		g.StartGame()

		gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
		for g.Playing() == pb.PlayState_PLAYING {
			done := false
			switch spec.cond {
			case stopAfterTurns:
				done = g.Turn() >= spec.threshold
			case stopWhenBagLE:
				done = g.Bag().TilesRemaining() <= spec.threshold
			}
			if done {
				break
			}
			plays := gen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
			if len(plays) == 0 {
				break
			}
			if err := g.PlayMove(plays[0], false, 0); err != nil {
				b.Fatal(err)
			}
		}
		if g.Playing() != pb.PlayState_PLAYING {
			continue
		}
		cgpStrings = append(cgpStrings, g.ToCGP(false))
	}

	simmers := make([]*Simmer, 0, numPositions)
	for _, cgpStr := range cgpStrings {
		parsed, err := cgp.ParseCGP(DefaultConfig, cgpStr)
		if err != nil {
			b.Fatal(err)
		}
		parsed.RecalculateBoard()

		gen := movegen.NewGordonGenerator(gd, parsed.Board(), parsed.Rules().LetterDistribution())
		plays := gen.GenAll(parsed.RackFor(parsed.PlayerOnTurn()), false)
		if len(plays) == 0 {
			continue
		}
		n := numSimPlays
		if len(plays) < n {
			n = len(plays)
		}
		s := &Simmer{}
		s.Init(parsed.Game, calcs, leaves.(*equity.CombinedStaticCalculator), DefaultConfig)
		s.SetThreads(1)
		if err := s.PrepareSim(simPlies, plays[:n]); err != nil {
			b.Fatal(err)
		}
		simmers = append(simmers, s)
	}

	if len(simmers) == 0 {
		b.Fatal("no valid positions generated")
	}
	return simmers
}

func runSimBench(b *testing.B, simmers []*Simmer) {
	b.Helper()
	ctx := context.Background()
	runtime.MemProfileRate = 1
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for posIdx, s := range simmers {
			for iter := 0; iter < itersPerPos; iter++ {
				iterCount := uint64(i*itersPerPos*len(simmers) + posIdx*itersPerPos + iter + 1)
				s.simSingleIteration(ctx, simPlies, 0, iterCount, nil)
			}
		}
	}
}

// BenchmarkSimEarlyGame runs 1K 5-ply sim iterations across 10 early-game
// positions (2 turns played).
func BenchmarkSimEarlyGame(b *testing.B) {
	zerolog.SetGlobalLevel(zerolog.Disabled)
	runtime.MemProfileRate = 0
	simmers := buildSimmers(b, earlyGame)
	runSimBench(b, simmers)
}

// BenchmarkSimMidGame runs 1K 5-ply sim iterations across 10 mid-game
// positions (8 turns played).
func BenchmarkSimMidGame(b *testing.B) {
	zerolog.SetGlobalLevel(zerolog.Disabled)
	runtime.MemProfileRate = 0
	simmers := buildSimmers(b, midGame)
	runSimBench(b, simmers)
}

// BenchmarkSimLateGame runs 1K 5-ply sim iterations across 10 late-game
// positions (bag has ≤14 tiles remaining).
func BenchmarkSimLateGame(b *testing.B) {
	zerolog.SetGlobalLevel(zerolog.Disabled)
	runtime.MemProfileRate = 0
	simmers := buildSimmers(b, lateGame)
	runSimBench(b, simmers)
}
