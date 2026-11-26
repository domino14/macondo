package crosscheck

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/wmp"
)

var (
	positionsFlag = flag.String("crosscheck.positions", "",
		"file of positions (one CGP per line): written by TestWritePositions, timed by BenchmarkStatic")
	positionsLexicon = flag.String("crosscheck.lexicon", "NWL23", "lexicon of the positions TestWritePositions writes")
	positionsGames   = flag.Int("crosscheck.posgames", 100, "games TestWritePositions plays")
	benchWMP         = flag.Bool("crosscheck.wmp", false, "BenchmarkStatic uses the lexicon's word map")
)

// TestWritePositions writes every position of games the Go static player
// plays against itself, for timing the best-move search in both engines
// on the same positions (BenchmarkStatic here, `zig build bench`).
func TestWritePositions(t *testing.T) {
	if *positionsFlag == "" {
		t.Skip("no -crosscheck.positions file")
	}
	cfg := config.DefaultConfig()
	rules, err := game.NewBasicGameRules(cfg, *positionsLexicon, board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(*positionsFlag)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	players := []*pb.PlayerInfo{{Nickname: "a", RealName: "a"}, {Nickname: "b", RealName: "b"}}
	n := 0
	for gi := 0; gi < *positionsGames; gi++ {
		g, err := game.NewGame(rules, players)
		if err != nil {
			t.Fatal(err)
		}
		var seed [32]byte
		copy(seed[:], fmt.Sprintf("positions-%s-%d", *positionsLexicon, gi))
		g.SeedBag(seed)
		g.StartGame()
		p := staticPlayer(t, cfg, g)
		for g.Playing() == pb.PlayState_PLAYING {
			fmt.Fprintln(w, g.ToCGP(false))
			n++
			m := turnplayer.GenBestStaticTurn(g, p, g.PlayerOnTurn())
			chosen := new(move.Move)
			chosen.CopyFrom(m)
			if err := g.PlayMove(chosen, false, 0); err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Logf("%d positions written to %s", n, *positionsFlag)
}

// BenchmarkStatic times GenBestStaticTurn on every position of the
// -crosscheck.positions file. Set-up (parsing, board recalculation) is
// outside the timing. The equity sum is the same as `zig build bench`'s.
func BenchmarkStatic(b *testing.B) {
	if *positionsFlag == "" {
		b.Skip("no -crosscheck.positions file")
	}
	cfg := config.DefaultConfig()
	text, err := os.ReadFile(*positionsFlag)
	if err != nil {
		b.Fatal(err)
	}
	type pos struct {
		g *game.Game
		p *turnplayer.AIStaticTurnPlayer
	}
	var positions []pos
	for _, line := range strings.Split(strings.TrimSpace(string(text)), "\n") {
		g := parseCGP(b, cfg, line)
		p := staticPlayer(b, cfg, g)
		if *benchWMP {
			w, err := wmp.EnsureWMP(cfg.WGLConfig(), g.LexiconName())
			if err != nil {
				b.Fatal(err)
			}
			p.MoveGenerator().(*movegen.GordonGenerator).SetWMP(w)
		}
		positions = append(positions, pos{g, p})
	}
	sum := 0.0
	for _, ps := range positions {
		sum += turnplayer.GenBestStaticTurn(ps.g, ps.p, ps.g.PlayerOnTurn()).Equity()
	}
	b.ResetTimer()
	for b.Loop() {
		for _, ps := range positions {
			turnplayer.GenBestStaticTurn(ps.g, ps.p, ps.g.PlayerOnTurn())
		}
	}
	perPos := float64(b.Elapsed().Nanoseconds()) / float64(b.N*len(positions))
	b.ReportMetric(perPos/1000, "us/position")
	b.ReportMetric(1e9/perPos, "positions/s")
	b.Logf("%d positions, equity sum %.6f", len(positions), sum)
}

// BenchmarkSim times Go's single-threaded sim on every position of the
// -crosscheck.positions file: the 15 best plays by static equity, two
// plies, -crosscheck.simiters iterations, with the lexicon's word map. The
// UCGI equivalent is `go sim plies 2 iterations <n> plays 15` per
// position.
func BenchmarkSim(b *testing.B) {
	if *positionsFlag == "" {
		b.Skip("no -crosscheck.positions file")
	}
	cfg := config.DefaultConfig()
	text, err := os.ReadFile(*positionsFlag)
	if err != nil {
		b.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(text)), "\n")
	b.ResetTimer()
	for b.Loop() {
		for _, line := range lines {
			b.StopTimer()
			g := parseCGP(b, cfg, line)
			g.SeedBag(automatic.DeriveSeed(1, 0))
			p := staticPlayer(b, cfg, g)
			plays := p.GenerateMoves(15)
			calc, err := equity.NewCombinedStaticCalculator(g.LexiconName(), cfg, "", "")
			if err != nil {
				b.Fatal(err)
			}
			s := &montecarlo.Simmer{}
			s.Init(g, []equity.EquityCalculator{calc}, calc, cfg)
			s.SetThreads(1)
			s.TryLoadWMP(cfg.WGLConfig(), g.LexiconName())
			if err := s.PrepareSim(2, plays); err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
			s.SimSingleThread(*simIterations, 2)
		}
	}
	rollouts := float64(b.N * len(lines) * *simIterations * 15)
	b.ReportMetric(float64(b.Elapsed().Microseconds())/rollouts, "us/rollout")
}

var benchEndgamePlies = flag.Int("crosscheck.egplies", 4, "plies BenchmarkEndgame solves to")

// TestWriteEndgames writes the first endgame position (bag empty) of games
// the Go static player plays against itself, -crosscheck.posgames of them,
// for timing the endgame solver in each engine on the same positions
// (BenchmarkEndgame here, BENCH_MODE=endgame for `zig build bench`).
func TestWriteEndgames(t *testing.T) {
	if *positionsFlag == "" {
		t.Skip("no -crosscheck.positions file")
	}
	cfg := config.DefaultConfig()
	rules, err := game.NewBasicGameRules(cfg, *positionsLexicon, board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(*positionsFlag)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	players := []*pb.PlayerInfo{{Nickname: "a", RealName: "a"}, {Nickname: "b", RealName: "b"}}
	n := 0
	for gi := 0; gi < *positionsGames; gi++ {
		g, err := game.NewGame(rules, players)
		if err != nil {
			t.Fatal(err)
		}
		var seed [32]byte
		copy(seed[:], fmt.Sprintf("endgames-%s-%d", *positionsLexicon, gi))
		g.SeedBag(seed)
		g.StartGame()
		p := staticPlayer(t, cfg, g)
		for g.Playing() == pb.PlayState_PLAYING {
			if g.Bag().TilesRemaining() == 0 {
				fmt.Fprintln(w, g.ToCGP(false))
				n++
				break
			}
			m := turnplayer.GenBestStaticTurn(g, p, g.PlayerOnTurn())
			chosen := new(move.Move)
			chosen.CopyFrom(m)
			if err := g.PlayMove(chosen, false, 0); err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Logf("%d endgames written to %s", n, *positionsFlag)
}

// BenchmarkEndgame solves every endgame of the -crosscheck.positions file
// to -crosscheck.egplies plies on one thread, with the solver's defaults
// and a table of 2^24 entries, as `zig build bench` with BENCH_MODE=endgame
// does. Set-up is outside the timing.
func BenchmarkEndgame(b *testing.B) {
	if *positionsFlag == "" {
		b.Skip("no -crosscheck.positions file")
	}
	cfg := config.DefaultConfig()
	text, err := os.ReadFile(*positionsFlag)
	if err != nil {
		b.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(text)), "\n")
	var nodes uint64
	valueSum := 0
	b.ResetTimer()
	for b.Loop() {
		nodes, valueSum = 0, 0
		for _, line := range lines {
			b.StopTimer()
			s, _ := goEndgameSolver(b, cfg, line)
			b.StartTimer()
			v, _, err := s.Solve(context.Background(), *benchEndgamePlies)
			if err != nil {
				b.Fatal(err)
			}
			nodes += s.SearchedNodes()
			valueSum += int(v)
		}
	}
	perPos := float64(b.Elapsed().Nanoseconds()) / float64(b.N*len(lines))
	b.ReportMetric(perPos/1e6, "ms/position")
	b.ReportMetric(float64(nodes)*1e9/(perPos*float64(len(lines))), "nodes/s")
	b.Logf("%d endgames at %d plies, %d nodes, value sum %d", len(lines), *benchEndgamePlies, nodes, valueSum)
}
