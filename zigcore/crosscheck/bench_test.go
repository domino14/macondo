package crosscheck

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

var (
	positionsFlag = flag.String("crosscheck.positions", "",
		"file of positions (one CGP per line): written by TestWritePositions, timed by BenchmarkStatic")
	positionsLexicon = flag.String("crosscheck.lexicon", "NWL23", "lexicon of the positions TestWritePositions writes")
	positionsGames   = flag.Int("crosscheck.posgames", 100, "games TestWritePositions plays")
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
		positions = append(positions, pos{g, staticPlayer(b, cfg, g)})
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
