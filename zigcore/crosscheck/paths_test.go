package crosscheck

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/domino14/word-golib/kwg"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

var (
	pathGames = flag.Int("crosscheck.pathgames", 3,
		"games per lexicon in the board-paths cross-check (and 10x as many random-tile boards)")
	pathTakebacks = flag.Int("crosscheck.takebacks", 3,
		"moves played and taken back at each position of the board-paths cross-check")
)

// TestBoardPathsCrossCheck compares boards the Zig engine reaches without
// loading them from a CGP against the Go engine's board loaded from one
// (cross-sets worked out from scratch): every square, every move with its
// score and equity, and the best move. Three paths:
//
//   - a game played on in the engine, move after move, with the cross-sets
//     brought up to date incrementally (as games, autoplay and sims do);
//   - a move played on a loaded board and taken back again (several per
//     position, the takeback working the board out afresh);
//   - a position loaded with its last move (CGP lm) and taken back, as
//     inference does.
//
// The games are random-ish (half the moves the top scorer, half any play),
// and random-tile boards add shapes and phonies no game reaches.
func TestBoardPathsCrossCheck(t *testing.T) {
	t.Parallel()
	inc := startEngine(t)
	fresh := startEngine(t)
	cfg := config.DefaultConfig()
	dataPath := cfg.GetString(config.ConfigDataPath)
	if dataPath == "" {
		t.Skip("no data path; set MACONDO_DATA_PATH")
	}
	for _, e := range []*Engine{inc, fresh} {
		if err := e.Send("setoption name data-path value " + dataPath); err != nil {
			t.Fatal(err)
		}
	}
	for _, bc := range boardCases {
		if bc.peg {
			continue
		}
		t.Run(bc.lexicon+"-"+bc.layout, func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dataPath, "lexica", "gaddag", bc.lexicon+".kwg")); err != nil {
				t.Skipf("no %s word graph", bc.lexicon)
			}
			w := &pathWalker{t: t, inc: inc, fresh: fresh, cfg: cfg, bc: bc}
			for gi := 0; gi < *pathGames; gi++ {
				rules, err := game.NewBasicGameRules(cfg, bc.lexicon, bc.layout, bc.dist, game.CrossScoreAndSet, game.VarClassic)
				if err != nil {
					t.Fatal(err)
				}
				players := []*pb.PlayerInfo{{Nickname: "a", RealName: "a"}, {Nickname: "b", RealName: "b"}}
				g, err := game.NewGame(rules, players)
				if err != nil {
					t.Fatal(err)
				}
				var bagSeed [32]byte
				copy(bagSeed[:], fmt.Sprintf("paths-%s-%d", bc.lexicon, gi))
				g.SeedBag(bagSeed)
				g.StartGame()
				w.walk(g, uint64(gi), 1000)
				if t.Failed() {
					return
				}
			}
			for i := 0; i < 10**pathGames; i++ {
				g := parseCGP(t, cfg, randomTilesCGP(t, cfg, bc, uint64(1000+i)))
				var bagSeed [32]byte
				copy(bagSeed[:], fmt.Sprintf("paths-random-%s-%d", bc.lexicon, i))
				g.SeedBag(bagSeed)
				w.walk(g, uint64(1000+i), 4)
				if t.Failed() {
					return
				}
			}
			t.Logf("%d positions on the incremental path, %d plays taken back, %d last moves taken back: all match",
				w.positions, w.takebacks, w.lastMoves)
		})
	}
}

type pathWalker struct {
	t          *testing.T
	inc, fresh *Engine
	cfg        *config.Config
	bc         boardCase

	positions, takebacks, lastMoves int
}

// cgpOf is the Go game's position as a CGP, layout included.
func (w *pathWalker) cgpOf(g *game.Game) string {
	s := g.ToCGP(false)
	if w.bc.layout != board.CrosswordGameLayout && !strings.Contains(s, "bdn ") {
		s += " bdn " + w.bc.layout + ";"
	}
	return s
}

// compareAll compares the engine's position with cgpStr's as the Go engine
// loads it: the board (readSites as in compareBoard), every move and the
// best move.
func (w *pathWalker) compareAll(e *Engine, cgpStr, how string, readSites bool) {
	w.t.Helper()
	lines, err := e.Command("cgp", "cgp")
	if err != nil {
		w.t.Fatal(err)
	}
	if want := strings.Join(strings.Fields(cgpStr)[:4], " "); lines[0] != "cgp "+want {
		w.t.Errorf("%s: Zig's position is %q, Go's %q", how, lines[0], want)
	}
	compareBoard(w.t, e, parseCGP(w.t, w.cfg, cgpStr).Board(), how+": "+cgpStr, readSites)
	compareMoves(w.t, e, w.cfg, cgpStr)
	compareBest(w.t, e, w.cfg, cgpStr)
}

// play plays m in engine e (which must hold the position g is in) and in g,
// with the mover's new rack, and checks the score.
func (w *pathWalker) play(e *Engine, g *game.Game, m *move.Move) {
	w.t.Helper()
	mover := g.PlayerOnTurn()
	text := ucgiMove(m, g.Alphabet())
	played := new(move.Move)
	played.CopyFrom(m)
	if err := g.PlayMove(played, false, 0); err != nil {
		w.t.Fatal(err)
	}
	lines, err := e.Command("play "+text+" rack "+g.RackFor(mover).TilesOn().UserVisible(g.Alphabet()), "playok")
	if err != nil {
		w.t.Fatal(err)
	}
	if want := "playok score " + strconv.Itoa(m.Score()); len(lines) != 1 || lines[0] != want {
		w.t.Fatalf("play %s: Zig said %q, Go scored it %d", text, lines, m.Score())
	}
}

// walk plays g on to its end or for at most plies moves, keeping the
// incremental engine in step, and at each position also tries the
// takeback paths in the fresh engine.
func (w *pathWalker) walk(g *game.Game, seed uint64, plies int) {
	t := w.t
	rng := rand.New(rand.NewPCG(seed, 0x9a7b5))
	gd, err := kwg.GetKWG(w.cfg.WGLConfig(), w.bc.lexicon, kwg.WithDistribution(w.bc.dist))
	if err != nil {
		t.Fatal(err)
	}
	alph := g.Alphabet()
	if err := w.inc.Send("position cgp " + w.cgpOf(g)); err != nil {
		t.Fatal(err)
	}
	for ply := 0; ply < plies && g.Playing() == pb.PlayState_PLAYING; ply++ {
		cgpStr := w.cgpOf(g)
		w.compareAll(w.inc, cgpStr, "played on incrementally", true)
		w.positions++
		if t.Failed() {
			return
		}

		gen := movegen.NewGordonGenerator(gd, g.Board(), g.Rules().LetterDistribution())
		rack := g.RackFor(g.PlayerOnTurn())
		var tilePlays []*move.Move
		for _, m := range gen.GenAll(rack, false) {
			if m.Action() == move.MoveTypePlay {
				c := new(move.Move)
				c.CopyFrom(m)
				tilePlays = append(tilePlays, c)
			}
		}
		if len(tilePlays) == 0 {
			return
		}
		rackText := rack.TilesOn().UserVisible(alph)

		// A few plays made on the loaded board and taken back.
		if err := w.fresh.Send("position cgp " + cgpStr); err != nil {
			t.Fatal(err)
		}
		for range *pathTakebacks {
			m := tilePlays[rng.IntN(len(tilePlays))]
			after := parseCGP(t, w.cfg, cgpStr)
			var bagSeed [32]byte
			copy(bagSeed[:], fmt.Sprintf("takeback-%d-%d", seed, rng.Uint64()))
			after.SeedBag(bagSeed)
			w.play(w.fresh, after, m)
			if after.Playing() == pb.PlayState_PLAYING {
				w.compareAll(w.fresh, w.cgpOf(after), "played on a loaded board", true)
			}
			if _, err := w.fresh.Command("takeback rack "+rackText, "takebackok"); err != nil {
				t.Fatal(err)
			}
			w.compareAll(w.fresh, cgpStr, "played and taken back", false)
			w.takebacks++
			if t.Failed() {
				return
			}
		}

		// The main line: half the time the top scorer, else any play.
		pick := tilePlays[rng.IntN(len(tilePlays))]
		if rng.IntN(2) == 0 {
			for _, m := range tilePlays {
				if m.Score() > pick.Score() {
					pick = m
				}
			}
		}
		lm := lastMoveNotation(pick, g.Board(), alph)
		w.play(w.inc, g, pick)
		if g.Playing() != pb.PlayState_PLAYING {
			return
		}

		// The position after it, loaded with its last move and taken back.
		afterLM := strings.TrimSpace(w.cgpOf(g)) + fmt.Sprintf(" lm %s %d;", lm, pick.Score())
		if err := w.fresh.Send("position cgp " + afterLM); err != nil {
			t.Fatal(err)
		}
		if _, err := w.fresh.Command("takeback rack "+rackText, "takebackok"); err != nil {
			t.Fatalf("%s: %v", afterLM, err)
		}
		w.compareAll(w.fresh, cgpStr, "loaded with lm and taken back", false)
		w.lastMoves++
		if t.Failed() {
			return
		}
	}
}
