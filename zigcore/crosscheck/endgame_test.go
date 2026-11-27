package crosscheck

import (
	"context"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/domino14/word-golib/kwg"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

var (
	endgamesFlag = flag.Int("crosscheck.endgames", 6,
		"endgames per lexicon in the endgame cross-check")
	endgamePlies = flag.Int("crosscheck.endgameplies", 4,
		"the deepest search in the endgame cross-check (each endgame gets 1 to this many plies)")
)

// TestEndgameCrossCheck solves endgames in both engines on one thread, with
// the same position keys (Go's, handed to the Zig engine) and table size,
// and compares the value, the depth reached, the number of positions
// searched and the best move.
func TestEndgameCrossCheck(t *testing.T) {
	t.Parallel()
	e := startEngine(t)
	e.Timeout = 5 * time.Minute
	cfg := config.DefaultConfig()
	dataPath := cfg.GetString(config.ConfigDataPath)
	if dataPath == "" {
		t.Skip("no data path; set MACONDO_DATA_PATH")
	}
	if err := e.Send("setoption name data-path value " + dataPath); err != nil {
		t.Fatal(err)
	}
	// 2^24 entries in both: Go's least, and the Zig engine's from 256 MB.
	if err := e.Send("setoption name hash-mb value 256"); err != nil {
		t.Fatal(err)
	}
	for _, bc := range boardCases {
		if bc.peg {
			continue
		}
		t.Run(bc.lexicon+"-"+bc.layout, func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dataPath, "lexica", "gaddag", bc.lexicon+".kwg")); err != nil {
				t.Skipf("no %s word graph", bc.lexicon)
			}
			for i := 0; i < *endgamesFlag; i++ {
				cgpStr := endgamePosition(t, cfg, bc, uint64(i))
				plies := 1 + i%*endgamePlies
				for _, v := range endgameVariants {
					compareEndgame(t, e, cfg, cgpStr, plies, v)
				}
				if t.Failed() {
					return
				}
			}
			t.Logf("%d endgames match", *endgamesFlag)
		})
	}
}

// endgamePosition plays a game with the Go engine (half the moves the top
// scorer, half any play, as the position cross-check does) to the first
// position with the bag empty and both players holding tiles.
func endgamePosition(t *testing.T, cfg *config.Config, bc boardCase, seed uint64) string {
	t.Helper()
	for attempt := uint64(0); ; attempt++ {
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
		copy(bagSeed[:], fmt.Sprintf("endgame-%s-%d-%d", bc.lexicon, seed, attempt))
		g.SeedBag(bagSeed)
		g.StartGame()
		rng := rand.New(rand.NewPCG(seed, attempt))
		gd, err := kwg.GetKWG(cfg.WGLConfig(), bc.lexicon, kwg.WithDistribution(bc.dist))
		if err != nil {
			t.Fatal(err)
		}
		gen := movegen.NewGordonGenerator(gd, g.Board(), rules.LetterDistribution())
		for g.Playing() == pb.PlayState_PLAYING {
			if g.Bag().TilesRemaining() == 0 {
				s := g.ToCGP(false)
				if bc.layout != board.CrosswordGameLayout {
					s += " bdn " + bc.layout + ";"
				}
				return s
			}
			var tilePlays []*move.Move
			for _, m := range gen.GenAll(g.RackFor(g.PlayerOnTurn()), false) {
				if m.Action() == move.MoveTypePlay {
					tilePlays = append(tilePlays, m)
				}
			}
			if len(tilePlays) == 0 {
				break
			}
			pick := tilePlays[rng.IntN(len(tilePlays))]
			if rng.IntN(2) == 0 {
				for _, m := range tilePlays {
					if m.Score() > pick.Score() {
						pick = m
					}
				}
			}
			chosen := new(move.Move)
			chosen.CopyFrom(pick)
			if err := g.PlayMove(chosen, false, 0); err != nil {
				t.Fatal(err)
			}
		}
	}
}

// goEndgameSolver sets up Go's solver as the cross-check compares it: one
// thread and a table of 2^24 entries. Its other options are the defaults;
// the cache of playout tails, which the Zig engine lacks, changes neither
// values nor the count of positions searched.
func goEndgameSolver(t testing.TB, cfg *config.Config, cgpStr string) (*negamax.Solver, *negamax.TranspositionTable) {
	t.Helper()
	g := parseCGP(t, cfg, cgpStr)
	g.SetBackupMode(game.SimulationMode)
	gd, err := kwg.GetKWG(cfg.WGLConfig(), g.LexiconName(), kwg.WithDistribution(g.Rules().LetterDistributionName()))
	if err != nil {
		t.Fatal(err)
	}
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Rules().LetterDistribution())
	s := new(negamax.Solver)
	if err := s.Init(gen, g); err != nil {
		t.Fatal(err)
	}
	s.SetThreads(1)
	tt := &negamax.TranspositionTable{}
	tt.Reset(1e-12, g.Board().Dim())
	s.SetTTable(tt)
	s.SetTTableFraction(1e-12)
	return s, tt
}

// An endgame configuration to compare: the Go solver's settings and the
// Zig engine's go endgame options. With nodes set the searches must be the
// same node for node; otherwise (the Zig engine's defaults, which take
// MAGPIE's aspiration windows and choice of going-out replies) only the
// value and the depth reached.
type endgameVariant struct {
	name   string
	goSet  func(*negamax.Solver)
	zigOpt string
	nodes  bool
	// selfCheck runs only the engine, which checks its own shortcuts
	// against full generations as it goes (checklists) and fails on a
	// difference: for its defaults, whose values at a depth short of the
	// end of the game are MAGPIE's rather than Go's.
	selfCheck bool
	// threads is the Zig engine's thread count (0: one).
	threads int
}

// goGame turns off what Go's solver lacks: the forced-pass bypass, which
// changes the values of searches short of the end of the game, and the
// sure bounds that let the Zig engine prove a value, and so stop, sooner.
const goGame = "noforcedpass nosureout noprecisepure "

var endgameVariants = []endgameVariant{
	{"go-ordering", func(*negamax.Solver) {}, goGame + "goorder noaspiration goplayout gooutreply nofirstreply nocap", true, false, 0},
	{"magpie-ordering", func(s *negamax.Solver) { s.SetMagpieOrdering(true) }, goGame + "magpieorder noaspiration goplayout gooutreply nofirstreply nolazyorder nocap", true, false, 0},
	// The engine's defaults but the playout's pick, which changes the
	// values of searches that stop short of the end of the game.
	{"zig-defaults", func(*negamax.Solver) {}, goGame + "goplayout", false, false, 0},
	{"zig-defaults-selfcheck", nil, "checklists", false, true, 0},
	// The parallel searches, which must find the values one thread does.
	{"zig-defaults-pvsplit", func(*negamax.Solver) {}, goGame + "goplayout pvsplit", false, false, 4},
	{"zig-defaults-abdada", func(*negamax.Solver) {}, goGame + "goplayout abdada", false, false, 4},
}

func compareEndgame(t *testing.T, e *Engine, cfg *config.Config, cgpStr string, plies int, v endgameVariant) {
	t.Helper()
	if v.threads > 1 {
		if err := e.Send(fmt.Sprintf("setoption name threads value %d", v.threads)); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := e.Send("setoption name threads value 1"); err != nil {
				t.Fatal(err)
			}
		}()
	}
	if v.selfCheck {
		if err := e.Send("position cgp " + cgpStr); err != nil {
			t.Fatal(err)
		}
		if _, err := e.Command(fmt.Sprintf("go endgame plies %d %s", plies, v.zigOpt), "bestmove"); err != nil {
			t.Fatalf("%s, %s: %v", cgpStr, v.name, err)
		}
		return
	}
	s, tt := goEndgameSolver(t, cfg, cgpStr)
	v.goSet(s)
	keys := filepath.Join(t.TempDir(), "zobrist.json")
	f, err := os.Create(keys)
	if err != nil {
		t.Fatal(err)
	}
	tt.Zobrist().Dump(f)
	f.Close()

	val, seq, err := s.Solve(context.Background(), plies)
	if err != nil {
		t.Fatalf("%s: Go: %v", cgpStr, err)
	}
	want := fmt.Sprintf("value %d nodes %d", val, s.SearchedNodes())
	wantBest := "pass"
	if len(seq) > 0 {
		wantBest = ucgiMove(seq[0], s.Game().Alphabet())
	}

	if err := e.Send("setoption name zobrist-file value " + keys); err != nil {
		t.Fatal(err)
	}
	if err := e.Send("position cgp " + cgpStr); err != nil {
		t.Fatal(err)
	}
	lines, err := e.Command(fmt.Sprintf("go endgame plies %d %s", plies, v.zigOpt), "bestmove")
	if err != nil {
		t.Fatalf("%s: %v", cgpStr, err)
	}
	if len(lines) < 2 {
		t.Fatalf("%s: Zig said %q", cgpStr, lines)
	}
	f2 := strings.Fields(lines[len(lines)-2])
	if len(f2) < 9 || f2[0] != "info" || f2[1] != "endgame" {
		t.Fatalf("%s: bad info line %q", cgpStr, lines[len(lines)-2])
	}
	got := fmt.Sprintf("value %s nodes %s", f2[5], f2[7])
	gotBest := strings.TrimPrefix(lines[len(lines)-1], "bestmove ")
	if !v.nodes {
		// Only the value: equal values can come with other best moves.
		want = fmt.Sprintf("value %d", val)
		got = "value " + f2[5]
		wantBest, gotBest = "", ""
	}
	if got != want || gotBest != wantBest {
		t.Errorf("%s, %d plies, %s: Go %s best %s (%s), Zig %s best %s\n  %s", cgpStr, plies, v.name, want, wantBest,
			s.ShortDetails(), got, gotBest, strings.Join(lines, "\n  "))
	}
	if d, _ := strconv.Atoi(f2[3]); d != plies {
		t.Errorf("%s: Zig reached depth %d of %d", cgpStr, d, plies)
	}
}
