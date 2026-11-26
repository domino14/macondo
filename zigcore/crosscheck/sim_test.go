package crosscheck

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
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
)

var (
	simPositions  = flag.Int("crosscheck.sims", 4, "positions per case in the sim cross-check")
	simIterations = flag.Int("crosscheck.simiters", 60, "iterations per sim in the sim cross-check")
)

// simResult is one play's sim statistics as both engines report them.
type simResult struct {
	win, equity float64
	iters       int
}

// TestSimCrossCheck sims positions from seeded static games in both
// engines, single-threaded with the same seed, and compares every play's
// mean win probability and equity exactly: the same seed must deal the
// same opponent racks and draws, and the rollouts must make the same plays.
// Positions keep at least three tiles in the bag, where Go's sim would
// search endgames instead of playing them out (not ported yet).
func TestSimCrossCheck(t *testing.T) {
	e := startEngine(t)
	cfg := config.DefaultConfig()
	dataPath := cfg.GetString(config.ConfigDataPath)
	if dataPath == "" {
		t.Skip("no data path; set MACONDO_DATA_PATH")
	}
	if err := e.Send("setoption name data-path value " + dataPath); err != nil {
		t.Fatal(err)
	}
	const master = 20261126
	for _, bc := range boardCases {
		if bc.peg {
			continue
		}
		t.Run(bc.lexicon+"-"+bc.layout, func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dataPath, "lexica", "gaddag", bc.lexicon+".kwg")); err != nil {
				t.Skipf("no %s word graph", bc.lexicon)
			}
			positions := simPositionsFrom(t, cfg, bc, automatic.DeriveSeed(master, 0), *simPositions)
			for i, cgpStr := range positions {
				plies := 2 + i%3
				seed := uint64(i + 1)
				cmd := fmt.Sprintf("go sim plies %d iterations %d plays 10 seed %d", plies, *simIterations, seed)
				if err := e.Send("position cgp " + cgpStr); err != nil {
					t.Fatal(err)
				}
				lines, err := e.Command(cmd, "bestmove")
				if err != nil {
					t.Fatal(err)
				}
				zig := map[string]simResult{}
				for _, l := range lines[:len(lines)-1] {
					f := strings.Fields(l)
					if len(f) != 12 || f[0] != "info" || f[1] != "sim" {
						t.Fatalf("bad sim line %q", l)
					}
					var r simResult
					r.iters, _ = strconv.Atoi(f[3])
					r.win, _ = strconv.ParseFloat(f[7], 64)
					r.equity, _ = strconv.ParseFloat(f[9], 64)
					zig[f[5]] = r
				}
				want := goSim(t, cfg, cgpStr, automatic.DeriveSeed(seed, 0), plies, *simIterations, 10)
				if len(want) != len(zig) {
					t.Fatalf("%s: Go simmed %d plays, Zig %d", cgpStr, len(want), len(zig))
				}
				for m, w := range want {
					z, ok := zig[m]
					if !ok {
						t.Fatalf("%s: Zig did not sim %s", cgpStr, m)
					}
					if w != z {
						t.Fatalf("%s (%s): %s: Go %+v, Zig %+v", cgpStr, cmd, m, w, z)
					}
				}
			}
			t.Logf("%d sims match", len(positions))
		})
	}
}

// simPositionsFrom plays a seeded static game and returns up to n of its
// positions, as CGPs with the opponent's rack hidden, spread over the turns
// that leave at least three tiles in the bag.
func simPositionsFrom(t *testing.T, cfg *config.Config, bc boardCase, seed [32]byte, n int) []string {
	t.Helper()
	rules, err := game.NewBasicGameRules(cfg, bc.lexicon, bc.layout, bc.dist, game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		t.Fatal(err)
	}
	players := []*pb.PlayerInfo{{Nickname: "p1", RealName: "p1"}, {Nickname: "p2", RealName: "p2"}}
	g, err := game.NewGame(rules, players)
	if err != nil {
		t.Fatal(err)
	}
	g.SeedBag(seed)
	g.StartGame()
	p := staticPlayer(t, cfg, g)
	var all []string
	for g.Playing() == pb.PlayState_PLAYING && g.Bag().TilesRemaining() >= 3 {
		alph := g.Alphabet()
		me := g.PlayerOnTurn()
		all = append(all, fmt.Sprintf("%s %s/ %d/%d %d lex %s; ld %s; bdn %s;",
			g.Board().ToFEN(alph), g.RackFor(me).TilesOn().UserVisible(alph),
			g.PointsFor(me), g.PointsFor(1-me), g.ScorelessTurns(), bc.lexicon, bc.dist, bc.layout))
		m := turnplayer.GenBestStaticTurn(g, p, me)
		chosen := new(move.Move)
		chosen.CopyFrom(m)
		if err := g.PlayMove(chosen, false, 0); err != nil {
			t.Fatal(err)
		}
	}
	if len(all) <= n {
		return all
	}
	out := make([]string, n)
	for i := range out {
		out[i] = all[i*(len(all)-1)/(n-1+boolInt(n == 1))]
	}
	return out
}

// goSim sims the nplays best plays by static equity of a CGP position in
// Go's engine, single-threaded, seeded with seed at turn 0, and returns each
// play's statistics by UCGI move. The bag is sorted first: Go leaves it in
// whatever order building the game left it, the Zig engine in machine
// letter order, and the sim shuffles that order.
func goSim(t *testing.T, cfg *config.Config, cgpStr string, seed [32]byte, plies, iters, nplays int) map[string]simResult {
	t.Helper()
	g := parseCGP(t, cfg, cgpStr)
	slices.Sort(g.Bag().Tiles())
	g.SeedBag(seed)
	p := staticPlayer(t, cfg, g)
	plays := p.GenerateMoves(nplays)
	leavesFile := ""
	if g.Rules().BoardName() == board.SuperCrosswordGameLayout {
		leavesFile = "super-leaves.klv2"
	}
	calc, err := equity.NewCombinedStaticCalculator(g.LexiconName(), cfg, leavesFile, "")
	if err != nil {
		t.Fatal(err)
	}
	s := &montecarlo.Simmer{}
	s.Init(g, []equity.EquityCalculator{calc}, calc, cfg)
	s.SetThreads(1)
	s.SetEndgameLeaves(false)
	if err := s.PrepareSim(plies, plays); err != nil {
		t.Fatal(err)
	}
	s.SimSingleThread(iters, plies)
	out := map[string]simResult{}
	for _, sp := range s.PlaysByWinProb().PlaysNoLock() {
		out[ucgiMove(sp.Move(), g.Alphabet())] = simResult{
			win:    sp.WinProb(),
			equity: sp.EquityMean(),
			iters:  sp.WinProbIterations(),
		}
	}
	return out
}
