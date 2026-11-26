package crosscheck

import (
	"context"
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

// simCase is how a position is simmed: a fixed number of iterations, or
// with stop set, best-arm identification (the elite bot's stop rule) with
// a budget of cap rollouts per play.
type simCase struct {
	plies, iters, stop, cap, bestCap int
	seed                             uint64
	// threads is the Zig engine's: a best-arm sim is the same on any
	// number, so it is compared with Go's single-threaded sim too.
	threads int
	// cycle makes each best-arm sample a bag cycle (the elite bot's
	// setting).
	cycle bool
}

func (c simCase) command() string {
	if c.stop > 0 {
		cmd := fmt.Sprintf("go sim plies %d plays 10 seed %d stop %d cap %d bestcap %d", c.plies, c.seed, c.stop, c.cap, c.bestCap)
		if c.cycle {
			cmd += " cycle 1"
		}
		return cmd
	}
	return fmt.Sprintf("go sim plies %d iterations %d plays 10 seed %d", c.plies, c.iters, c.seed)
}

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
			sims := 0
			for i, cgpStr := range positions {
				plies := 2 + i%3
				for _, c := range []simCase{
					{plies: plies, iters: *simIterations, seed: uint64(i + 1)},
					{plies: plies, stop: 99, cap: 3 * *simIterations, bestCap: 2000, seed: uint64(i + 1)},
					{plies: plies, stop: 99, cap: 3 * *simIterations, bestCap: 2000, seed: uint64(i + 1), threads: 4},
					{plies: plies, stop: 99, cap: 3 * *simIterations, bestCap: 2000, seed: uint64(i + 1), cycle: true},
					{plies: plies, stop: 99, cap: 3 * *simIterations, bestCap: 2000, seed: uint64(i + 1), cycle: true, threads: 4},
				} {
					compareSim(t, e, cfg, cgpStr, c)
					sims++
				}
			}
			t.Logf("%d sims match", sims)
		})
	}
	t.Logf("%d best-arm sims sampled adaptively", adaptiveSims)
	if adaptiveSims == 0 {
		t.Error("no best-arm sim got past its round-robin phase")
	}
}

// compareSim sims a position in both engines and compares every play's
// statistics, and for a best-arm sim the play picked and why it stopped.
func compareSim(t *testing.T, e *Engine, cfg *config.Config, cgpStr string, c simCase) {
	t.Helper()
	cmd := c.command()
	threads := max(c.threads, 1)
	if err := e.Send(fmt.Sprintf("setoption name threads value %d", threads)); err != nil {
		t.Fatal(err)
	}
	if err := e.Send("position cgp " + cgpStr); err != nil {
		t.Fatal(err)
	}
	lines, err := e.Command(cmd, "bestmove")
	if err != nil {
		t.Fatal(err)
	}
	zig := map[string]simResult{}
	zigStatus := ""
	for _, l := range lines[:len(lines)-1] {
		if strings.HasPrefix(l, "info nodes ") {
			// Progress.
			continue
		}
		f := strings.Fields(l)
		if len(f) == 6 && f[1] == "sim" && f[2] == "status" {
			zigStatus = f[3]
			continue
		}
		if len(f) != 12 || f[0] != "info" || f[1] != "sim" {
			t.Fatalf("bad sim line %q", l)
		}
		var r simResult
		r.iters, _ = strconv.Atoi(f[3])
		r.win, _ = strconv.ParseFloat(f[7], 64)
		r.equity, _ = strconv.ParseFloat(f[9], 64)
		zig[f[5]] = r
	}
	zigBest := strings.TrimPrefix(lines[len(lines)-1], "bestmove ")
	want, best, status := goSim(t, cfg, cgpStr, automatic.DeriveSeed(c.seed, 0), c, 10)
	if len(want) != len(zig) {
		t.Fatalf("%s: Go simmed %d plays, Zig %d", cgpStr, len(want), len(zig))
	}
	for m, w := range want {
		z, ok := zig[m]
		if !ok {
			t.Fatalf("%s: Zig did not sim %s", cgpStr, m)
		}
		if w != z {
			t.Fatalf("%s (%s, %d threads): %s: Go %+v, Zig %+v", cgpStr, cmd, threads, m, w, z)
		}
	}
	if c.stop > 0 && (best != zigBest || status != zigStatus) {
		t.Fatalf("%s (%s, %d threads): Go picked %s (%s), Zig %s (%s)", cgpStr, cmd, threads, best, status, zigBest, zigStatus)
	}
	if c.stop > 0 {
		counts := map[int]bool{}
		for _, r := range zig {
			counts[r.iters] = true
		}
		if len(counts) > 1 {
			adaptiveSims++
		}
	}
}

// adaptiveSims counts the best-arm sims that got past the round-robin
// phase, where every play has the same number of rollouts.
var adaptiveSims int

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
func goSim(t *testing.T, cfg *config.Config, cgpStr string, seed [32]byte, c simCase, nplays int) (map[string]simResult, string, string) {
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
	s.SetBAIBagCycle(c.cycle)
	if err := s.PrepareSim(c.plies, plays); err != nil {
		t.Fatal(err)
	}
	if c.stop > 0 {
		if c.stop != 99 {
			t.Fatalf("stop %d not set up", c.stop)
		}
		s.SetStoppingCondition(montecarlo.Stop99)
		s.SetStopRule(montecarlo.StopRuleBAI)
		s.SetStopCapPerPlay(c.cap)
		s.SetBAIBestPlayCap(c.bestCap)
		if err := s.Simulate(context.Background()); err != nil {
			t.Fatal(err)
		}
	} else {
		s.SimSingleThread(c.iters, c.plies)
	}
	out := map[string]simResult{}
	ranked := s.PlaysByWinProb().PlaysNoLock()
	for _, sp := range ranked {
		out[ucgiMove(sp.Move(), g.Alphabet())] = simResult{
			win:    sp.WinProb(),
			equity: sp.EquityMean(),
			iters:  sp.WinProbIterations(),
		}
	}
	return out, ucgiMove(ranked[0].Move(), g.Alphabet()), s.BAIStatus()
}
