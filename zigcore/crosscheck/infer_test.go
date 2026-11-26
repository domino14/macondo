package crosscheck

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/rangefinder"
)

var (
	inferPositions = flag.Int("crosscheck.infers", 4, "positions per case in the inference cross-check")
	inferBudget    = flag.Int("crosscheck.inferbudget", 30, "mini-sims per inference in the inference cross-check")
)

// inferCase is a turn of a static game: the position before the move, the
// mover's rack, and the move.
type inferCase struct {
	before string // CGP, the mover on turn with their rack, the other rack known
	m      *move.Move
}

// TestInferCrossCheck reads the opponent's leave after their move in both
// engines: Go's rangefinder on one thread in budget mode (as the elite bot
// in paired autoplay), and the Zig engine's go infer on the position after
// the move, with the move in the CGP's lm. The posteriors must be the same
// leaves in the same order with the same weights.
func TestInferCrossCheck(t *testing.T) {
	e := startEngine(t)
	// An inference is up to a few hundred mini-sims.
	e.Timeout = 5 * time.Minute
	cfg := config.DefaultConfig()
	dataPath := cfg.GetString(config.ConfigDataPath)
	if dataPath == "" {
		t.Skip("no data path; set MACONDO_DATA_PATH")
	}
	if err := e.Send("setoption name data-path value " + dataPath); err != nil {
		t.Fatal(err)
	}
	const master = 20261127
	kinds := map[string]int{}
	// Time spent per engine, by the Zig engine's thread count (Go reads on
	// one thread throughout).
	var goTime, zigTime [2]time.Duration
	var reads [2]int
	for _, bc := range boardCases {
		if bc.peg {
			continue
		}
		t.Run(bc.lexicon+"-"+bc.layout, func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dataPath, "lexica", "gaddag", bc.lexicon+".kwg")); err != nil {
				t.Skipf("no %s word graph", bc.lexicon)
			}
			cases := inferCasesFrom(t, cfg, bc, automatic.DeriveSeed(master, 1), *inferPositions)
			for i, c := range cases {
				// Every other read on four threads: Zig's reads are the
				// same on any number, and compared with Go's on one.
				threads := 1 + 3*(i%2)
				if err := e.Send(fmt.Sprintf("setoption name threads value %d", threads)); err != nil {
					t.Fatal(err)
				}
				kind, goT, zigT := compareInfer(t, e, cfg, c, uint64(i+1), *inferBudget)
				kinds[kind]++
				goTime[i%2] += goT
				zigTime[i%2] += zigT
				reads[i%2]++
			}
			t.Logf("%d inferences match", len(cases))
		})
	}
	t.Logf("paths: %v", kinds)
	for k, threads := range []int{1, 4} {
		if reads[k] > 0 {
			t.Logf("%d reads, Zig on %d thread(s): Go %v, Zig %v (Go/Zig %.2f)", reads[k], threads,
				goTime[k], zigTime[k], goTime[k].Seconds()/zigTime[k].Seconds())
		}
	}
}

// compareInfer returns the kind of read and the time each engine took
// (Go's from PrepareFinder, Zig's from the go command).
func compareInfer(t *testing.T, e *Engine, cfg *config.Config, c inferCase, seed uint64, budget int) (string, time.Duration, time.Duration) {
	t.Helper()
	g := parseCGP(t, cfg, c.before)
	liveSeed := automatic.DeriveSeed(seed, 0)
	g.SeedBag(liveSeed)
	alph := g.Alphabet()
	mover := g.PlayerOnTurn()
	m := new(move.Move)
	m.CopyFrom(c.m)
	lm := lastMoveNotation(m, g.Board(), alph)
	if err := g.PlayMove(m, true, 0); err != nil {
		t.Fatal(err)
	}
	me := g.PlayerOnTurn()
	// The position after the move as the player on turn sees it.
	after := fmt.Sprintf("%s %s/ %d/%d %d lex %s; ld %s; bdn %s; lm %s %d;",
		g.Board().ToFEN(alph), g.RackFor(me).TilesOn().UserVisible(alph),
		g.PointsFor(me), g.PointsFor(mover), g.ScorelessTurns(),
		g.LexiconName(), g.Rules().LetterDistributionName(), g.Rules().BoardName(), lm, m.Score())

	leaves := ""
	if g.Rules().BoardName() == board.SuperCrosswordGameLayout {
		leaves = "super-leaves.klv2"
	}
	calc, err := equity.NewCombinedStaticCalculator(g.LexiconName(), cfg, leaves, equity.PEGAdjustmentFilename)
	if err != nil {
		t.Fatal(err)
	}
	rf := &rangefinder.RangeFinder{}
	rf.Init(g, []equity.EquityCalculator{calc}, cfg)
	rf.SetThreads(1)
	rf.SetTauCap(0.1)
	rf.SetBudget(budget)
	rf.SetTracing(true)
	goStart := time.Now()
	if err := rf.PrepareFinder(g.RackFor(me).TilesOn()); err != nil {
		t.Fatalf("%s: Go: %v", after, err)
	}
	if err := rf.Infer(context.Background()); err != nil {
		t.Fatalf("%s: Go: %v", after, err)
	}
	goTime := time.Since(goStart)
	want := rf.Inferences().InferredRacks

	if err := e.Send("position cgp " + after); err != nil {
		t.Fatal(err)
	}
	zigStart := time.Now()
	lines, err := e.Command(fmt.Sprintf("go infer budget %d seed %d trace 1", budget, seed), "inferok")
	if err != nil {
		t.Fatalf("%s: %v", after, err)
	}
	zigTime := time.Since(zigStart)
	// The measurements first, in order, so that a difference shows where it
	// starts.
	var zigDraws []string
	for len(lines) > 0 && strings.HasPrefix(lines[0], "info infer draw ") {
		zigDraws = append(zigDraws, lines[0])
		lines = lines[1:]
	}
	// (Go traces the sampled reads, not the enumerated ones.)
	if tr := rf.Trace(); tr != nil && len(tr.Draws) > 0 {
		if len(tr.Draws) != len(zigDraws) {
			t.Errorf("%s: Go measured %d leaves, Zig %d", after, len(tr.Draws), len(zigDraws))
		}
		for i, d := range tr.Draws {
			if i >= len(zigDraws) {
				break
			}
			f := strings.Fields(zigDraws[i])
			lik, _ := strconv.ParseFloat(f[8], 64)
			if f[4] != strconv.Itoa(d.Round) || f[6] != d.Leave || math.Float64bits(lik) != math.Float64bits(d.Measured) {
				t.Fatalf("%s: draw %d: Go round %d %s %v, Zig %s", after, i, d.Round, d.Leave, d.Measured, zigDraws[i])
			}
		}
	}
	if len(lines) < 2 || !strings.HasPrefix(lines[0], "info infer leaves ") {
		t.Fatalf("%s: Zig said %q", after, lines)
	}
	head := strings.Fields(lines[0])
	got := lines[1 : len(lines)-1]
	if len(got) != len(want) {
		t.Fatalf("%s: Go read %d leaves, Zig %d (%s)", after, len(want), len(got), lines[0])
	}
	for i, ir := range want {
		f := strings.Fields(got[i])
		if len(f) != 6 || f[2] != "leave" || f[4] != "weight" {
			t.Fatalf("bad infer line %q", got[i])
		}
		w, _ := strconv.ParseFloat(f[5], 64)
		leave := tilemapping.MachineWord(ir.Leave).UserVisible(alph)
		if f[3] != leave || math.Float64bits(w) != math.Float64bits(ir.Weight) {
			t.Fatalf("%s: leave %d: Go %s %v, Zig %s %v", after, i, leave, ir.Weight, f[3], w)
		}
	}
	switch {
	case m.Action() == move.MoveTypeExchange:
		return "exchange", goTime, zigTime
	case head[13] == "1":
		return "enumerated-kept-" + head[5], goTime, zigTime
	}
	return "sampled-kept-" + head[5], goTime, zigTime
}

// lastMoveNotation writes m in CGP's lm notation, the tiles it plays
// through (0 in the move, read from b, which holds them) in parentheses
// and tiles of more than one character in brackets.
func lastMoveNotation(m *move.Move, b *board.GameBoard, alph *tilemapping.TileMapping) string {
	tile := func(ml tilemapping.MachineLetter) string {
		s := ml.UserVisible(alph, false)
		if utf8.RuneCountInString(s) > 1 && !strings.HasPrefix(s, "[") {
			return "[" + s + "]"
		}
		return s
	}
	var sb strings.Builder
	if m.Action() == move.MoveTypeExchange {
		sb.WriteString("ex.")
		for _, ml := range m.Tiles() {
			sb.WriteString(tile(ml))
		}
		return sb.String()
	}
	sb.WriteString(m.BoardCoords() + ".")
	row, col, vert := m.CoordsAndVertical()
	open := false
	for i, ml := range m.Tiles() {
		if ml == 0 {
			if !open {
				sb.WriteString("(")
				open = true
			}
			r, c := row, col+i
			if vert {
				r, c = row+i, col
			}
			sb.WriteString(tile(b.GetLetter(r, c)))
			continue
		}
		if open {
			sb.WriteString(")")
			open = false
		}
		sb.WriteString(tile(ml))
	}
	if open {
		sb.WriteString(")")
	}
	return sb.String()
}

// inferCasesFrom plays a static game and keeps up to n turns spread over
// it whose move can be read: a play of one to six tiles or an exchange,
// with tiles left in the bag afterwards.
func inferCasesFrom(t *testing.T, cfg *config.Config, bc boardCase, seed [32]byte, n int) []inferCase {
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
	var all, ones []inferCase
	for turn := 0; g.Playing() == pb.PlayState_PLAYING; turn++ {
		alph := g.Alphabet()
		me := g.PlayerOnTurn()
		before := fmt.Sprintf("%s %s/%s %d/%d %d lex %s; ld %s; bdn %s;",
			g.Board().ToFEN(alph), g.RackFor(me).TilesOn().UserVisible(alph),
			g.RackFor(1-me).TilesOn().UserVisible(alph),
			g.PointsFor(me), g.PointsFor(1-me), g.ScorelessTurns(), bc.lexicon, bc.dist, bc.layout)
		if (turn == 4 || turn == 8) && g.Bag().TilesRemaining() > 1 {
			// The best one-tile play instead, which leaves six tiles to
			// read: the static player rarely makes one, and only those use
			// the leave-value term. The game goes on with its own move.
			// A player of its own: the game's keeps to the best move.
			for _, alt := range staticPlayer(t, cfg, g).GenerateMoves(1 << 20) {
				if alt.Action() == move.MoveTypePlay && alt.TilesPlayed() == 1 {
					one := new(move.Move)
					one.CopyFrom(alt)
					ones = append(ones, inferCase{before: before, m: one})
					break
				}
			}
		}
		m := turnplayer.GenBestStaticTurn(g, p, me)
		chosen := new(move.Move)
		chosen.CopyFrom(m)
		readable := (chosen.Action() == move.MoveTypePlay && chosen.TilesPlayed() < game.RackTileLimit) ||
			chosen.Action() == move.MoveTypeExchange
		if readable && g.Bag().TilesRemaining() > chosen.TilesPlayed() {
			all = append(all, inferCase{before: before, m: chosen})
		}
		if err := g.PlayMove(chosen, false, 0); err != nil {
			t.Fatal(err)
		}
	}
	out := all
	if len(all) > n {
		out = make([]inferCase, n)
		for i := range out {
			out[i] = all[i*(len(all)-1)/(n-1+boolInt(n == 1))]
		}
	}
	out = append(out, ones...)
	return out
}

// TestInferSimCrossCheck sims the position after the opponent's move with
// their rack dealt from the inference's posterior, as the elite bot does
// (best-arm, bag cycling asked for but off with a posterior), and plainly
// for a fixed number of iterations, in both engines.
func TestInferSimCrossCheck(t *testing.T) {
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
	const master = 20261128
	sims := 0
	for _, bc := range boardCases {
		if bc.peg {
			continue
		}
		t.Run(bc.lexicon+"-"+bc.layout, func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dataPath, "lexica", "gaddag", bc.lexicon+".kwg")); err != nil {
				t.Skipf("no %s word graph", bc.lexicon)
			}
			cases := inferCasesFrom(t, cfg, bc, automatic.DeriveSeed(master, 1), *simPositions)
			for i, c := range cases {
				threads := 1 + 3*(i%2)
				if err := e.Send(fmt.Sprintf("setoption name threads value %d", threads)); err != nil {
					t.Fatal(err)
				}
				for _, bai := range []bool{true, false} {
					if !bai && threads > 1 {
						// A plain sim is only the same on one thread.
						continue
					}
					compareInferSim(t, e, cfg, c, uint64(i+1), *inferBudget, bai)
					sims++
				}
			}
		})
	}
	t.Logf("%d sims with inference match", sims)
}

func compareInferSim(t *testing.T, e *Engine, cfg *config.Config, c inferCase, seed uint64, budget int, bai bool) {
	t.Helper()
	g := parseCGP(t, cfg, c.before)
	liveSeed := automatic.DeriveSeed(seed, 0)
	g.SeedBag(liveSeed)
	alph := g.Alphabet()
	mover := g.PlayerOnTurn()
	m := new(move.Move)
	m.CopyFrom(c.m)
	lm := lastMoveNotation(m, g.Board(), alph)
	if err := g.PlayMove(m, true, 0); err != nil {
		t.Fatal(err)
	}
	me := g.PlayerOnTurn()
	after := fmt.Sprintf("%s %s/ %d/%d %d lex %s; ld %s; bdn %s; lm %s %d;",
		g.Board().ToFEN(alph), g.RackFor(me).TilesOn().UserVisible(alph),
		g.PointsFor(me), g.PointsFor(mover), g.ScorelessTurns(),
		g.LexiconName(), g.Rules().LetterDistributionName(), g.Rules().BoardName(), lm, m.Score())
	leaves := ""
	if g.Rules().BoardName() == board.SuperCrosswordGameLayout {
		leaves = "super-leaves.klv2"
	}
	calc, err := equity.NewCombinedStaticCalculator(g.LexiconName(), cfg, leaves, equity.PEGAdjustmentFilename)
	if err != nil {
		t.Fatal(err)
	}
	rf := &rangefinder.RangeFinder{}
	rf.Init(g, []equity.EquityCalculator{calc}, cfg)
	rf.SetThreads(1)
	rf.SetTauCap(0.1)
	rf.SetBudget(budget)
	if err := rf.PrepareFinder(g.RackFor(me).TilesOn()); err != nil {
		t.Fatalf("%s: Go: %v", after, err)
	}
	if err := rf.Infer(context.Background()); err != nil {
		t.Fatalf("%s: Go: %v", after, err)
	}
	inf := rf.Inferences()
	if !inf.Complete || len(inf.InferredRacks) == 0 {
		t.Fatalf("%s: Go read no posterior", after)
	}

	// The sim, as the Zig engine sees the position: the opponent's rack
	// unknown, the bag every unseen tile in order.
	g.ThrowRacksInFor(1 - me)
	slices.Sort(g.Bag().Tiles())
	plays := staticPlayer(t, cfg, g).GenerateMoves(10)
	s := &montecarlo.Simmer{}
	s.Init(g, []equity.EquityCalculator{calc}, calc, cfg)
	s.SetThreads(1)
	s.SetEndgameLeaves(false)
	s.SetBAIBagCycle(true)
	const plies = 2
	if err := s.PrepareSim(plies, plays); err != nil {
		t.Fatal(err)
	}
	s.SetInferences(inf.InferredRacks, inf.RackLength, montecarlo.InferenceCompletePosterior)
	var cmd string
	if bai {
		s.SetStoppingCondition(montecarlo.Stop99)
		s.SetStopRule(montecarlo.StopRuleBAI)
		s.SetStopCapPerPlay(3 * *simIterations)
		s.SetBAIBestPlayCap(2000)
		if err := s.Simulate(context.Background()); err != nil {
			t.Fatal(err)
		}
		cmd = fmt.Sprintf("go sim plies %d plays 10 seed %d stop 99 cap %d bestcap 2000 cycle 1 infer %d", plies, seed, 3**simIterations, budget)
	} else {
		s.SimSingleThread(*simIterations, plies)
		cmd = fmt.Sprintf("go sim plies %d iterations %d plays 10 seed %d infer %d", plies, *simIterations, seed, budget)
	}
	want := map[string]simResult{}
	ranked := s.PlaysByWinProb().PlaysNoLock()
	for _, sp := range ranked {
		want[ucgiMove(sp.Move(), alph)] = simResult{win: sp.WinProb(), equity: sp.EquityMean(), iters: sp.WinProbIterations()}
	}
	best := ucgiMove(ranked[0].Move(), alph)

	if err := e.Send("position cgp " + after); err != nil {
		t.Fatal(err)
	}
	lines, err := e.Command(cmd, "bestmove")
	if err != nil {
		t.Fatalf("%s: %v", after, err)
	}
	zig := map[string]simResult{}
	read := false
	for _, l := range lines[:len(lines)-1] {
		f := strings.Fields(l)
		switch {
		case strings.HasPrefix(l, "info nodes "), strings.HasPrefix(l, "info sim status "):
			continue
		case strings.HasPrefix(l, "info infer "):
			read = f[2] == "leaves"
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
	if !read {
		t.Fatalf("%s (%s): Zig read no posterior: %q", after, cmd, lines)
	}
	if len(want) != len(zig) {
		t.Fatalf("%s: Go simmed %d plays, Zig %d", after, len(want), len(zig))
	}
	for mv, w := range want {
		if z, ok := zig[mv]; !ok || z != w {
			t.Fatalf("%s (%s): %s: Go %+v, Zig %+v", after, cmd, mv, w, z)
		}
	}
	if zigBest := strings.TrimPrefix(lines[len(lines)-1], "bestmove "); zigBest != best {
		t.Fatalf("%s (%s): Go picked %s, Zig %s", after, cmd, best, zigBest)
	}
}
