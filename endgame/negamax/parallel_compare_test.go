package negamax

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/domino14/word-golib/kwg"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/movegen"
)

// compareTable is the one transposition table the comparisons below solve
// with. A solver otherwise makes its own, a quarter of the machine's memory
// by default, and on a big machine a run of solves then spends its time
// allocating and collecting tables.
var compareTable = &TranspositionTable{}

func useCompareTable(s *Solver) {
	fraction := 0.02
	if v := os.Getenv("ENDGAME_COMPARE_TT_FRACTION"); v != "" {
		fraction, _ = strconv.ParseFloat(v, 64)
	}
	s.SetTTable(compareTable)
	s.SetTTableFraction(fraction)
}

// TestCompareParallelAlgorithms times the parallel searches against each
// other. It only runs when asked to:
//
//	ENDGAME_COMPARE=abdada,pvsplit ENDGAME_COMPARE_THREADS=1,4,16 \
//	  go test ./endgame/negamax -run TestCompareParallelAlgorithms -v
func TestCompareParallelAlgorithms(t *testing.T) {
	algos := os.Getenv("ENDGAME_COMPARE")
	if algos == "" {
		t.Skip("set ENDGAME_COMPARE to a list of parallel algorithms")
	}
	threadCounts := []int{16}
	if v := os.Getenv("ENDGAME_COMPARE_THREADS"); v != "" {
		threadCounts = nil
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			threadCounts = append(threadCounts, n)
		}
	}
	runs := 1
	if v := os.Getenv("ENDGAME_COMPARE_RUNS"); v != "" {
		runs, _ = strconv.Atoi(v)
	}
	only := os.Getenv("ENDGAME_COMPARE_POS")

	fromCGP := func(pos, lex string) func() *Solver {
		return func() *Solver {
			g, err := cgp.ParseCGP(DefaultConfig, pos)
			if err != nil {
				t.Fatal(err)
			}
			gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), lex)
			if err != nil {
				t.Fatal(err)
			}
			g.SetBackupMode(game.SimulationMode)
			g.RecalculateBoard()
			gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
			s := new(Solver)
			if err := s.Init(gen, g.Game); err != nil {
				t.Fatal(err)
			}
			return s
		}
	}
	fromBoard := func(lex, dist string, b board.VsWho, r1, r2 string, p1, p2, onTurn int) func() *Solver {
		return func() *Solver {
			s, err := setUpSolver(lex, dist, b, 0, r1, r2, p1, p2, onTurn)
			if err != nil {
				t.Fatal(err)
			}
			return s
		}
	}
	// fromGCG is the position after turn turns of a game, and then some moves.
	fromGCG := func(lex, file string, turn int, moves ...[2]string) func() *Solver {
		return func() *Solver {
			rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout,
				"English", game.CrossScoreAndSet, game.VarClassic)
			if err != nil {
				t.Fatal(err)
			}
			history, err := gcgio.ParseGCG(DefaultConfig, "../../gcgio/testdata/"+file)
			if err != nil {
				t.Fatal(err)
			}
			g, err := game.NewFromHistory(history, rules, turn)
			if err != nil {
				t.Fatal(err)
			}
			for _, m := range moves {
				if _, err := g.PlayScoringMove(m[0], m[1], false); err != nil {
					t.Fatal(err)
				}
			}
			gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), lex)
			if err != nil {
				t.Fatal(err)
			}
			g.SetBackupMode(game.SimulationMode)
			gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
			s := new(Solver)
			if err := s.Init(gen, g); err != nil {
				t.Fatal(err)
			}
			return s
		}
	}
	viaOpp := "5U4OHMIC/5N3WREATH/5T4FAX2/5i3B1VIA1/5N3L1E3/5G2VELDT2/5E3S5/" +
		"5DREKS1F3/8YELL3/4ABASER1U3/4GYM3ZO3/WAITE5OR2J/10OI2A/3QUOIT1PINNER/" +
		"4RENEGADE2P AIINOOU/CDIOST? 392/450 0 lex CSW24;"
	lexChange := "PISOLITH7/E14/r13E/I1MEW2F3H2C/ATENOLOL2NY2Z/G6A2AR1BE/U6VERBATIM/" +
		"A4UPO2EX1NA/6OR7/5WO8/4IRKSOmE1T2/5I1QUINOA2/5N1U4Y2/5G1A7/" +
		"GRICES1DELISTED ADDEJRT/FNV 536/474 0 lex "
	// The positions of this package's tests, with the values they assert
	// (want 0: none asserted).
	positions := []struct {
		name  string
		make  func() *Solver
		plies int
		want  int16
	}{
		{"vsroy", fromBoard("America", "english", board.VsRoy, "WZ", "EFHIKOQ", 427, 331, 1), 8, 116},
		{"vsjoey", fromBoard("NWL18", "english", board.VsJoey, "DIV", "AEFILMR", 412, 371, 1), 14, 55},
		{"canik", fromBoard("NWL20", "english", board.VsCanik, "DEHILOR", "BGIV", 389, 384, 1), 4, 11},
		{"joel", fromBoard("NWL18", "english", board.VsJoel, "EIQSS", "AAFIRTW", 393, 373, 1), 4, 25},
		{"polish", fromBoard("OSPS49", "polish", board.APolishEndgame, "BGHUWZZ", "IKMÓŹŻ", 304, 258, 0), 14, 5},
		{"passfirst", fromCGP("GATELEGs1POGOED/R4MOOLI3X1/AA10U2/YU4BREDRIN2/1TITULE3E1IN1/"+
			"1E4N3c1BOK/1C2O4CHARD1/QI1FLAWN2E1OE1/IS2E1HIN1A1W2/1MOTIVATE1T1S2/1S2N5S4/"+
			"3PERJURY5/15/15/15 FV/AADIZ 442/388 0 lex CSW19;", "CSW19"), 8, -63},
		{"zeropt", fromCGP("IBADAT1B7/2CAFE1OD1TRANQ/2TUT2RENIED2/3REV2YOMIM2/4RAFT1NISI2/"+
			"5COR2N1x2/6LA1AGEE2/6LIAISED2/5POKY2W3/4JOWS7/V2LUZ9/ORPIN10/L1OE11/TUX12/"+
			"I14 EEEEGH?/AGHNOSU 308/265 0 lex CSW19;", "CSW19"), 11, -42},
		{"noah7", fromGCG("NWL18", "noah_vs_mishu.gcg", 28, [2]string{"H7", "T..."},
			[2]string{"N5", "C..."}, [2]string{"10A", ".IN"}, [2]string{"13L", "...R"}), 7, 44},
		{"noah8", fromGCG("NWL18", "noah_vs_mishu.gcg", 28, [2]string{"H7", "T..."},
			[2]string{"N5", "C..."}, [2]string{"10A", ".IN"}, [2]string{"13L", "...R"}), 8, 44},
		{"frentz", fromGCG("CSW19", "vs_frentz.gcg", 22), 3, 99},
		{"truncpv", fromGCG("CSW24", "truncated_endgame_pv.gcg", 23), 7, 27},
		{"jmz", fromCGP("11CLAG/9BRAAI1/11N3/11D3/10BOW2/10ERA2/6A3A1R2/C2WEDDInGS1P1I/"+
			"U1VOX1O3T3N/RHO1E1PERILUNES/FEM1A1T3Y3O/1HINT1I7U/2TO2V2GEED1L/"+
			"1AYU2ENTITLES1/QI1N3OAF1KEiR EIOS/JMZ 318/517 0 lex CSW21;", "CSW21"), 5, 60},
		{"eldarstick", fromCGP("4EXODE6/1DOFF1KERATIN1U/1OHO8YEN/1POOJA1B3MEWS/5SQUINTY2A/"+
			"4RHINO1e3V/2B4C2R3E/GOAT1D1E2ZIN1d/1URACILS2E4/1PIG1S4T4/"+
			"2L2R4T4/2L2A1GENII3/2A2T1L7/5E1A7/5D1M7 AEEIRUW/V 410/409 0 lex CSW21;", "CSW21"), 3, 72},
		{"viaopp", fromCGP(viaOpp, "CSW24"), 3, -78},
		{"lexcsw21", fromCGP(lexChange+"CSW21;", "CSW21"), 2, 47},
		{"lexnwl23", fromCGP(lexChange+"NWL23;", "NWL23"), 2, 43},
		{"aeeerru", fromCGP("14W/13TA/12SIG/11NOME/11ERE1/5Z5GID1/O4OD5C1V/LA1LININ4O1I/"+
			"EL3ON2ULTIMO/AA2HIT4IDYl/1A2ED8i/1POUR6V2N/4B5JEFES/1TRAYF3WAXES1/"+
			"PIET1ANCHO1TUSK AEEERRU/BGQR 281/428 0 lex CSW21;", "CSW21"), 9, 38},
		{"airt", fromCGP("OUTREmER4DZO/EPEE3E1MOVIE1/2MAInSTAY2VAN/HEEL3R2FUEL1/7A7/"+
			"5CRINGINGS1/4L2N5A1/2AXOIDS3WINY/2BUB8T1/4UH7E1/4LI7R1/3DEN7OK/"+
			"2JO1D8A/2OI10F/COTTA9S AIRT/GPQW 469/454 0 lex CSW21;", "CSW21"), 14, 57},
		{"drsu", fromCGP("15/12B2/6INTRADOS1/2RETINUE3A2/10DATO1/9KALIF1/7WAY2E2/VOE3FEM6/"+
			"2ZE1VEXILS4/1QIN2HERON4/TANTrUM2BY4/1PEE1GEO7/1I1R3PHATIC2/"+
			"1k1I4OWE4/3C1JIAO1GUANO DRSU/DGLLRS 437/397 0 lex CSW21;", "CSW21"), 14, 44},
		{"verydeep", fromCGP("14C/13QI/12FIE/10VEE1R/9KIT2G/8CIG1IDE/8UTA2AS/7ST1SYPh1/"+
			"6JA5A1/5WOLD2BOBA/3PLOT1R1NU1EX/Y1VEIN1NOR1mOA1/UT1AT1N1L2FEH1/GUR2WIRER5/"+
			"SNEEZED8 ADENOOO/AHIILMM 353/236 0 lex CSW19;", "CSW21"), 25, -116},
	}
	// ENDGAME_COMPARE_CGP="<lexicon> <plies> <cgp>" adds a position.
	if v := os.Getenv("ENDGAME_COMPARE_CGP"); v != "" {
		f := strings.SplitN(v, " ", 3)
		plies, err := strconv.Atoi(f[1])
		if err != nil {
			t.Fatal(err)
		}
		positions = append(positions[:0], struct {
			name  string
			make  func() *Solver
			plies int
			want  int16
		}{"cgp", fromCGP(f[2], f[0]), plies, 0})
	}

	for _, pos := range positions {
		if only != "" && !strings.Contains(","+only+",", ","+pos.name+",") {
			continue
		}
		for _, threads := range threadCounts {
			for _, algo := range strings.Split(algos, ",") {
				for run := 0; run < runs; run++ {
					s := pos.make()
					useCompareTable(s)
					pvSplitHelp = true
					if algo == "pvsplit-nohelp" {
						algo, pvSplitHelp = ParallelAlgoPVSplit, false
						defer func() { pvSplitHelp = true }()
					}
					label := algo
					if !pvSplitHelp {
						label += "-nohelp"
					}
					s.SetThreads(threads)
					if threads > 1 {
						if err := s.SetParallelAlgorithm(algo); err != nil {
							t.Fatal(err)
						}
					}
					start := time.Now()
					v, _, err := s.Solve(context.Background(), pos.plies)
					if err != nil {
						t.Fatal(err)
					}
					if pos.want != 0 && v != pos.want {
						t.Errorf("%s: %s with %d threads gave %d, want %d", pos.name, label, threads, v, pos.want)
					}
					fmt.Printf("RESULT %-9s plies=%-2d threads=%-2d %-14s value=%-5d %8.2fs nodes=%d\n",
						pos.name, pos.plies, threads, label, v, time.Since(start).Seconds(), s.nodes.Load())
				}
			}
		}
	}
}

// randomEndgame plays the highest-scoring move for each side from a fresh game
// until the bag is empty and returns the position reached, or "" if the game
// ended first.
func randomEndgame(t *testing.T, lex string) string {
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, "english",
		game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		t.Fatal(err)
	}
	g, err := game.NewGame(rules, []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "Player 1"},
		{Nickname: "p2", RealName: "Player 2"},
	})
	if err != nil {
		t.Fatal(err)
	}
	g.StartGame()
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), lex)
	if err != nil {
		t.Fatal(err)
	}
	gen := movegen.NewGordonGenerator(gd, g.Board(), rules.LetterDistribution())
	gen.SetSortingParameter(movegen.SortByScore)
	for g.Playing() == pb.PlayState_PLAYING && g.Bag().TilesRemaining() > 0 {
		gen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
		if err := g.PlayMove(gen.Plays()[0], false, 0); err != nil {
			t.Fatal(err)
		}
	}
	if g.Playing() != pb.PlayState_PLAYING {
		return ""
	}
	return g.ToCGP(false)
}

// TestCompareParallelRandomEndgames solves random endgames with one thread and
// with each parallel search, checks that the values agree and adds up the
// time each took:
//
//	ENDGAME_COMPARE_RANDOM=50 ENDGAME_COMPARE=abdada,pvsplit ENDGAME_COMPARE_PLIES=6 \
//	  go test ./endgame/negamax -run TestCompareParallelRandomEndgames -v
func TestCompareParallelRandomEndgames(t *testing.T) {
	n, _ := strconv.Atoi(os.Getenv("ENDGAME_COMPARE_RANDOM"))
	if n == 0 {
		t.Skip("set ENDGAME_COMPARE_RANDOM to a number of positions")
	}
	algos := strings.Split(os.Getenv("ENDGAME_COMPARE"), ",")
	plies := 6
	if v := os.Getenv("ENDGAME_COMPARE_PLIES"); v != "" {
		plies, _ = strconv.Atoi(v)
	}
	threads := 16
	if v := os.Getenv("ENDGAME_COMPARE_THREADS"); v != "" {
		threads, _ = strconv.Atoi(v)
	}
	// Positions the one-thread solve finishes faster than this say nothing
	// about parallel search.
	minSeconds := 1.0
	if v := os.Getenv("ENDGAME_COMPARE_MIN_SECONDS"); v != "" {
		minSeconds, _ = strconv.ParseFloat(v, 64)
	}
	const lex = "NWL23"
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), lex)
	if err != nil {
		t.Fatal(err)
	}
	solve := func(pos, algo string, threads int) (int16, float64) {
		g, err := cgp.ParseCGP(DefaultConfig, pos)
		if err != nil {
			t.Fatal(err)
		}
		g.SetBackupMode(game.SimulationMode)
		g.RecalculateBoard()
		gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
		s := new(Solver)
		if err := s.Init(gen, g.Game); err != nil {
			t.Fatal(err)
		}
		useCompareTable(s)
		s.SetThreads(threads)
		if threads > 1 {
			if err := s.SetParallelAlgorithm(algo); err != nil {
				t.Fatal(err)
			}
		}
		start := time.Now()
		v, _, err := s.Solve(context.Background(), plies)
		if err != nil {
			t.Fatal(err)
		}
		return v, time.Since(start).Seconds()
	}
	totals := make([]float64, len(algos))
	wins := make([]int, len(algos))
	var single float64
	used := 0
	for used < n {
		pos := randomEndgame(t, lex)
		if pos == "" {
			continue
		}
		want, base := solve(pos, "", 1)
		if base < minSeconds {
			continue
		}
		used++
		single += base
		line := fmt.Sprintf("RANDOM %3d value=%-5d 1-thread=%7.2fs", used, want, base)
		best := 0
		times := make([]float64, len(algos))
		for i, algo := range algos {
			v, secs := solve(pos, algo, threads)
			if v != want {
				t.Errorf("%s: %s gave %d, one thread gave %d", pos, algo, v, want)
			}
			times[i] = secs
			totals[i] += secs
			if secs < times[best] {
				best = i
			}
			line += fmt.Sprintf(" %s=%7.2fs", algo, secs)
		}
		wins[best]++
		fmt.Println(line, " ", pos)
	}
	for i, algo := range algos {
		fmt.Printf("TOTAL %-8s %8.2fs  speedup over one thread %.2fx  fastest on %d of %d\n",
			algo, totals[i], single/totals[i], wins[i], used)
	}
}
