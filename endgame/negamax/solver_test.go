package negamax

import (
	"context"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/matryer/is"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

func TestMain(m *testing.M) {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	os.Exit(m.Run())
}

var DefaultConfig = config.DefaultConfig()

func setUpSolver(lex, distName string, bvs board.VsWho, plies int, rack1, rack2 string,
	p1pts, p2pts int, onTurn int) (*Solver, error) {

	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, distName, game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		panic(err)
	}

	players := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "Player 1"},
		{Nickname: "p2", RealName: "Player 2"},
	}

	g, err := game.NewGame(rules, players)
	if err != nil {
		panic(err)
	}

	g.StartGame()
	g.SetBackupMode(game.SimulationMode)
	// Throw in the random racks dealt to our players.
	g.ThrowRacksIn()

	gd, err := kwg.GetKWG(g.Config().WGLConfig(), lex)
	if err != nil {
		panic(err)
	}

	dist := rules.LetterDistribution()
	gen := movegen.NewGordonGenerator(gd, g.Board(), dist)
	gen.SetPlayRecorder(movegen.AllPlaysRecorder)
	alph := g.Alphabet()

	tilesInPlay := g.Board().SetToGame(alph, bvs)
	err = g.Bag().RemoveTiles(tilesInPlay.OnBoard)
	if err != nil {
		panic(err)
	}
	cross_set.GenAllCrossSets(g.Board(), gd, dist)

	g.SetRacksForBoth([]*tilemapping.Rack{
		tilemapping.RackFromString(rack1, alph),
		tilemapping.RackFromString(rack2, alph),
	})
	g.SetPointsFor(0, p1pts)
	g.SetPointsFor(1, p2pts)
	g.SetPlayerOnTurn(onTurn)
	fmt.Println("Racks are", g.RackLettersFor(0), g.RackLettersFor(1))
	fmt.Println(g.Board().ToDisplayText(alph))

	s := new(Solver)

	err = s.Init(gen, g)
	if err != nil {
		panic(err)
	}
	return s, nil
}

func TestSolveComplex(t *testing.T) {
	is := is.New(t)
	plies := 8

	s, err := setUpSolver("America", "english", board.VsRoy, plies, "WZ", "EFHIKOQ", 427, 331,
		1)
	is.NoErr(err)

	v, _, _ := s.Solve(context.Background(), plies)
	is.Equal(v, int16(116))
	// Quackle finds a 122-pt win. However, I think it's wrong because it
	// doesn't take into account that opp can pass to prevent a setup
	// (the setup being: EF 3F to block the Z, then YO, YOK/KHI, QI)
	// The setup only works if Roy plays off his W before YO.
}

// This one is kind of tricky! You need to look at least 13 plies deep to
// find the right solution for some reason, even though the right solution is
// only 7 moves long.
// PV; val 55
// 1: J11 .R (2)
// 2:  F2 DI. (19)
// 3:  6N .I (11)
// 4: (Pass) (0)
// 5: 12I E. (4)
// 6: (Pass) (0)
// 7: H12 FLAM (49)

func TestSolveOther3(t *testing.T) {
	plies := 14
	is := is.New(t)
	s, err := setUpSolver("NWL18", "english", board.VsJoey, plies, "DIV", "AEFILMR", 412, 371,
		1)
	is.NoErr(err)
	s.lazySMPOptim = true
	// sweet spot seems to be threads = 8 for now
	s.threads = 8
	v, _, _ := s.Solve(context.Background(), plies)
	is.Equal(v, int16(55))
}

func TestSolveStandard(t *testing.T) {
	// This endgame is solved with at least 3 plies. Most endgames should
	// start with 3 plies (so the first player can do an out in 2) and
	// then proceed with iterative deepening.
	plies := 4

	is := is.New(t)

	s, err := setUpSolver("NWL20", "english", board.VsCanik, plies, "DEHILOR", "BGIV", 389, 384,
		1)
	is.NoErr(err)
	bestV, bestSeq, err := s.Solve(context.Background(), plies)
	is.NoErr(err)
	is.Equal(bestV, int16(11))
	fmt.Println(bestSeq)
	is.Equal(len(bestSeq), 3)
}

func TestSolveStandard2(t *testing.T) {
	// Another standard 3-ply endgame.
	is := is.New(t)
	plies := 4

	s, err := setUpSolver("NWL18", "english", board.VsJoel, plies, "EIQSS", "AAFIRTW", 393, 373,
		1)
	is.NoErr(err)

	v, _, _ := s.Solve(context.Background(), plies)
	is.Equal(v, int16(25))
}

// func TestSolveNegamaxFunc(t *testing.T) {
// 	plies := 4

// 	is := is.New(t)

// 	s, err := setUpSolver("NWL18", "english", board.VsCanik, plies, "DEHILOR", "BGIV", 389, 384,
// 		1)
// 	is.NoErr(err)

// 	// Test just the negamax function without the search etc functionality.
// 	s.requestedPlies = 4
// 	s.currentIDDepth = 4
// 	s.zobrist.Initialize(s.game.Board().Dim())
// 	globalTranspositionTable.reset(0.01)

// 	s.stmMovegen.SetSortingParameter(movegen.SortByNone)
// 	defer s.stmMovegen.SetSortingParameter(movegen.SortByScore)

// 	s.game.SetMaxScorelessTurns(2)
// 	defer s.game.SetMaxScorelessTurns(game.DefaultMaxScorelessTurns)

// 	s.initialSpread = s.game.CurrentSpread()
// 	s.initialTurnNum = s.game.Turn()
// 	s.solvingPlayer = s.game.PlayerOnTurn()

// 	ctx := context.Background()
// 	pv := &PVLine{}
// 	score, err := s.negamax(ctx, 0, s.requestedPlies, -HugeNumber, HugeNumber, pv)
// 	is.NoErr(err)
// 	// we win by 6
// 	is.Equal(score, int16(6))
// 	is.Equal(len(pv.Moves), 3)
// }

func TestVeryDeep(t *testing.T) {
	is := is.New(t)
	plies := 25
	// The following is a very deep endgame that requires 25 plies to solve.
	deepEndgame := "14C/13QI/12FIE/10VEE1R/9KIT2G/8CIG1IDE/8UTA2AS/7ST1SYPh1/6JA5A1/5WOLD2BOBA/3PLOT1R1NU1EX/Y1VEIN1NOR1mOA1/UT1AT1N1L2FEH1/GUR2WIRER5/SNEEZED8 ADENOOO/AHIILMM 353/236 0 lex CSW19;"
	g, err := cgp.ParseCGP(DefaultConfig, deepEndgame)
	is.NoErr(err)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	g.RecalculateBoard()
	gen := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(gen, g.Game)
	// s.iterativeDeepeningOptim = false
	// s.transpositionTableOptim = false
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
	v, _, _ := s.Solve(context.Background(), plies)

	is.Equal(v, int16(-116))
	// XXX: figure out how to get the seq back
	// is.Equal(len(seq), 25)
}

// This endgame's first move must be a pass, otherwise Nigel can set up
// an unblockable ZA.
func TestPassFirst(t *testing.T) {
	is := is.New(t)

	plies := 8
	// https://www.cross-tables.com/annotated.php?u=25243#22
	pos := "GATELEGs1POGOED/R4MOOLI3X1/AA10U2/YU4BREDRIN2/1TITULE3E1IN1/1E4N3c1BOK/1C2O4CHARD1/QI1FLAWN2E1OE1/IS2E1HIN1A1W2/1MOTIVATE1T1S2/1S2N5S4/3PERJURY5/15/15/15 FV/AADIZ 442/388 0 lex CSW19;"
	g, err := cgp.ParseCGP(DefaultConfig, pos)
	is.NoErr(err)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW19")
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	g.RecalculateBoard()
	gen1 := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(gen1, g.Game)
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
	v, seq, _ := s.Solve(context.Background(), plies)

	// -63 is the true value: solving to 10 plies (7 tiles between the two
	// racks, so 9 plies covers every line) returns -63 with or without the
	// heuristics. This test searches only 8 plies, which is not deep enough to
	// prove it; the greedy playout gets there anyway, where the static leaf
	// used to report -60.
	t.Logf("value %d over %d moves, %d of them searched", v, len(seq), s.NumSearchedMoves())
	is.Equal(v, int16(-63))
	is.Equal(seq[0].Action(), move.MoveTypePass)
	is.True(len(seq) >= 6)
}

func TestPolish(t *testing.T) {
	is := is.New(t)
	plies := 14
	s, err := setUpSolver(
		"OSPS49", "polish", board.APolishEndgame, plies, "BGHUWZZ", "IKMÓŹŻ", 304,
		258, 0)

	is.NoErr(err)
	v, _, err := s.Solve(context.Background(), plies)
	is.NoErr(err)

	/*
	   Best sequence has a spread difference of 5
	   Best sequence:
	   1) N7 ZG..
	   2) M1 ŻM..
	   3) (Pass)
	   4) 6L .I
	   5) B8 ZU.
	   6) 9A K.
	   7) (Pass)
	   8) (Pass)

	*/

	is.Equal(v, int16(5))
}

func TestPolishFromGcg(t *testing.T) {
	plies := 14
	is := is.New(t)

	rules, err := game.NewBasicGameRules(DefaultConfig, "OSPS49", board.CrosswordGameLayout, "Polish", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, "OSPS49")
	cfg.Set(config.ConfigDefaultLetterDistribution, "polish")

	gameHistory, err := gcgio.ParseGCG(cfg, "../../gcgio/testdata/polish_endgame.gcg")
	is.NoErr(err)
	gameHistory.ChallengeRule = pb.ChallengeRule_SINGLE

	g, err := game.NewFromHistory(gameHistory, rules, 46)
	is.NoErr(err)

	gd, err := kwg.GetKWG(cfg.WGLConfig(), "OSPS49")
	is.NoErr(err)

	g.SetBackupMode(game.SimulationMode)
	gen := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)

	s.Init(gen, g)
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))

	v, _, _ := s.Solve(context.Background(), plies)
	is.Equal(v, int16(5))
}

func TestStuckPruning(t *testing.T) {
	// This is very slow: solving it exactly needs 11 plies. The greedy playout
	// reaches the same +72 at 2 plies in a fraction of a second -- see
	// TestStuckEndgameSolvedShallow in heuristics_test.go, which uses this same
	// position.
	t.Skip()
	is := is.New(t)
	plies := 11
	// See EldarVsNigel in sample_testing_boards.go
	// In this endgame, Eldar is stuck with a V and Nigel can 1-tile him.
	// The optimal sequences are:

	// 1.	+72	13►	1. [AL 11B 8 EEIRUW] [none V] 2. [DUM 15F 6 EEIRW] [none V] 3. [IT 11J 4 EERW] [none V] 4. [EWT 10I 28 ER] [none V] 5. [RE 3F 18] [-] 6. [end V +8]
	// 1.	+72	14►	1. [DUM 15F 6 AEEIRW] [none V] 2. [AL 11B 8 EEIRW] [none V] 3. [IT 11J 4 EERW] [none V] 4. [EWT 10I 28 ER] [none V] 5. [RE 3F 18] [-] 6. [end V +8]
	// 1.	+72	10►	1. [IT 11J 4 AEERUW] [none V] 2. [EWT 10I 28 AERU] [none V] 3. [AL 11B 8 ERU] [none V] 4. [DUM 15F 6 ER] [none V] 5. [RE 3F 18] [-] 6. [end V +8]
	// 1.	+72	13►	1. [RASH F3 7 AEEIUW] [none V] 2. [AL 11B 8 EEIUW] [none V] 3. [DUM 15F 6 EEIW] [none V] 4. [IT 11J 4 EEW] [none V] 5. [EWT 10I 28 E] [none V] 6. [RE 3F 11] [-] 7. [end V +8]
	// 1.	+72	5►	1. [RE 3F 18 AEIUW] [none V] 2. [AL 11B 8 EIUW] [none V] 3. [DUM 15F 6 EIW] [none V] 4. [IT 11J 4 EW] [none V] 5. [EWT 10I 28] [-] 6. [end V +8]
	// Courtesy of elbbarcs.com -- thank you

	deepEndgame := "4EXODE6/1DOFF1KERATIN1U/1OHO8YEN/1POOJA1B3MEWS/5SQUINTY2A/4RHINO1e3V/2B4C2R3E/GOAT1D1E2ZIN1d/1URACILS2E4/1PIG1S4T4/2L2R4T4/2L2A1GENII3/2A2T1L7/5E1A7/5D1M7 AEEIRUW/V 410/409 0 lex CSW19;"
	g, err := cgp.ParseCGP(DefaultConfig, deepEndgame)
	is.NoErr(err)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW19")
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	g.RecalculateBoard()
	gen := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)

	s.Init(gen, g.Game)
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
	v, _, _ := s.Solve(context.Background(), plies)
	is.Equal(v, int16(72))
}

// Test that iterative deepening actually works properly.
func TestProperIterativeDeepening(t *testing.T) {
	is := is.New(t)
	// Should get the same result with 7 or 8 plies.
	plyCount := []int{7, 8}
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	for _, plies := range plyCount {

		gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../../gcgio/testdata/noah_vs_mishu.gcg")
		is.NoErr(err)

		g, err := game.NewFromHistory(gameHistory, rules, 28)
		is.NoErr(err)
		// Make a few plays:
		g.PlayScoringMove("H7", "T...", false)
		g.PlayScoringMove("N5", "C...", false)
		g.PlayScoringMove("10A", ".IN", false)
		// Note that this is not right; user should play the P off at 6I,
		// but this is for testing purposes only:
		g.PlayScoringMove("13L", "...R", false)
		is.Equal(g.PointsFor(0), 339)
		is.Equal(g.PointsFor(1), 381)

		gd, err := kwg.GetKWG(g.Config().WGLConfig(), g.LexiconName())
		is.NoErr(err)

		gen := movegen.NewGordonGenerator(
			gd, g.Board(), g.Bag().LetterDistribution(),
		)
		s := new(Solver)
		s.Init(gen, g)

		fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
		// Prior to solving the endgame, set to simulation mode.
		g.SetBackupMode(game.SimulationMode)
		v, seq, _ := s.Solve(context.Background(), plies)
		is.Equal(v, int16(44))
		// In particular, the sequence should start with 6I A.
		// Player on turn needs to block the P spot. Anything else
		// shows a serious bug.
		// Note: PV can be truncated due to TTable collisions in multi-threaded search,
		// so we just check that we have at least 3 moves and the first move is correct.
		is.True(len(seq) >= 3)
		is.Equal(seq[0].ShortDescription(), " 6I A.")
	}
}

func TestFromGCG(t *testing.T) {
	plies := 3
	is := is.New(t)

	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW19", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../../gcgio/testdata/vs_frentz.gcg")
	is.NoErr(err)

	g, err := game.NewFromHistory(gameHistory, rules, 22)
	is.NoErr(err)

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW19")
	is.NoErr(err)

	g.SetBackupMode(game.SimulationMode)
	gen := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)

	s.Init(gen, g)
	// s.iterativeDeepeningOn = false
	// s.simpleEvaluation = true
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
	v, seq, _ := s.Solve(context.Background(), plies)
	is.Equal(v, int16(99))
	is.Equal(len(seq), 1)
	// t.Fail()
}

func TestZeroPtFirstPlay(t *testing.T) {
	is := is.New(t)
	plies := 11
	/**
			Best sequence has a spread difference of -42
	Best sequence:
	1)  6M .u
	2)  6F ...N
	3)  N1 E.
	4)  O2 .UAG
	5) (Pass)
	6)  6E S....
	7) (Pass)
	8)  A1 .O
	9) (Pass)
	10) (Pass)
	*/

	deepEndgame := "IBADAT1B7/2CAFE1OD1TRANQ/2TUT2RENIED2/3REV2YOMIM2/4RAFT1NISI2/5COR2N1x2/6LA1AGEE2/6LIAISED2/5POKY2W3/4JOWS7/V2LUZ9/ORPIN10/L1OE11/TUX12/I14 EEEEGH?/AGHNOSU 308/265 0 lex CSW19;"
	g, err := cgp.ParseCGP(DefaultConfig, deepEndgame)
	is.NoErr(err)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW19")
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	g.RecalculateBoard()
	gen := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)

	s.Init(gen, g.Game)
	// s.iterativeDeepeningOptim = false
	// s.transpositionTableOptim = false

	// f, err := os.Create("/tmp/endgamelog-new")
	// is.NoErr(err)
	// defer f.Close()
	// s.logStream = f

	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
	v, _, _ := s.Solve(context.Background(), plies)
	is.Equal(v, int16(-42))
}

// This benchmark is really only here to measure allocations etc.

func BenchmarkPassFirst(b *testing.B) {
	is := is.New(b)

	plies := 8
	// https://www.cross-tables.com/annotated.php?u=25243#22
	pos := "GATELEGs1POGOED/R4MOOLI3X1/AA10U2/YU4BREDRIN2/1TITULE3E1IN1/1E4N3c1BOK/1C2O4CHARD1/QI1FLAWN2E1OE1/IS2E1HIN1A1W2/1MOTIVATE1T1S2/1S2N5S4/3PERJURY5/15/15/15 FV/AADIZ 442/388 0 lex CSW19;"
	g, err := cgp.ParseCGP(DefaultConfig, pos)
	is.NoErr(err)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW19")
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	g.RecalculateBoard()
	gen1 := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(gen1, g.Game)
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
	b.ResetTimer()
	// Large majority of this allocation is the transposition table.
	// 10/24/23  1	1578926502 ns/op	8699932296 B/op	 2720435 allocs/op
	// 11/9/23 1	1572327101 ns/op	8654502392 B/op	 1511844 allocs/op
	// 4/6/26  1	1190851193 ns/op	8690409824 B/op	   34308 allocs/op
	// 4/7/26  1	1186071173 ns/op	8672290200 B/op	    6830 allocs/op

	for i := 0; i < b.N; i++ {
		v, seq, _ := s.Solve(context.Background(), plies)

		// See TestPassFirst: -63 is the true value; 8 plies is not deep enough
		// to prove it, but the greedy playout gets there anyway.
		is.Equal(v, int16(-63))
		is.Equal(seq[0].Action(), move.MoveTypePass)
		is.True(len(seq) >= 6)
	}
}

// func TestSolveNegamaxFunc2(t *testing.T) {
// 	plies := 11

// 	is := is.New(t)

// 	deepEndgame := "IBADAT1B7/2CAFE1OD1TRANQ/2TUT2RENIED2/3REV2YOMIM2/4RAFT1NISI2/5COR2N1x2/6LA1AGEE2/6LIAISED2/5POKY2W3/4JOWS7/V2LUZ9/ORPIN10/L1OE11/TUX12/I14 EEEEGH?/AGHNOSU 308/265 0 lex CSW19;"
// 	g, err := cgp.ParseCGP(&DefaultConfig, deepEndgame)
// 	is.NoErr(err)
// 	gd, err := kwg.GetKWG(&DefaultConfig, "CSW19")
// 	is.NoErr(err)

// 	g.SetBackupMode(game.SimulationMode)
// 	g.SetStateStackLength(plies)
// 	g.RecalculateBoard()
// 	gen := movegen.NewGordonGenerator(
// 		gd, g.Board(), g.Bag().LetterDistribution(),
// 	)
// 	s := new(Solver)
// 	s.Init(gen, g)

// 	// Test just the negamax function without the search etc functionality.
// 	s.requestedPlies = plies
// 	s.currentIDDepth = plies
// 	s.zobrist.Initialize(s.game.Board().Dim())
// 	globalTranspositionTable.reset(0.01)

// 	s.stmMovegen.SetSortingParameter(movegen.SortByNone)
// 	defer s.stmMovegen.SetSortingParameter(movegen.SortByScore)

// 	s.game.SetMaxScorelessTurns(2)
// 	defer s.game.SetMaxScorelessTurns(game.DefaultMaxScorelessTurns)

// 	s.initialSpread = s.game.CurrentSpread()
// 	s.initialTurnNum = s.game.Turn()
// 	s.solvingPlayer = s.game.PlayerOnTurn()

// 	ctx := context.Background()
// 	pv := &PVLine{}
// 	score, err := s.negamax(ctx, 0, s.requestedPlies, -HugeNumber, HugeNumber, pv)
// 	is.NoErr(err)
// 	// we win by 1
// 	is.Equal(score, int16(1))

func benchmarkVsRoy(b *testing.B, prunedKWG bool) {
	b.Helper()
	plies := 8
	for range b.N {
		b.StopTimer()
		s, err := setUpSolver("America", "english", board.VsRoy, plies, "WZ", "EFHIKOQ", 427, 331, 1)
		if err != nil {
			b.Fatal(err)
		}
		s.SetPrunedKWGOptim(prunedKWG)
		b.StartTimer()
		v, _, err := s.Solve(context.Background(), plies)
		b.StopTimer()
		if err != nil {
			b.Fatal(err)
		}
		if v != 116 {
			b.Fatalf("unexpected result %d", v)
		}
	}
}

func BenchmarkVsRoyWithPrunedKWG(b *testing.B) {
	benchmarkVsRoy(b, true)
}

func BenchmarkVsRoyWithoutPrunedKWG(b *testing.B) {
	benchmarkVsRoy(b, false)
}

// }

// Autoplay runs one game per goroutine, and each game solves its endgames on
// its own transposition table. Solve several endgames at once, each with its
// own solver and table, and check every answer matches the known one -- and
// that the solves really were in flight at the same time.
func TestConcurrentSolvesIndependentTables(t *testing.T) {
	is := is.New(t)
	type position struct {
		lex     string
		bvs     board.VsWho
		plies   int
		r1, r2  string
		p1, p2  int
		want    int16
		threads int
	}
	// Cheap positions (a few seconds each at most), so the test stays quick
	// on a small CI runner.
	base := []position{
		{"NWL20", board.VsCanik, 4, "DEHILOR", "BGIV", 389, 384, 11, 1},
		{"NWL18", board.VsJoel, 4, "EIQSS", "AAFIRTW", 393, 373, 25, 2},
	}
	// Each position four times, so several solvers search the same tree at
	// once.
	var positions []position
	for range 4 {
		positions = append(positions, base...)
	}

	solvers := make([]*Solver, len(positions))
	for i, p := range positions {
		s, err := setUpSolver(p.lex, "english", p.bvs, p.plies, p.r1, p.r2, p.p1, p.p2, 1)
		is.NoErr(err)
		s.SetTTable(&TranspositionTable{})
		s.SetTTableFraction(0.01)
		s.SetThreads(p.threads)
		solvers[i] = s
	}

	vals := make([]int16, len(positions))
	errs := make([]error, len(positions))
	took := make([]time.Duration, len(positions))
	var inFlight, maxInFlight atomic.Int32
	begin := make(chan struct{})
	var wg sync.WaitGroup
	for i := range positions {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-begin
			n := inFlight.Add(1)
			for {
				m := maxInFlight.Load()
				if n <= m || maxInFlight.CompareAndSwap(m, n) {
					break
				}
			}
			st := time.Now()
			vals[i], _, errs[i] = solvers[i].Solve(context.Background(), positions[i].plies)
			took[i] = time.Since(st)
			inFlight.Add(-1)
		}(i)
	}
	close(begin)
	wg.Wait()

	for i := range positions {
		is.NoErr(errs[i])
		is.Equal(vals[i], positions[i].want)
		t.Logf("solve %d (%s, %d plies): %d in %v", i, positions[i].lex,
			positions[i].plies, vals[i], took[i])
	}
	t.Logf("max solves in flight at once: %d of %d", maxInFlight.Load(), len(positions))
	is.True(maxInFlight.Load() > 1)
}

// A transposition-table entry stored deeper than the node holds a deeper
// search's value, and nodes met those through passes and through threads
// searching other depths. Multi-threaded 5-ply solves of this position
// returned anything from 36 to 43 instead of 41, the value with one thread
// and with the table off; lazy SMP on two threads returned 40 every time.
func TestMultiThreadedSolveKeepsFixedDepthValue(t *testing.T) {
	is := is.New(t)
	pos := "14W/13TA/12SIG/11NOME/11ERE1/5Z5GID1/O4OD5C1V/LA1LININ4O1I/" +
		"EL3ON2ULTIMO/AA2HIT4IDYl/1A2ED8i/1POUR6V2N/4B5JEFES/1TRAYF3WAXES1/" +
		"PIET1ANCHO1TUSK AEEERRU/BGQR 281/428 0 lex CSW21;"
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	is.NoErr(err)
	configs := []struct {
		algo    string
		threads int
	}{
		{ParallelAlgoLazySMP, 2},
		{ParallelAlgoLazySMP, 4},
		{ParallelAlgoABDADA, 4},
		{ParallelAlgoTreeSplit, 4},
		{ParallelAlgoPVSplit, 4},
		{ParallelAlgoPVSplit, 16},
	}
	for _, c := range configs {
		for run := 0; run < 3; run++ {
			g, err := cgp.ParseCGP(DefaultConfig, pos)
			is.NoErr(err)
			g.SetBackupMode(game.SimulationMode)
			g.RecalculateBoard()
			gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
			s := new(Solver)
			is.NoErr(s.Init(gen, g.Game))
			s.SetTTable(&TranspositionTable{})
			s.SetTTableFraction(0.01)
			s.SetThreads(c.threads)
			is.NoErr(s.SetParallelAlgorithm(c.algo))
			v, _, err := s.Solve(context.Background(), 5)
			is.NoErr(err)
			t.Logf("%s, %d threads, run %d: %d", c.algo, c.threads, run+1, v)
			is.Equal(v, int16(41))
		}
	}
}

// A pure entry, from a search that reached only finished games, holds the
// value at the depth that search needed and every greater one, not at a
// smaller one: there some of its lines are cut off and estimated. Solving
// AEEERRU/BGQR to 11 plies (38) and then to 5 on the same table used to return
// 38 instead of the 5-ply value, 41.
func TestDeeperSolveLeavesFixedDepthValue(t *testing.T) {
	is := is.New(t)
	pos := "14W/13TA/12SIG/11NOME/11ERE1/5Z5GID1/O4OD5C1V/LA1LININ4O1I/" +
		"EL3ON2ULTIMO/AA2HIT4IDYl/1A2ED8i/1POUR6V2N/4B5JEFES/1TRAYF3WAXES1/" +
		"PIET1ANCHO1TUSK AEEERRU/BGQR 281/428 0 lex CSW21;"
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	is.NoErr(err)
	g, err := cgp.ParseCGP(DefaultConfig, pos)
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	g.RecalculateBoard()
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
	s := new(Solver)
	is.NoErr(s.Init(gen, g.Game))
	s.SetTTable(&TranspositionTable{})
	s.SetTTableFraction(0.01)
	s.SetThreads(1)
	v, _, err := s.Solve(context.Background(), 11)
	is.NoErr(err)
	is.Equal(v, int16(38))
	// Solve resets the table, so search 5 plies on the one the 11-ply solve
	// left behind.
	s.game.SetEndgameMode(true)
	err = s.iterativelyDeepen(context.Background(), 5)
	s.game.SetEndgameMode(false)
	is.NoErr(err)
	is.Equal(s.bestPVValue, int16(41))
}

// TestVeryDeep's position with many threads. Helpers stored bounds for the
// main line's positions, and a principal-variation node narrowed its window
// with them until it closed; searching on with α >= β returned bounds on the
// wrong side, stored as finished-game values. Solves came back -115 to -118.
// Principal-variation nodes no longer take bounds from the table at all.
func TestManySolveThreadsVeryDeep(t *testing.T) {
	is := is.New(t)
	deepEndgame := "14C/13QI/12FIE/10VEE1R/9KIT2G/8CIG1IDE/8UTA2AS/7ST1SYPh1/6JA5A1/5WOLD2BOBA/3PLOT1R1NU1EX/Y1VEIN1NOR1mOA1/UT1AT1N1L2FEH1/GUR2WIRER5/SNEEZED8 ADENOOO/AHIILMM 353/236 0 lex CSW19;"
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	is.NoErr(err)
	for _, algo := range []string{ParallelAlgoLazySMP, ParallelAlgoABDADA, ParallelAlgoPVSplit} {
		for run := 0; run < 2; run++ {
			g, err := cgp.ParseCGP(DefaultConfig, deepEndgame)
			is.NoErr(err)
			g.SetBackupMode(game.SimulationMode)
			g.RecalculateBoard()
			gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
			s := new(Solver)
			is.NoErr(s.Init(gen, g.Game))
			s.SetTTable(&TranspositionTable{})
			s.SetTTableFraction(0.01)
			s.SetThreads(64)
			is.NoErr(s.SetParallelAlgorithm(algo))
			v, _, err := s.Solve(context.Background(), 25)
			is.NoErr(err)
			t.Logf("%s, run %d: %d", algo, run+1, v)
			is.Equal(v, int16(-116))
			// The line must add up to the value. It did not, a few times in
			// a hundred: a principal-variation node took a lower bound from
			// the table, which left it unable to tell its best move from
			// the first one it tried.
			is.Equal(s.lineMismatches, 0)
		}
	}
}

// TestPVReachesEndOfGame is a position whose principal variation used to come
// back cut off mid-game: transposition-table hits at non-PV nodes return a
// value without its moves, so the line stopped at the ply limit with tiles on
// both racks while its value counted the rest of the game. Summing the moves
// then gave +19 against a value of +27.
func TestPVReachesEndOfGame(t *testing.T) {
	is := is.New(t)

	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW24", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../../gcgio/testdata/truncated_endgame_pv.gcg")
	is.NoErr(err)
	// p2 to move with ?FIINNT, down 393-418; p1 holds BEFLT.
	g, err := game.NewFromHistory(gameHistory, rules, 23)
	is.NoErr(err)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW24")
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())

	s := new(Solver)
	is.NoErr(s.Init(gen, g))
	// Multithreaded, as the game analyzer runs it; this is where most of the
	// truncation came from.
	s.SetThreads(8)
	solver := g.PlayerOnTurn()
	initialSpread := g.SpreadFor(solver)

	v, seq, err := s.Solve(context.Background(), 7)
	is.NoErr(err)
	is.Equal(v, int16(27))

	replay := g.Copy()
	replay.SetBackupMode(game.SimulationMode)
	replay.SetEndgameMode(true)
	for i, m := range seq {
		t.Logf("%d: %s %d  searched=%v", i+1, m.ShortDescription(), m.Score(), i < s.NumSearchedMoves())
		is.NoErr(replay.PlayMove(m, false, 0))
	}
	t.Logf("spread after line %d, playing %v", replay.SpreadFor(solver)-initialSpread, replay.Playing())
	// The line plays the game out instead of stopping at the ply limit.
	is.Equal(replay.Playing(), pb.PlayState_GAME_OVER)
	// And it adds up to the value it reports.
	is.Equal(replay.SpreadFor(solver)-initialSpread, int(v))
}

// The parallel searches need at least two plies; a one-ply solve with several
// threads used to fail with an error and return 0.
func TestOnePlySolveWithThreads(t *testing.T) {
	is := is.New(t)
	pos := "14W/13TA/12SIG/11NOME/11ERE1/5Z5GID1/O4OD5C1V/LA1LININ4O1I/" +
		"EL3ON2ULTIMO/AA2HIT4IDYl/1A2ED8i/1POUR6V2N/4B5JEFES/1TRAYF3WAXES1/" +
		"PIET1ANCHO1TUSK AEEERRU/BGQR 281/428 0 lex CSW21;"
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	is.NoErr(err)
	var values []int16
	for _, threads := range []int{1, 4} {
		for _, algo := range []string{ParallelAlgoAuto, ParallelAlgoLazySMP, ParallelAlgoABDADA, ParallelAlgoPVSplit} {
			g, err := cgp.ParseCGP(DefaultConfig, pos)
			is.NoErr(err)
			g.SetBackupMode(game.SimulationMode)
			g.RecalculateBoard()
			gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
			s := new(Solver)
			is.NoErr(s.Init(gen, g.Game))
			s.SetTTable(&TranspositionTable{})
			s.SetTTableFraction(0.01)
			s.SetThreads(threads)
			if threads > 1 {
				is.NoErr(s.SetParallelAlgorithm(algo))
			}
			v, _, err := s.Solve(context.Background(), 1)
			is.NoErr(err)
			values = append(values, v)
		}
	}
	for _, v := range values {
		is.Equal(v, values[0])
	}
	is.True(values[0] != 0)
}

// Endgames from MAGPIE's test suite (test/endgame_test.c), with the values
// MAGPIE asserts. Each is solved with one thread and with four.
func TestMagpieEndgames(t *testing.T) {
	is := is.New(t)
	viaMover := "5U4OHMIC/5N3WREATH/5T4FAX2/5i3B1VIA1/5N3L1E3/5G2VELDT2/5E3S5/" +
		"5DREKS1F3/8YELL3/4ABASER1U3/4GYM3ZO3/WAITE5OR2J/10OI2A/3QUOIT1PINNER/" +
		"4RENEGADE2P CDIOST?/AIINOOU 450/392 0 lex CSW24;"
	viaOpp := "5U4OHMIC/5N3WREATH/5T4FAX2/5i3B1VIA1/5N3L1E3/5G2VELDT2/5E3S5/" +
		"5DREKS1F3/8YELL3/4ABASER1U3/4GYM3ZO3/WAITE5OR2J/10OI2A/3QUOIT1PINNER/" +
		"4RENEGADE2P AIINOOU/CDIOST? 392/450 0 lex CSW24;"
	lexChange := "PISOLITH7/E14/r13E/I1MEW2F3H2C/ATENOLOL2NY2Z/G6A2AR1BE/U6VERBATIM/" +
		"A4UPO2EX1NA/6OR7/5WO8/4IRKSOmE1T2/5I1QUINOA2/5N1U4Y2/5G1A7/" +
		"GRICES1DELISTED ADDEJRT/FNV 536/474 0 lex "
	cases := []struct {
		name  string
		pos   string
		plies int
		want  int16
	}{
		{"incremental_movegen_identical_jmz",
			"11CLAG/9BRAAI1/11N3/11D3/10BOW2/10ERA2/6A3A1R2/C2WEDDInGS1P1I/" +
				"U1VOX1O3T3N/RHO1E1PERILUNES/FEM1A1T3Y3O/1HINT1I7U/2TO2V2GEED1L/" +
				"1AYU2ENTITLES1/QI1N3OAF1KEiR EIOS/JMZ 318/517 0 lex CSW21;", 5, 60},
		{"keep_one_tile_bound_case0",
			"OUTREmER4DZO/EPEE3E1MOVIE1/2MAInSTAY2VAN/HEEL3R2FUEL1/7A7/" +
				"5CRINGINGS1/4L2N5A1/2AXOIDS3WINY/2BUB8T1/4UH7E1/4LI7R1/3DEN7OK/" +
				"2JO1D8A/2OI10F/COTTA9S AIRT/GPQW 469/454 0 lex CSW21;", 14, 57},
		{"keep_one_tile_bound_case1",
			"15/12B2/6INTRADOS1/2RETINUE3A2/10DATO1/9KALIF1/7WAY2E2/VOE3FEM6/" +
				"2ZE1VEXILS4/1QIN2HERON4/TANTrUM2BY4/1PEE1GEO7/1I1R3PHATIC2/" +
				"1k1I4OWE4/3C1JIAO1GUANO DRSU/DGLLRS 437/397 0 lex CSW21;", 14, 44},
		{"eldar_v_stick",
			"4EXODE6/1DOFF1KERATIN1U/1OHO8YEN/1POOJA1B3MEWS/5SQUINTY2A/" +
				"4RHINO1e3V/2B4C2R3E/GOAT1D1E2ZIN1d/1URACILS2E4/1PIG1S4T4/" +
				"2L2R4T4/2L2A1GENII3/2A2T1L7/5E1A7/5D1M7 AEEIRUW/V 410/409 0 lex CSW21;", 3, 72},
		{"via_mover_must_bingo_1ply", viaMover, 1, 181},
		{"via_opp_must_block_1ply", viaOpp, 1, -78},
		{"via_opp_must_block_3ply", viaOpp, 3, -78},
		{"lexicon_CSW21", lexChange + "CSW21;", 2, 47},
		{"lexicon_NWL23", lexChange + "NWL23;", 2, 43},
	}
	for _, c := range cases {
		for _, threads := range []int{1, 4} {
			g, err := cgp.ParseCGP(DefaultConfig, c.pos)
			is.NoErr(err)
			gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), g.LexiconName())
			is.NoErr(err)
			g.SetBackupMode(game.SimulationMode)
			g.RecalculateBoard()
			gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
			s := new(Solver)
			is.NoErr(s.Init(gen, g.Game))
			s.SetTTable(&TranspositionTable{})
			s.SetTTableFraction(0.01)
			s.SetThreads(threads)
			v, _, err := s.Solve(context.Background(), c.plies)
			is.NoErr(err)
			t.Logf("%s, %d plies, %d threads: %d", c.name, c.plies, threads, v)
			is.Equal(v, c.want)
		}
	}
}
