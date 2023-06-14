package negamax

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/matryer/is"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tilemapping"
)

func TestMain(m *testing.M) {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	os.Exit(m.Run())
}

var DefaultConfig = config.DefaultConfig()

func setUpSolver(lex, distName string, bvs board.VsWho, plies int, rack1, rack2 string,
	p1pts, p2pts int, onTurn int) (*Solver, error) {

	rules, err := game.NewBasicGameRules(&DefaultConfig, lex, board.CrosswordGameLayout, distName, game.CrossScoreAndSet, game.VarClassic)
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
	g.SetStateStackLength(plies)
	// Throw in the random racks dealt to our players.
	g.ThrowRacksIn()

	gd, err := kwg.Get(g.Config(), lex)
	if err != nil {
		panic(err)
	}

	dist := rules.LetterDistribution()
	gen := movegen.NewGordonGenerator(gd, g.Board(), dist)
	gen.SetPlayRecorder(movegen.AllMinimalPlaysRecorder)
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
	runtime.SetMutexProfileFraction(5)
	plies := 14
	is := is.New(t)
	s, err := setUpSolver("NWL18", "english", board.VsJoey, plies, "DIV", "AEFILMR", 412, 371,
		1)
	is.NoErr(err)
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	s.lazySMPOptim = true
	// sweet spot seems to be threads = 3 for now
	s.threads = 3
	v, _, _ := s.Solve(context.Background(), plies)
	is.Equal(v, int16(55))
}

func TestSolveStandard(t *testing.T) {
	// This endgame is solved with at least 3 plies. Most endgames should
	// start with 3 plies (so the first player can do an out in 2) and
	// then proceed with iterative deepening.
	plies := 4

	is := is.New(t)

	s, err := setUpSolver("NWL18", "english", board.VsCanik, plies, "DEHILOR", "BGIV", 389, 384,
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
	g, err := cgp.ParseCGP(&DefaultConfig, deepEndgame)
	is.NoErr(err)
	gd, err := kwg.Get(&DefaultConfig, "CSW19")
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(plies)
	g.RecalculateBoard()
	gen := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(gen, g)
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
	g, err := cgp.ParseCGP(&DefaultConfig, pos)
	is.NoErr(err)
	gd, err := kwg.Get(&DefaultConfig, "CSW19")
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(plies)
	g.RecalculateBoard()
	gen1 := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(gen1, g)
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
	v, seq, _ := s.Solve(context.Background(), plies)

	is.Equal(v, int16(-60))
	is.Equal(seq[0].Type(), move.MoveTypePass)
	is.Equal(len(seq), 6)
}

func TestPolish(t *testing.T) {
	is := is.New(t)
	plies := 14
	s, err := setUpSolver(
		"OSPS44", "polish", board.APolishEndgame, plies, "BGHUWZZ", "IKMÓŹŻ", 304,
		258, 0)

	is.NoErr(err)
	s.earlyPassOptim = false
	v, seq, err := s.Solve(context.Background(), plies)
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
	is.Equal(len(seq), 8)

}

func TestPolishFromGcg(t *testing.T) {
	plies := 14
	is := is.New(t)

	rules, err := game.NewBasicGameRules(&DefaultConfig, "OSPS44", board.CrosswordGameLayout, "Polish", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	cfg := config.DefaultConfig()
	cfg.DefaultLexicon = "OSPS44"
	cfg.DefaultLetterDistribution = "polish"

	gameHistory, err := gcgio.ParseGCG(&cfg, "../../gcgio/testdata/polish_endgame.gcg")
	is.NoErr(err)
	gameHistory.ChallengeRule = pb.ChallengeRule_SINGLE

	g, err := game.NewFromHistory(gameHistory, rules, 46)
	is.NoErr(err)

	gd, err := kwg.Get(&DefaultConfig, "OSPS44")
	is.NoErr(err)

	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(plies)
	gen := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(gen, g)
	s.earlyPassOptim = false
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))

	v, seq, _ := s.Solve(context.Background(), plies)
	is.Equal(v, int16(5))
	is.Equal(len(seq), 8)
}

func TestStuckPruning(t *testing.T) {
	// This is very slow.
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
	g, err := cgp.ParseCGP(&DefaultConfig, deepEndgame)
	is.NoErr(err)
	gd, err := kwg.Get(&DefaultConfig, "CSW19")
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(plies)
	g.RecalculateBoard()
	gen := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(gen, g)
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
	v, _, _ := s.Solve(context.Background(), plies)
	is.Equal(v, int16(72))
}

// Test that iterative deepening actually works properly.
func TestProperIterativeDeepening(t *testing.T) {
	is := is.New(t)
	// Should get the same result with 7 or 8 plies.
	plyCount := []int{7, 8}
	rules, err := game.NewBasicGameRules(&DefaultConfig, "NWL18", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	for _, plies := range plyCount {

		gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../../gcgio/testdata/noah_vs_mishu.gcg")
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

		gd, err := kwg.Get(g.Config(), g.LexiconName())
		is.NoErr(err)

		gen := movegen.NewGordonGenerator(
			gd, g.Board(), g.Bag().LetterDistribution(),
		)
		s := new(Solver)
		s.Init(gen, g)
		fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
		// Prior to solving the endgame, set to simulation mode.
		g.SetBackupMode(game.SimulationMode)
		g.SetStateStackLength(plies)
		v, seq, _ := s.Solve(context.Background(), plies)
		is.Equal(v, int16(44))
		// In particular, the sequence should start with 6I A.
		// Player on turn needs to block the P spot. Anything else
		// shows a serious bug.
		is.Equal(len(seq), 5)
		is.Equal(seq[0].ShortDescription(), " 6I A.")
	}
}

func TestFromGCG(t *testing.T) {
	plies := 3
	is := is.New(t)

	rules, err := game.NewBasicGameRules(&DefaultConfig, "CSW19", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../../gcgio/testdata/vs_frentz.gcg")
	is.NoErr(err)

	g, err := game.NewFromHistory(gameHistory, rules, 22)
	is.NoErr(err)

	gd, err := kwg.Get(&DefaultConfig, "CSW19")
	is.NoErr(err)

	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(plies)
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
	g, err := cgp.ParseCGP(&DefaultConfig, deepEndgame)
	is.NoErr(err)
	gd, err := kwg.Get(&DefaultConfig, "CSW19")
	is.NoErr(err)
	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(plies)
	g.RecalculateBoard()
	gen := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(gen, g)
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

// func TestSolveNegamaxFunc2(t *testing.T) {
// 	plies := 11

// 	is := is.New(t)

// 	deepEndgame := "IBADAT1B7/2CAFE1OD1TRANQ/2TUT2RENIED2/3REV2YOMIM2/4RAFT1NISI2/5COR2N1x2/6LA1AGEE2/6LIAISED2/5POKY2W3/4JOWS7/V2LUZ9/ORPIN10/L1OE11/TUX12/I14 EEEEGH?/AGHNOSU 308/265 0 lex CSW19;"
// 	g, err := cgp.ParseCGP(&DefaultConfig, deepEndgame)
// 	is.NoErr(err)
// 	gd, err := kwg.Get(&DefaultConfig, "CSW19")
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
// }
