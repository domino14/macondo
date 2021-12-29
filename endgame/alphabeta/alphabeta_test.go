package alphabeta

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/matryer/is"
	"github.com/rs/zerolog"

	airunner "github.com/domino14/macondo/ai/runner"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	"github.com/domino14/macondo/movegen"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	for _, lex := range []string{"America", "NWL18", "pseudo_twl1979", "CSW19", "OSPS44"} {
		gdgPath := filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag")
		if _, err := os.Stat(gdgPath); os.IsNotExist(err) {
			gaddagmaker.GenerateGaddag(filepath.Join(DefaultConfig.LexiconPath, lex+".txt"), true, true)
			err = os.Rename("out.gaddag", gdgPath)
			if err != nil {
				panic(err)
			}
		}
	}
	os.Exit(m.Run())
}

func setUpSolver(lex, distName string, bvs board.VsWho, plies int, rack1, rack2 string,
	p1pts, p2pts int, onTurn int) (*Solver, error) {

	rules, err := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout,
		lex, distName)

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

	gd, err := gaddag.Get(g.Config(), lex)
	if err != nil {
		panic(err)
	}

	dist := rules.LetterDistribution()
	generator := movegen.NewGordonGenerator(gd, g.Board(), dist)
	alph := g.Alphabet()

	tilesInPlay := g.Board().SetToGame(alph, bvs)
	err = g.Bag().RemoveTiles(tilesInPlay.OnBoard)
	if err != nil {
		panic(err)
	}
	cross_set.GenAllCrossSets(g.Board(), gd, dist)

	g.SetRacksForBoth([]*alphabet.Rack{
		alphabet.RackFromString(rack1, alph),
		alphabet.RackFromString(rack2, alph),
	})
	g.SetPointsFor(0, p1pts)
	g.SetPointsFor(1, p2pts)
	g.SetPlayerOnTurn(onTurn)
	fmt.Println(g.Board().ToDisplayText(alph))

	s := new(Solver)
	s.Init(generator, g)
	return s, nil
}

func TestSolveComplex(t *testing.T) {
	t.Skip()
	is := is.New(t)
	plies := 8

	s, err := setUpSolver("America", "english", board.VsRoy, plies, "WZ", "EFHIKOQ", 427, 331,
		1)
	is.NoErr(err)

	v, _, _ := s.Solve(plies)
	is.Equal(v, 116)
	// Quackle finds a 122-pt win. However, I think it's wrong because it
	// doesn't take into account that opp can pass to prevent a setup
	// (the setup being: EF 3F to block the Z, then YO, YOK/KHI, QI)
	// The setup only works if Roy plays off his W before YO.
}

/*
func TestSolveOther(t *testing.T) {
	// This is an extremely complex endgame; 8 plies takes over an hour
	// and consumes > 30 GB of memory. It seems to be a loss no matter what.
	t.Skip()
	plies := 8

	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	game.SetStateStackLength(plies)

	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		game, &strategy.NoLeaveStrategy{},
	)
	alph := game.Alphabet()
	ourRack := alphabet.RackFromString("DGILOPR", alph)
	theirRack := alphabet.RackFromString("EGNOQR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)

	generator.SetBoardToGame(alph, board.VsAlec)
	s := new(Solver)
	s.Init(generator, game)

	game.SetPointsFor(1, 369)
	game.SetPointsFor(0, 420)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)
	t.Fail()
} */

/*
func TestSolveOther2(t *testing.T) {
	// An attempt to solve the game from above after a turn in. It's still
	// a loss; this goes a bit faster.
	t.Skip()
	plies := 8

	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	game.SetStateStackLength(plies)

	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		game, &strategy.NoLeaveStrategy{},
	)
	alph := game.Alphabet()
	ourRack := alphabet.RackFromString("DGILOR", alph)
	theirRack := alphabet.RackFromString("ENQR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
	generator.SetBoardToGame(alph, board.VsAlec2)
	s := new(Solver)
	s.Init(generator, game)

	game.SetPointsFor(1, 383)
	game.SetPointsFor(0, 438)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)
	t.Fail()
}
*/

func TestSolveOther3(t *testing.T) {
	t.Skip()
	plies := 7
	is := is.New(t)
	s, err := setUpSolver("NWL18", "english", board.VsJoey, plies, "DIV", "AEFILMR", 412, 371,
		1)
	is.NoErr(err)

	v, _, _ := s.Solve(plies)
	is.True(v > 0)
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
	v, _, _ := s.Solve(plies)

	is.Equal(v, float32(11))
}

func TestSolveStandard2(t *testing.T) {
	// Another standard 3-ply endgame.
	is := is.New(t)
	plies := 3

	s, err := setUpSolver("NWL18", "english", board.VsJoel, plies, "EIQSS", "AAFIRTW", 393, 373,
		1)
	is.NoErr(err)

	v, _, _ := s.Solve(plies)
	is.Equal(v, float32(25))
}

func TestPolish(t *testing.T) {
	is := is.New(t)
	plies := 14
	s, err := setUpSolver(
		"OSPS44", "polish", board.APolishEndgame, plies, "BGHUWZZ", "IKMÓŹŻ", 304,
		258, 0)

	is.NoErr(err)
	v, seq, err := s.Solve(plies)
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

	is.Equal(v, float32(5))
	is.Equal(len(seq), 8)

}

func TestPolishFromGcg(t *testing.T) {
	plies := 14
	is := is.New(t)

	rules, err := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout,
		"OSPS44", "Polish")
	is.NoErr(err)

	cfg := config.DefaultConfig()
	cfg.DefaultLexicon = "OSPS44"
	cfg.DefaultLetterDistribution = "polish"

	gameHistory, err := gcgio.ParseGCG(&cfg, "../../gcgio/testdata/polish_endgame.gcg")
	is.NoErr(err)
	gameHistory.ChallengeRule = pb.ChallengeRule_SINGLE

	g, err := game.NewFromHistory(gameHistory, rules, 46)
	is.NoErr(err)

	gd, err := gaddag.Get(&DefaultConfig, "OSPS44")
	is.NoErr(err)

	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(plies)
	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(generator, g)
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))

	v, seq, _ := s.Solve(plies)
	is.Equal(v, float32(5))
	is.Equal(len(seq), 8)
}

func TestSpuriousPasses(t *testing.T) {
	// This position is 1 move after the position in TestPolish. It should
	// essentially be moves 2-7 of that sequence.
	is := is.New(t)
	plies := 14
	s, err := setUpSolver(
		"OSPS44", "polish", board.APolishEndgame2, plies, "BHUWZ", "IKMÓŹŻ", 316,
		258, 1)

	is.NoErr(err)
	v, seq, err := s.Solve(plies)
	is.NoErr(err)

	/* optimal endgame should look like this:
	   1) M1 ŻM..
	   2) (Pass)
	   3) 6L .I
	   4) B8 ZU.
	   5) 9A K.
	   6) (Pass)
	   7) (Pass)
	*/

	is.Equal(v, float32(7))
	is.Equal(len(seq), 7)
}

func TestSpuriousPassesFromGcg(t *testing.T) {
	plies := 14
	is := is.New(t)

	rules, err := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout,
		"OSPS44", "Polish")
	is.NoErr(err)

	cfg := config.DefaultConfig()
	cfg.DefaultLexicon = "OSPS44"
	cfg.DefaultLetterDistribution = "polish"

	gameHistory, err := gcgio.ParseGCG(&cfg, "../../gcgio/testdata/polish_endgame.gcg")
	is.NoErr(err)
	gameHistory.ChallengeRule = pb.ChallengeRule_SINGLE

	g, err := game.NewFromHistory(gameHistory, rules, 47)
	is.NoErr(err)

	gd, err := gaddag.Get(&DefaultConfig, "OSPS44")
	is.NoErr(err)

	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(plies)
	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(generator, g)
	// s.iterativeDeepeningOn = false
	// s.simpleEvaluation = true
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))

	v, seq, _ := s.Solve(plies)
	is.Equal(v, float32(7))
	is.Equal(len(seq), 7)
}

/*
func TestSolveMaven(t *testing.T) {
	// This endgame is the one in maven. Start by pre-playing TSK as
	// they suggest.
	t.Skip()
	plies := 9

	gd, err := GaddagFromLexicon("pseudo_twl1979")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	game.SetStateStackLength(plies)

	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		game, &strategy.NoLeaveStrategy{},
	)
	alph := game.Alphabet()
	ourRack := alphabet.RackFromString("?AEIR", alph)
	theirRack := alphabet.RackFromString("LZ", alph)

	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)

	generator.SetBoardToGame(alph, board.JoeVsPaul)
	s := new(Solver)
	s.Init(generator, game)

	game.SetPointsFor(1, 300)
	game.SetPointsFor(0, 300)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)
	// if v < 0 {
	// 	t.Errorf("Expected > 0, %v was", v)
	// }
	t.Fail()
} */

func TestStuck(t *testing.T) {
	is := is.New(t)

	s, err := setUpSolver("NWL18", "english", board.VsAlec, 0, "EGNOQR", "DGILOPR", 420, 369,
		1)
	is.NoErr(err)
	alph := s.game.Alphabet()
	ourRack := alphabet.RackFromString("DGILOPR", alph)
	theirRack := alphabet.RackFromString("EGNOQR", alph)
	s.clearStuckTables()
	s.movegen.GenAll(ourRack, false)
	stmPlays := s.movegen.Plays()
	s.movegen.GenAll(theirRack, false)
	otsPlays := s.movegen.Plays()
	stmStuck := s.computeStuck(stmPlays, ourRack, s.stmPlayed)
	otsStuck := s.computeStuck(otsPlays, theirRack, s.otsPlayed)
	is.Equal(len(stmStuck), 0)
	is.Equal(len(otsStuck), 1)
	is.Equal(otsStuck[0].UserVisible(alph), 'Q')
}

func TestValuation(t *testing.T) {
	is := is.New(t)

	s, err := setUpSolver("NWL18", "english", board.VsAlec, 0, "EGNOQR", "DGILOPR",
		420, 369, 1)
	is.NoErr(err)

	plays := s.generateSTMPlays(nil)
	// This is subject to change depending on the C & D values, but
	// it's roughly accurate
	alph := s.game.Alphabet()
	is.Equal(plays[0].Valuation(), float32(36.5))
	is.Equal(plays[0].Tiles().UserVisible(alph), "DO..R.")
}

/*

type TestGenerator struct {
	plays []*move.Move
	alph  *alphabet.Alphabet
}

func (t *TestGenerator) GenAll(rack *alphabet.Rack) {
	if rack.String() == "BGIV" {
		t.plays = []*move.Move{
			move.NewScoringMoveSimple(24, "1G", "VIG.", "B", t.alph),
			move.NewScoringMoveSimple(9, "G7", "B...", "GIV", t.alph),
			move.NewScoringMoveSimple(18, "2B", "VIG", "B", t.alph),
			move.NewScoringMoveSimple(9, "13K", "B..", "GIV", t.alph),
		}
	} else if rack.String() == "DEHILOR" {
		t.plays = []*move.Move{
			move.NewScoringMoveSimple(33, "2J", ".O.DI", "EHLR", t.alph),
			move.NewScoringMoveSimple(30, "4A", "HOER", "DIL", t.alph),
			move.NewScoringMoveSimple(25, "K7", "DHOLE", "IR", t.alph),
			move.NewScoringMoveSimple(21, "1G", "HIL.", "DEOR", t.alph),
		}
	} else if rack.String() == "B" {
		t.plays = []*move.Move{
			move.NewScoringMoveSimple(9, "G7", "B...", "", t.alph),
			move.NewScoringMoveSimple(4, "G5", ".B", "", t.alph),
			move.NewScoringMoveSimple(9, "13K", "B..", "", t.alph),
		}
	} else if rack.String() == "GIV" {
		t.plays = []*move.Move{
			move.NewScoringMoveSimple(24, "1G", "VIG.", "", t.alph),
			move.NewScoringMoveSimple(18, "2B", "VIG", "", t.alph),
		}
	} else if rack.String() == "DIL" {
		t.plays = []*move.Move{
			move.NewScoringMoveSimple(21, "A13", ".ID", "L", t.alph),
			move.NewScoringMoveSimple(12, "2B", "LID", "", t.alph),
		}
	} else if rack.String() == "EHLR" {
		t.plays = []*move.Move{
			move.NewScoringMoveSimple(27, "A13", ".EH", "LR", t.alph),
			move.NewScoringMoveSimple(23, "N14", "HE", "LR", t.alph),
		}
	} else if rack.String() == "IR" {
		t.plays = []*move.Move{
			move.NewScoringMoveSimple(18, "A13", ".IR", "", t.alph),
		}
	} else if rack.String() == "DEOR" {
		t.plays = []*move.Move{
			move.NewScoringMoveSimple(8, "J10", "ORDE.", "", t.alph),
		}
	} else {
		panic("Unexpected rack: " + rack.String())
	}
}

func (t *TestGenerator) SetSortingParameter(s movegen.SortBy) {
	// NOOP
 }

func (t *TestGenerator) Plays() []*move.Move {
	return t.plays
}

func (t *TestGenerator) Reset() {}

func (t *TestGenerator) SetOppRack(rack *alphabet.Rack) {}

func TestMinimalCase(t *testing.T) {
	plies := 2

	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	game.SetStateStackLength(plies)
	alph := game.Alphabet()

	generator := &TestGenerator{alph: alph}
	ourRack := alphabet.RackFromString("BGIV", alph)
	theirRack := alphabet.RackFromString("DEHILOR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	tilesPlayedAndInRacks := game.Board().SetToGame(alph, board.VsCanik)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.OnBoard)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.Rack1)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.Rack2)

	s := new(Solver)
	s.Init(generator, game)

	game.SetPointsFor(1, 294) // was 384
	game.SetPointsFor(0, 389)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)
	dot := &dotfile{}
	genDotFile(s.rootNode, dot)
	saveDotFile(s.rootNode, dot, "out.dot")

	if v < 0 {
		t.Errorf("Expected > 0, %v was", v)
	}
}

func TestMinimalCase2(t *testing.T) {
	// 3 instead of two plies. it should find the endgame
	plies := 3

	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	game.SetStateStackLength(plies)
	alph := game.Alphabet()

	generator := &TestGenerator{alph: alph}
	ourRack := alphabet.RackFromString("BGIV", alph)
	theirRack := alphabet.RackFromString("DEHILOR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)

	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	tilesPlayedAndInRacks := game.Board().SetToGame(alph, board.VsCanik)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.OnBoard)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.Rack1)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.Rack2)

	s := new(Solver)
	s.Init(generator, game)

	game.SetPointsFor(1, 384)
	game.SetPointsFor(0, 389)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)

	// dot := &dotfile{}
	// genDotFile(s.rootNode, dot)
	// saveDotFile(s.rootNode, dot, "out.dot")

	if v != 11 {
		t.Errorf("Expected 11-pt spread swing, found %v", v)
	}

}

func TestMinimalCase3(t *testing.T) {
	// Identical to case 2 but the players are flipped. This should
	// make absolutely no difference.
	plies := 3

	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	game.SetStateStackLength(plies)
	alph := game.Alphabet()
	ourRack := alphabet.RackFromString("BGIV", alph)
	theirRack := alphabet.RackFromString("DEHILOR", alph)
	game.SetRackFor(0, ourRack)
	game.SetRackFor(1, theirRack)
	generator := &TestGenerator{alph: alph}
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	tilesPlayedAndInRacks := game.Board().SetToGame(alph, board.VsCanik)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.OnBoard)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.Rack1)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.Rack2)

	s := new(Solver)
	s.Init(generator, game)

	game.SetPointsFor(0, 384)
	game.SetPointsFor(1, 389)
	game.SetPlayerOnTurn(0)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)

	// dot := &dotfile{}
	// genDotFile(s.rootNode, dot)
	// saveDotFile(s.rootNode, dot, "out.dot")

	if v != 11 {
		t.Errorf("Expected 11-pt spread swing, found %v", v)
	}
}

func TestMinimalCase4(t *testing.T) {
	// 5 plies; should still find the correct endgame. Note that this uses
	// iterative deepening by default.
	plies := 5

	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	game.SetStateStackLength(plies)
	alph := game.Alphabet()
	ourRack := alphabet.RackFromString("BGIV", alph)
	theirRack := alphabet.RackFromString("DEHILOR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
	generator := &TestGenerator{alph: alph}
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	tilesPlayedAndInRacks := game.Board().SetToGame(alph, board.VsCanik)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.OnBoard)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.Rack1)
	game.Bag().RemoveTiles(tilesPlayedAndInRacks.Rack2)

	s := new(Solver)
	s.Init(generator, game)

	game.SetPointsFor(1, 384)
	game.SetPointsFor(0, 389)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)

	// dot := &dotfile{}
	// genDotFile(s.rootNode, dot)
	// saveDotFile(s.rootNode, dot, "out.dot")

	if v != 11 {
		t.Errorf("Expected 11-pt spread swing, found %v", v)
	}

}
*/

/*
func TestAnotherOneTiler(t *testing.T) {
	t.Skip() // for now. Quackle actually finds a better endgame play, but
	// I might need to let this run all night.
	plies := 5 // why is quackle so much faster at this endgame?

	gd, err := GaddagFromLexicon("CSW19")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	game.SetStateStackLength(plies)

	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		game, &strategy.NoLeaveStrategy{},
	)
	alph := game.Alphabet()
	ourRack := alphabet.RackFromString("AEEIRUW", alph)
	theirRack := alphabet.RackFromString("V", alph)
	game.SetRackFor(0, ourRack)
	game.SetRackFor(1, theirRack)
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.EldarVsNigel)
	s := new(Solver)
	s.Init(generator, game)
	// s.iterativeDeepeningOn = false
	// s.simpleEvaluation = true

	game.SetPointsFor(0, 410)
	game.SetPointsFor(1, 409)
	game.SetPlayerOnTurn(0)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)
	// if v < 0 {
	// 	t.Errorf("Expected > 0, %v was", v)
	// }
	// t.Fail()
}
*/

/*

func TestYetAnotherOneTiler(t *testing.T) {
	t.Skip()
	plies := 10

	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	game.SetStateStackLength(plies)

	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		game, &strategy.NoLeaveStrategy{},
	)
	alph := game.Alphabet()

	ourRack := alphabet.RackFromString("AEIINTY", alph)
	theirRack := alphabet.RackFromString("CLLPR", alph)
	game.SetRackFor(0, ourRack)
	game.SetRackFor(1, theirRack)
	generator.SetBoardToGame(alph, board.NoahVsMishu)
	s := new(Solver)
	s.Init(generator, game)

	game.SetPointsFor(0, 327)
	game.SetPointsFor(1, 368)
	game.SetPlayerOnTurn(0)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)
	// if v < 0 {
	// 	t.Errorf("Expected > 0, %v was", v)
	// }
	t.Fail()
}
*/

// Test that iterative deepening actually works properly.
func TestProperIterativeDeepening(t *testing.T) {
	is := is.New(t)
	// Should get the same result with 7 or 8 plies.
	plyCount := []int{7, 8}
	rules, err := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout,
		"NWL18", "English")
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

		gd, err := gaddag.Get(g.Config(), g.LexiconName())
		is.NoErr(err)

		generator := movegen.NewGordonGenerator(
			gd, g.Board(), g.Bag().LetterDistribution(),
		)
		s := new(Solver)
		s.Init(generator, g)
		fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
		// Prior to solving the endgame, set to simulation mode.
		g.SetBackupMode(game.SimulationMode)
		g.SetStateStackLength(plies)
		v, seq, _ := s.Solve(plies)
		is.Equal(v, float32(44))
		// In particular, the sequence should start with 6I A.
		// Player on turn needs to block the P spot. Anything else
		// shows a serious bug.
		is.Equal(len(seq), 5)
		is.Equal(seq[0].ShortDescription(), "6I A.")
	}
}

func BenchmarkID(b *testing.B) {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	is := is.New(b)
	rules, err := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout,
		"NWL18", "English")
	is.NoErr(err)
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

	gd, err := gaddag.Get(g.Config(), g.LexiconName())
	is.NoErr(err)

	generator := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {

		s := new(Solver)
		s.Init(generator, g)
		plies := 5

		g.SetBackupMode(game.SimulationMode)
		g.SetStateStackLength(plies)

		v, seq, _ := s.Solve(plies)
		is.Equal(v, float32(44))
		is.Equal(len(seq), 5)
		is.Equal(seq[0].ShortDescription(), "6I A.")
	}
}

func BenchmarkID2(b *testing.B) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	is := is.New(b)
	rules, err := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout,
		"OSPS44", "Polish")
	is.NoErr(err)

	cfg := config.DefaultConfig()
	cfg.DefaultLexicon = "OSPS44"
	cfg.DefaultLetterDistribution = "polish"

	gameHistory, err := gcgio.ParseGCG(&cfg, "../../gcgio/testdata/polish_endgame.gcg")
	is.NoErr(err)

	gameHistory.ChallengeRule = pb.ChallengeRule_SINGLE

	g, err := game.NewFromHistory(gameHistory, rules, 47)
	is.NoErr(err)

	gd, err := gaddag.Get(g.Config(), g.LexiconName())
	is.NoErr(err)

	generator := movegen.NewGordonGenerator(
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {

		s := new(Solver)
		s.Init(generator, g)
		plies := 11

		g.SetBackupMode(game.SimulationMode)
		g.SetStateStackLength(plies)

		v, seq, _ := s.Solve(plies)
		is.Equal(v, float32(7))
		is.Equal(len(seq), 7)
	}
}

func TestFromGCG(t *testing.T) {
	plies := 3
	is := is.New(t)

	rules, err := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout,
		"CSW19", "English")
	is.NoErr(err)

	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../../gcgio/testdata/vs_frentz.gcg")
	is.NoErr(err)

	g, err := game.NewFromHistory(gameHistory, rules, 22)
	is.NoErr(err)

	gd, err := gaddag.Get(&DefaultConfig, "CSW19")
	is.NoErr(err)

	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(plies)
	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		gd, g.Board(), g.Bag().LetterDistribution(),
	)

	s := new(Solver)
	s.Init(generator, g)
	// s.iterativeDeepeningOn = false
	// s.simpleEvaluation = true
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
	v, seq, _ := s.Solve(plies)
	is.Equal(v, float32(99))
	is.Equal(len(seq), 1)
	// t.Fail()
}

/*
// Test iterative deepening on a game, using a very minimal dictionary.
// This is written for debug purposes because I can't figure out wtf is
// wrong with my code.
// func TestMinimalIterativeDeepening(t *testing.T) {
// 	//t.Skip()
// 	plies := 8
// 	// Basically ignore the first two "words", they are only here so that
// 	// the bag initializer doesn't complain about missing letters.
// 	reducedDict := `BCFGJKMNO
// QUVWXZHT
// AI
// AS
// ED
// ES
// LA
// LAY
// LEY
// LI
// PI
// RYE
// TI`

// 	gd := gaddag.GaddagToSimpleGaddag(
// 		gaddagmaker.GenerateGaddagFromStream(strings.NewReader(reducedDict), "TESTENG"))

// 	curGameRepr, err := gcgio.ParseGCG("../../gcgio/testdata/noah_vs_mishu.gcg")
// 	if err != nil {
// 		t.Errorf("Got error %v", err)
// 	}
// 	game := mechanics.StateFromRepr(curGameRepr, "NWL18", 0)
// 	// Use the gaddag we just created, instead of "NWL18":
// 	fmt.Println("Replacing gaddag")
// 	game.Init(gd, alphabet.EnglishLetterDistribution())
// 	game.SetStateStackLength(plies)
// 	// Make a few plays:
// 	mechanics.AppendScoringMoveAt(game, curGameRepr, 28, "H7", "T...")
// 	mechanics.AppendScoringMoveAt(game, curGameRepr, 29, "N5", "C...")
// 	mechanics.AppendScoringMoveAt(game, curGameRepr, 30, "10A", ".IN")
// 	// Note that this is not right; user should play the P off at 6I,
// 	// but this is for testing purposes only:
// 	mechanics.AppendScoringMoveAt(game, curGameRepr, 31, "13L", "...R")

// 	err = game.PlayGameToTurn(curGameRepr, 32)
// 	if err != nil {
// 		t.Errorf("Error playing to turn %v", err)
// 	}

// 	if game.PointsFor(0) != 339 {
// 		t.Errorf("Points wrong: %v", game.PointsFor(0))
// 	}
// 	if game.PointsFor(1) != 381 {
// 		t.Errorf("Points wrong: %v", game.PointsFor(1))
// 	}

// 	generator := movegen.NewGordonGenerator(
// 		// The strategy doesn't matter right here
// 		game, &strategy.NoLeaveStrategy{},
// 	)
// 	s := new(Solver)
// 	s.Init(generator, game)
// 	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
// 	v, seq := s.Solve(plies)
// 	if v != 44 {
// 		t.Errorf("Spread is wrong: %v", v)
// 	}
// 	if len(seq) != 5 {
// 		// In particular, the sequence should start with 6I A.
// 		// Player on turn needs to block the P spot. Anything else
// 		// shows a serious bug.
// 		t.Errorf("Sequence is wrong: %v", seq)
// 	}
// 	dot := &dotfile{}
// 	genDotFile(s.rootNode, dot)
// 	saveDotFile(s.rootNode, dot, "out.dot")
// }
*/
