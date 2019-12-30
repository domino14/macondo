package alphabeta

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/gcgio"
	"github.com/domino14/macondo/mechanics"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
	"github.com/matryer/is"
)

var LexiconDir = os.Getenv("LEXICON_PATH")

func TestMain(m *testing.M) {
	for _, lex := range []string{"America", "NWL18", "pseudo_twl1979", "CSW19"} {
		gdgPath := filepath.Join(LexiconDir, "gaddag", lex+".gaddag")
		if _, err := os.Stat(gdgPath); os.IsNotExist(err) {
			gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, lex+".txt"), true, true)
			err = os.Rename("out.gaddag", gdgPath)
			if err != nil {
				panic(err)
			}
		}
	}
	os.Exit(m.Run())
}

func GaddagFromLexicon(lex string) (*gaddag.SimpleGaddag, error) {
	return gaddag.LoadGaddag(filepath.Join(LexiconDir, "gaddag", lex+".gaddag"))
}

func TestSolveComplex(t *testing.T) {
	t.Skip()
	plies := 8

	gd, err := GaddagFromLexicon("America")
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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.VsRoy)
	s := new(Solver)
	s.Init(generator, game)
	ourRack := alphabet.RackFromString("EFHIKOQ", alph)
	theirRack := alphabet.RackFromString("WZ", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
	game.SetPointsFor(1, 331)
	game.SetPointsFor(0, 427)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	if v != 122 {
		t.Errorf("Expected to find a 122-pt spread swing, found a swing of %v", v)
	}
}

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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.VsAlec)
	s := new(Solver)
	s.Init(generator, game)
	ourRack := alphabet.RackFromString("DGILOPR", alph)
	theirRack := alphabet.RackFromString("EGNOQR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
	game.SetPointsFor(1, 369)
	game.SetPointsFor(0, 420)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)
	t.Fail()
}

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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.VsAlec2)
	s := new(Solver)
	s.Init(generator, game)
	ourRack := alphabet.RackFromString("DGILOR", alph)
	theirRack := alphabet.RackFromString("ENQR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
	game.SetPointsFor(1, 383)
	game.SetPointsFor(0, 438)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)
	t.Fail()
}

func TestSolveOther3(t *testing.T) {
	t.Skip()
	plies := 7

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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.VsJoey)
	s := new(Solver)
	s.Init(generator, game)
	ourRack := alphabet.RackFromString("DIV", alph)
	theirRack := alphabet.RackFromString("AEFILMR", alph)
	game.SetRackFor(0, ourRack)
	game.SetRackFor(1, theirRack)
	game.SetPointsFor(0, 412)
	game.SetPointsFor(1, 371)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)
	if v < 0 {
		t.Errorf("Expected > 0, %v was", v)
	}
}

func TestSolveStandard(t *testing.T) {
	// This endgame is solved with at least 3 plies. Most endgames should
	// start with 3 plies (so the first player can do an out in 2) and
	// then proceed with iterative deepening.
	plies := 4

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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.VsCanik)
	s := new(Solver)
	s.Init(generator, game)
	ourRack := alphabet.RackFromString("BGIV", alph)
	theirRack := alphabet.RackFromString("DEHILOR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
	game.SetPointsFor(1, 384)
	game.SetPointsFor(0, 389)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)

	fmt.Println("Value found", v)
	if v != 11 {
		t.Errorf("Expected 11, was %v", v)
	}
}

func TestSolveStandard2(t *testing.T) {
	// Another standard 3-ply endgame.
	plies := 3

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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.VsJoel)
	s := new(Solver)
	s.Init(generator, game)
	ourRack := alphabet.RackFromString("AAFIRTW", alph)
	theirRack := alphabet.RackFromString("EIQSS", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
	game.SetPointsFor(1, 373)
	game.SetPointsFor(0, 393)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	fmt.Println("Value found", v)
	if v < 0 {
		t.Errorf("Expected > 0, %v was", v)
	}
}

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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.JoeVsPaul)
	s := new(Solver)
	s.Init(generator, game)
	ourRack := alphabet.RackFromString("?AEIR", alph)
	theirRack := alphabet.RackFromString("LZ", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
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
}

func TestStuck(t *testing.T) {
	is := is.New(t)
	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)

	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		game, &strategy.NoLeaveStrategy{},
	)
	alph := game.Alphabet()
	generator.SetBoardToGame(alph, board.VsAlec)

	s := new(Solver)
	s.Init(generator, game)
	ourRack := alphabet.RackFromString("DGILOPR", alph)
	theirRack := alphabet.RackFromString("EGNOQR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
	game.SetPointsFor(1, 369)
	game.SetPointsFor(0, 420)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)
	s.clearStuckTables()
	s.movegen.GenAll(ourRack)
	stmPlays := s.movegen.Plays()
	s.movegen.GenAll(theirRack)
	otsPlays := s.movegen.Plays()
	stmStuck := s.computeStuck(stmPlays, ourRack, s.stmPlayed)
	otsStuck := s.computeStuck(otsPlays, theirRack, s.otsPlayed)
	is.Equal(len(stmStuck), 0)
	is.Equal(len(otsStuck), 1)
	is.Equal(otsStuck[0].UserVisible(alph), 'Q')
}

func TestValuation(t *testing.T) {
	is := is.New(t)
	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)

	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		game, &strategy.NoLeaveStrategy{},
	)
	alph := game.Alphabet()
	generator.SetBoardToGame(alph, board.VsAlec)

	s := new(Solver)
	s.Init(generator, game)
	ourRack := alphabet.RackFromString("DGILOPR", alph)
	theirRack := alphabet.RackFromString("EGNOQR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
	game.SetPointsFor(1, 369)
	game.SetPointsFor(0, 420)
	game.SetPlayerOnTurn(1)
	game.SetPlaying(true)

	plays := s.generateSTMPlays()
	// This is subject to change depending on the C & D values, but
	// it's roughly accurate
	is.Equal(plays[0].Valuation(), float32(36.5))
	is.Equal(plays[0].Tiles().UserVisible(alph), "DO..R.")
}

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

func (t *TestGenerator) SetSortingParameter(s movegen.SortBy) { /* noop */ }

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
	ourRack := alphabet.RackFromString("BGIV", alph)
	theirRack := alphabet.RackFromString("DEHILOR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
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
	ourRack := alphabet.RackFromString("BGIV", alph)
	theirRack := alphabet.RackFromString("DEHILOR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
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
	ourRack := alphabet.RackFromString("BGIV", alph)
	theirRack := alphabet.RackFromString("DEHILOR", alph)
	game.SetRackFor(0, ourRack)
	game.SetRackFor(1, theirRack)
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
	ourRack := alphabet.RackFromString("BGIV", alph)
	theirRack := alphabet.RackFromString("DEHILOR", alph)
	game.SetRackFor(1, ourRack)
	game.SetRackFor(0, theirRack)
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

func TestAnotherOneTiler(t *testing.T) {
	// t.Skip()
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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.EldarVsNigel)
	s := new(Solver)
	s.Init(generator, game)
	// s.iterativeDeepeningOn = false
	// s.simpleEvaluation = true
	ourRack := alphabet.RackFromString("AEEIRUW", alph)
	theirRack := alphabet.RackFromString("V", alph)
	game.SetRackFor(0, ourRack)
	game.SetRackFor(1, theirRack)
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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.NoahVsMishu)
	s := new(Solver)
	s.Init(generator, game)
	// s.iterativeDeepeningOn = false
	// s.simpleEvaluation = true
	ourRack := alphabet.RackFromString("AEIINTY", alph)
	theirRack := alphabet.RackFromString("CLLPR", alph)
	game.SetRackFor(0, ourRack)
	game.SetRackFor(1, theirRack)
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

func TestYetAnotherOneTiler2(t *testing.T) {
	// t.Skip()
	plies := 7

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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.NoahVsMishu2)
	s := new(Solver)
	s.Init(generator, game)
	// s.disablePruning = true
	s.iterativeDeepeningOn = false
	// s.simpleEvaluation = true
	ourRack := alphabet.RackFromString("AEIINY", alph)
	theirRack := alphabet.RackFromString("LLPR", alph)
	game.SetRackFor(0, ourRack)
	game.SetRackFor(1, theirRack)
	game.SetPointsFor(0, 334)
	game.SetPointsFor(1, 374)
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

func TestYetAnotherOneTiler3(t *testing.T) {
	t.Skip()
	plies := 6

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
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.NoahVsMishu3)
	s := new(Solver)
	s.Init(generator, game)
	// s.iterativeDeepeningOn = false
	// s.simpleEvaluation = true
	ourRack := alphabet.RackFromString("AEIY", alph)
	theirRack := alphabet.RackFromString("LLP", alph)
	game.SetRackFor(0, ourRack)
	game.SetRackFor(1, theirRack)
	game.SetPointsFor(0, 339)
	game.SetPointsFor(1, 381)
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

func TestFromGCG(t *testing.T) {
	plies := 1

	curGameRepr, err := gcgio.ParseGCG("../../gcgio/testdata/vs_frentz.gcg")
	if err != nil {
		t.Errorf("Got error %v", err)
	}
	game := mechanics.StateFromRepr(curGameRepr, "CSW19", 0)
	game.SetStateStackLength(plies)
	err = game.PlayGameToTurn(curGameRepr, 21)
	if err != nil {
		t.Errorf("Error playing to turn %v", err)
	}
	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		game, &strategy.NoLeaveStrategy{},
	)

	s := new(Solver)
	s.Init(generator, game)
	// s.iterativeDeepeningOn = false
	// s.simpleEvaluation = true
	fmt.Println(game.Board().ToDisplayText(game.Alphabet()))
	v, _ := s.Solve(plies)
	if v != 99 {
		t.Errorf("Expected 99, was %v", v)
	}
	// t.Fail()
}
