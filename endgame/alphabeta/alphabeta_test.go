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
	"github.com/domino14/macondo/mechanics"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
	"github.com/matryer/is"
)

var LexiconDir = os.Getenv("LEXICON_PATH")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/gen_america.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "America.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/gen_america.gaddag")
	}
	if _, err := os.Stat("/tmp/nwl18.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "NWL18.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/nwl18.gaddag")
	}
	if _, err := os.Stat("/tmp/ospd1.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "pseudo_twl1979.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/ospd1.gaddag")
	}
	os.Exit(m.Run())
}

func TestSolveComplex(t *testing.T) {
	plies := 5

	gd, err := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
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
	if v != 6 {
		t.Errorf("Expected to find a 6-pt win, found a spread of %v", v)
	}
}

func TestSolveOther(t *testing.T) {
	// This is an extremely complex endgame; 8 plies takes over an hour
	// and consumes > 30 GB of memory. It seems to be a loss no matter what.
	t.Skip()
	plies := 8

	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
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
	plies := 8

	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
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
	// This endgame seems to require >= 7 plies to solve. Otherwise it loses.
	// t.Skip()
	plies := 7

	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
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
	t.Fail()
}

func TestSolveStandard(t *testing.T) {
	// This endgame is solved with at least 3 plies. Most endgames should
	// start with 3 plies (so the first player can do an out in 2) and
	// then proceed with iterative deepening.
	plies := 3

	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
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
	if v < 0 {
		t.Errorf("Expected > 0, %v was", v)
	}
}

func TestSolveStandard2(t *testing.T) {
	// Another standard 3-ply endgame.
	plies := 3

	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
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
	generator.SetBoardToGame(alph, 	board.VsJoel)
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
	t.Fail()
}

func TestSolveMaven(t *testing.T) {
	// This endgame is the one in maven. Start by pre-playing TSK as
	// they suggest.
	t.Skip()
	plies := 9

	gd, err := gaddag.LoadGaddag("/tmp/ospd1.gaddag")
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
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
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
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
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
	is.Equal(plays[0].Valuation(), 36)
	is.Equal(plays[0].Tiles().UserVisible(alph), "DO..R.")
}
