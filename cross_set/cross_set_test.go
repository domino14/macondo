package cross_set

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/move"
	"github.com/stretchr/testify/assert"
)

var DefaultConfig = config.DefaultConfig()

const (
	VsEd     = board.VsEd
	VsJeremy = board.VsJeremy
	VsMatt   = board.VsMatt
	VsMatt2  = board.VsMatt2
	VsOxy    = board.VsOxy
)

func GaddagFromLexicon(lex string) (*gaddag.SimpleGaddag, error) {
	return gaddag.LoadGaddag(filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag"))
}

type crossSetTestCase struct {
	row      int
	col      int
	crossSet CrossSet
	dir      board.BoardDirection
	score    int
}

func TestMain(m *testing.M) {
	for _, lex := range []string{"America", "NWL18"} {
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

func TestGenCrossSetLoadedGame(t *testing.T) {
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	if err != nil {
		t.Error(err)
	}
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	alph := dist.Alphabet()

	b := board.MakeBoard(board.CrosswordGameBoard)
	bcs := MakeBoardCrossSets(b)
	b.SetToGame(alph, VsMatt)
	fmt.Println(b.ToDisplayText(alph))
	// All horizontal for now.
	var testCases = []crossSetTestCase{
		{10, 10, CrossSetFromString("E", alph), board.HorizontalDirection, 11},
		{2, 4, CrossSetFromString("DKHLRSV", alph), board.HorizontalDirection, 9},
		{8, 7, CrossSetFromString("S", alph), board.HorizontalDirection, 11},
		// suffix - no hooks:
		{12, 8, CrossSet(0), board.HorizontalDirection, 11},
		// prefix - no hooks:
		{3, 1, CrossSet(0), board.HorizontalDirection, 10},
		// prefix and suffix, no path
		{6, 8, CrossSet(0), board.HorizontalDirection, 5},
		// More in-between
		{2, 10, CrossSetFromString("M", alph), board.HorizontalDirection, 2},
	}

	for _, tc := range testCases {
		GenCrossSet(b, bcs, tc.row, tc.col, tc.dir, gd, dist)
		if bcs.Get(tc.row, tc.col, board.HorizontalDirection) != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				tc.row, tc.col, tc.crossSet,
				bcs.Get(tc.row, tc.col, board.HorizontalDirection))
		}
		if b.GetCrossScore(tc.row, tc.col, board.HorizontalDirection) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				b.GetCrossScore(tc.row, tc.col, board.HorizontalDirection))
		}
	}
}

type crossSetEdgeTestCase struct {
	col         int
	rowContents string
	crossSet    CrossSet
	score       int
}

func TestGenCrossSetEdges(t *testing.T) {
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	if err != nil {
		t.Error(err)
	}
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	alph := dist.Alphabet()

	b := board.MakeBoard(board.CrosswordGameBoard)
	bcs := MakeBoardCrossSets(b)
	var testCases = []crossSetEdgeTestCase{
		{0, " A", CrossSetFromString("ABDFHKLMNPTYZ", alph), 1},
		{1, "A", CrossSetFromString("ABDEGHILMNRSTWXY", alph), 1},
		{13, "              F", CrossSetFromString("EIO", alph), 4},
		{14, "             F ", CrossSetFromString("AE", alph), 4},
		{14, "          WECH ", CrossSetFromString("T", alph), 12}, // phony!
		{14, "           ZZZ ", CrossSet(0), 30},
		{14, "       ZYZZYVA ", CrossSetFromString("S", alph), 43},
		{14, "        ZYZZYV ", CrossSetFromString("A", alph), 42}, // phony!
		{14, "       Z Z Y A ", CrossSetFromString("ABDEGHILMNRSTWXY", alph), 1},
		{12, "       z z Y A ", CrossSetFromString("E", alph), 5},
		{14, "OxYpHeNbUTAzON ", CrossSetFromString("E", alph), 15},
		{6, "OXYPHE BUTAZONE", CrossSetFromString("N", alph), 40},
		// Should still calculate score correctly despite no gaddag path.
		{0, " YHJKTKHKTLV", CrossSet(0), 42},
		{14, "   YHJKTKHKTLV ", CrossSet(0), 42},
		{6, "YHJKTK HKTLV", CrossSet(0), 42},
	}
	row := 4
	for _, tc := range testCases {
		b.SetRow(row, tc.rowContents, alph)
		GenCrossSet(b, bcs, row, tc.col, board.HorizontalDirection, gd, dist)
		if bcs.Get(row, tc.col, board.HorizontalDirection) != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				row, tc.col, tc.crossSet,
				bcs.Get(row, tc.col, board.HorizontalDirection))
		}
		if b.GetCrossScore(row, tc.col, board.HorizontalDirection) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				row, tc.col, tc.score,
				b.GetCrossScore(row, tc.col, board.HorizontalDirection))
		}
	}
}

func TestGenAllCrossSets(t *testing.T) {
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	if err != nil {
		t.Error(err)
	}
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	alph := dist.Alphabet()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetToGame(alph, VsEd)
	bcs := MakeBoardCrossSets(b)

	GenAllCrossSets(b, bcs, gd, dist)

	var testCases = []crossSetTestCase{
		{8, 8, CrossSetFromString("OS", alph), board.HorizontalDirection, 8},
		{8, 8, CrossSetFromString("S", alph), board.VerticalDirection, 9},
		{5, 11, CrossSetFromString("S", alph), board.HorizontalDirection, 5},
		{5, 11, CrossSetFromString("AO", alph), board.VerticalDirection, 2},
		{8, 13, CrossSetFromString("AEOU", alph), board.HorizontalDirection, 1},
		{8, 13, CrossSetFromString("AEIMOUY", alph), board.VerticalDirection, 3},
		{9, 13, CrossSetFromString("HMNPST", alph), board.HorizontalDirection, 1},
		{9, 13, TrivialCrossSet, board.VerticalDirection, 0},
		{14, 14, TrivialCrossSet, board.HorizontalDirection, 0},
		{14, 14, TrivialCrossSet, board.VerticalDirection, 0},
		{12, 12, CrossSet(0), board.HorizontalDirection, 0},
		{12, 12, CrossSet(0), board.VerticalDirection, 0},
	}

	for idx, tc := range testCases {
		// Compare values
		if bcs.Get(tc.row, tc.col, tc.dir) != tc.crossSet {
			t.Errorf("Test=%v For row=%v col=%v, Expected cross-set to be %v, got %v",
				idx, tc.row, tc.col, tc.crossSet,
				bcs.Get(tc.row, tc.col, tc.dir))
		}
		if b.GetCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				b.GetCrossScore(tc.row, tc.col, tc.dir))
		}
	}
	// This one has more nondeterministic (in-between LR) crosssets
	b.SetToGame(alph, VsMatt)
	GenAllCrossSets(b, bcs, gd, dist)
	testCases = []crossSetTestCase{
		{8, 7, CrossSetFromString("S", alph), board.HorizontalDirection, 11},
		{8, 7, CrossSet(0), board.VerticalDirection, 12},
		{5, 11, CrossSetFromString("BGOPRTWX", alph), board.HorizontalDirection, 2},
		{5, 11, CrossSet(0), board.VerticalDirection, 15},
		{8, 13, TrivialCrossSet, board.HorizontalDirection, 0},
		{8, 13, TrivialCrossSet, board.VerticalDirection, 0},
		{11, 4, CrossSetFromString("DRS", alph), board.HorizontalDirection, 6},
		{11, 4, CrossSetFromString("CGM", alph), board.VerticalDirection, 1},
		{2, 2, TrivialCrossSet, board.HorizontalDirection, 0},
		{2, 2, CrossSetFromString("AEI", alph), board.VerticalDirection, 2},
		{7, 12, CrossSetFromString("AEIOY", alph), board.HorizontalDirection, 0}, // it's a blank
		{7, 12, TrivialCrossSet, board.VerticalDirection, 0},
		{11, 8, CrossSet(0), board.HorizontalDirection, 4},
		{11, 8, CrossSetFromString("AEOU", alph), board.VerticalDirection, 1},
		{1, 8, CrossSetFromString("AEO", alph), board.HorizontalDirection, 1},
		{1, 8, CrossSetFromString("DFHLMNRSTX", alph), board.VerticalDirection, 1},
		{10, 10, CrossSetFromString("E", alph), board.HorizontalDirection, 11},
		{10, 10, TrivialCrossSet, board.VerticalDirection, 0},
	}
	for idx, tc := range testCases {
		// Compare values
		if bcs.Get(tc.row, tc.col, tc.dir) != tc.crossSet {
			t.Errorf("Test=%v For row=%v col=%v, Expected cross-set to be %v, got %v",
				idx, tc.row, tc.col, tc.crossSet,
				bcs.Get(tc.row, tc.col, tc.dir))
		}
		if b.GetCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				b.GetCrossScore(tc.row, tc.col, tc.dir))
		}
	}
}

func TestBoardsEqual(t *testing.T) {
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	if err != nil {
		t.Error(err)
	}
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	alph := dist.Alphabet()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetToGame(alph, VsMatt)
	bcs := MakeBoardCrossSets(b)
	GenAllCrossSets(b, bcs, gd, dist)

	c := board.MakeBoard(board.CrosswordGameBoard)
	c.SetToGame(alph, VsMatt)
	GenAllCrossSets(c, bcs, gd, dist)

	if !b.Equals(c) {
		log.Printf("Boards should be identical but they aren't")
	}
}

func TestPlaceMoveTiles(t *testing.T) {

	gd, _ := GaddagFromLexicon("America")
	b := board.MakeBoard(board.CrosswordGameBoard)
	alph := gd.GetAlphabet()

	b.SetToGame(alph, VsOxy)

	m := move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph)
	b.PlaceMoveTiles(m)
	for i, c := range "OXYPHENBUTAZONE" {
		assert.Equal(t, c, b.GetLetter(i, 0).UserVisible(alph))
	}
}

func TestUnplaceMoveTiles(t *testing.T) {
	gd, _ := GaddagFromLexicon("America")
	b := board.MakeBoard(board.CrosswordGameBoard)
	alph := gd.GetAlphabet()

	b.SetToGame(alph, VsOxy)

	m := move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph)
	b.PlaceMoveTiles(m)
	b.UnplaceMoveTiles(m)
	for i, c := range "  Y HEN UT  ON" {
		assert.Equal(t, c, b.GetLetter(i, 0).UserVisible(alph))
	}
}

// func TestRestoreFromBackup(t *testing.T) {
// 	gd, _ := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
// 	alph := gd.GetAlphabet()
// 	dist := alphabet.EnglishLetterDistribution()
// 	bag := dist.MakeBag(gd.GetAlphabet())

// 	// The same test cases as in the test above.
// 	var testCases = []updateCrossesForMoveTestCase{
// 		{VsMatt, move.NewScoringMoveSimple(38, "K9", "TAEL", "ABD", alph), "TAEL"},
// 		// Test right edge of board
// 		{VsMatt2, move.NewScoringMoveSimple(77, "O8", "TENsILE", "", alph), "TENsILE"},
// 		// Test through tiles
// 		{VsOxy, move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph),
// 			"OXYPHENBUTAZONE"},
// 		// Test top of board, horizontal
// 		{VsJeremy, move.NewScoringMoveSimple(14, "1G", "S.oWED", "D?", alph), "SNoWED"},
// 		// Test bottom of board, horizontal
// 		{VsJeremy, move.NewScoringMoveSimple(11, "15F", "F..ER", "", alph), "FOYER"},
// 	}

// 	// create a move.
// 	for _, tc := range testCases {
// 		b := board.MakeBoard(CrosswordGameBoard)
// 		b.SetToGame(alph, tc.testGame)
// 		b.GenAllCrossSets(gd, bag)
// 		b.UpdateAllAnchors()
// 		b.PlayMove(tc.m, gd, bag)
// 		b.RestoreFromBackup()

// 		// Create an identical board. We want to make sure nothing changed
// 		// after the rollback.
// 		c := board.MakeBoard(CrosswordGameBoard)
// 		c.SetToGame(alph, tc.testGame)
// 		c.GenAllCrossSets(gd, bag)
// 		c.UpdateAllAnchors()
// 		assert.True(t, b.Equals(c))
// 	}
// }

func TestUpdateSingleCrossSet(t *testing.T) {
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	if err != nil {
		t.Error(err)
	}
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	alph := dist.Alphabet()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetToGame(alph, board.VsMatt)
	bcs := MakeBoardCrossSets(b)
	GenAllCrossSets(b, bcs, gd, dist)

	// TAEL
	b.SetLetter(8, 10, 19)
	b.SetLetter(9, 10, 0)
	b.SetLetter(10, 10, 4)
	b.SetLetter(11, 10, 11)

	fmt.Println(b.ToDisplayText(alph))
	GenCrossSet(b, bcs, 7, 10, board.HorizontalDirection, gd, dist)
	b.Transpose()
	GenCrossSet(b, bcs, 10, 7, board.VerticalDirection, gd, dist)
	b.Transpose()

	if bcs.Get(7, 10, board.HorizontalDirection) != CrossSet(0) {
		t.Errorf("Expected 0, was %v",
			bcs.Get(7, 10, board.HorizontalDirection))
	}
	if bcs.Get(7, 10, board.VerticalDirection) != CrossSet(0) {
		t.Errorf("Expected 0, was %v",
			bcs.Get(7, 10, board.VerticalDirection))
	}
}
