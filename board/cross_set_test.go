package board

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/move"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/lexicon"
)

var LexiconDir = os.Getenv("LEXICON_DIR")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/gen_america.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "America.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/gen_america.gaddag")
	}
	os.Exit(m.Run())
}
func TestCrossSet(t *testing.T) {
	cs := CrossSet(0)
	cs.set(13)

	if uint64(cs) != 8192 /* 1<<13 */ {
		t.Errorf("Expected cross-set to be %v, got %v", 8192, cs)
	}
	cs.set(0)
	if uint64(cs) != 8193 {
		t.Errorf("Expected cross-set to be %v, got %v", 8193, cs)
	}
}

type testpair struct {
	l       alphabet.MachineLetter
	allowed bool
}

func TestCrossSetAllowed(t *testing.T) {
	cs := CrossSet(8193)

	var allowedTests = []testpair{
		{alphabet.MachineLetter(1), false},
		{alphabet.MachineLetter(0), true},
		{alphabet.MachineLetter(14), false},
		{alphabet.MachineLetter(13), true},
		{alphabet.MachineLetter(12), false},
	}

	for _, pair := range allowedTests {
		allowed := cs.Allowed(pair.l)
		if allowed != pair.allowed {
			t.Errorf("For %v, expected %v, got %v", pair.l, pair.allowed,
				allowed)
		}
	}
}

type crossSetTestCase struct {
	row      int
	col      int
	crossSet CrossSet
	dir      BoardDirection
	score    int
}

func TestGenCrossSetLoadedGame(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	b := MakeBoard(CrosswordGameBoard)
	alph := gd.GetAlphabet()
	dist := lexicon.EnglishLetterDistribution()
	bag := dist.MakeBag(gd.GetAlphabet())
	b.SetBoardToGame(alph, VsMatt)
	// All horizontal for now.
	var testCases = []crossSetTestCase{
		{10, 10, CrossSetFromString("E", alph), HorizontalDirection, 11},
		{2, 4, CrossSetFromString("DKHLRSV", alph), HorizontalDirection, 9},
		{8, 7, CrossSetFromString("S", alph), HorizontalDirection, 11},
		// suffix - no hooks:
		{12, 8, CrossSet(0), HorizontalDirection, 11},
		// prefix - no hooks:
		{3, 1, CrossSet(0), HorizontalDirection, 10},
		// prefix and suffix, no path
		{6, 8, CrossSet(0), HorizontalDirection, 5},
		// More in-between
		{2, 10, CrossSetFromString("M", alph), HorizontalDirection, 2},
	}

	for _, tc := range testCases {
		b.GenCrossSet(tc.row, tc.col, tc.dir, gd, bag)
		if b.GetCrossSet(tc.row, tc.col, HorizontalDirection) != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				tc.row, tc.col, tc.crossSet,
				b.GetCrossSet(tc.row, tc.col, HorizontalDirection))
		}
		if b.GetCrossScore(tc.row, tc.col, HorizontalDirection) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				b.GetCrossScore(tc.row, tc.col, HorizontalDirection))
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
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	dist := lexicon.EnglishLetterDistribution()
	bag := dist.MakeBag(gd.GetAlphabet())
	b := MakeBoard(CrosswordGameBoard)

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
		b.GenCrossSet(row, tc.col, HorizontalDirection, gd, bag)
		if b.GetCrossSet(row, tc.col, HorizontalDirection) != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				row, tc.col, tc.crossSet,
				b.GetCrossSet(row, tc.col, HorizontalDirection))
		}
		if b.GetCrossScore(row, tc.col, HorizontalDirection) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				row, tc.col, tc.score,
				b.GetCrossScore(row, tc.col, HorizontalDirection))
		}
	}
}

func TestGenAllCrossSets(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	dist := lexicon.EnglishLetterDistribution()
	bag := dist.MakeBag(gd.GetAlphabet())
	b := MakeBoard(CrosswordGameBoard)
	b.SetBoardToGame(alph, VsEd)

	b.GenAllCrossSets(gd, bag)

	var testCases = []crossSetTestCase{
		{8, 8, CrossSetFromString("OS", alph), HorizontalDirection, 8},
		{8, 8, CrossSetFromString("S", alph), VerticalDirection, 9},
		{5, 11, CrossSetFromString("S", alph), HorizontalDirection, 5},
		{5, 11, CrossSetFromString("AO", alph), VerticalDirection, 2},
		{8, 13, CrossSetFromString("AEOU", alph), HorizontalDirection, 1},
		{8, 13, CrossSetFromString("AEIMOUY", alph), VerticalDirection, 3},
		{9, 13, CrossSetFromString("HMNPST", alph), HorizontalDirection, 1},
		{9, 13, TrivialCrossSet, VerticalDirection, 0},
		{14, 14, TrivialCrossSet, HorizontalDirection, 0},
		{14, 14, TrivialCrossSet, VerticalDirection, 0},
		{12, 12, CrossSet(0), HorizontalDirection, 0},
		{12, 12, CrossSet(0), VerticalDirection, 0},
	}

	for idx, tc := range testCases {
		// Compare values
		if b.GetCrossSet(tc.row, tc.col, tc.dir) != tc.crossSet {
			t.Errorf("Test=%v For row=%v col=%v, Expected cross-set to be %v, got %v",
				idx, tc.row, tc.col, tc.crossSet,
				b.GetCrossSet(tc.row, tc.col, tc.dir))
		}
		if b.GetCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				b.GetCrossScore(tc.row, tc.col, tc.dir))
		}
	}
	// This one has more nondeterministic (in-between LR) crosssets
	b.SetBoardToGame(alph, VsMatt)
	b.GenAllCrossSets(gd, bag)
	testCases = []crossSetTestCase{
		{8, 7, CrossSetFromString("S", alph), HorizontalDirection, 11},
		{8, 7, CrossSet(0), VerticalDirection, 12},
		{5, 11, CrossSetFromString("BGOPRTWX", alph), HorizontalDirection, 2},
		{5, 11, CrossSet(0), VerticalDirection, 15},
		{8, 13, TrivialCrossSet, HorizontalDirection, 0},
		{8, 13, TrivialCrossSet, VerticalDirection, 0},
		{11, 4, CrossSetFromString("DRS", alph), HorizontalDirection, 6},
		{11, 4, CrossSetFromString("CGM", alph), VerticalDirection, 1},
		{2, 2, TrivialCrossSet, HorizontalDirection, 0},
		{2, 2, CrossSetFromString("AEI", alph), VerticalDirection, 2},
		{7, 12, CrossSetFromString("AEIOY", alph), HorizontalDirection, 0}, // it's a blank
		{7, 12, TrivialCrossSet, VerticalDirection, 0},
		{11, 8, CrossSet(0), HorizontalDirection, 4},
		{11, 8, CrossSetFromString("AEOU", alph), VerticalDirection, 1},
		{1, 8, CrossSetFromString("AEO", alph), HorizontalDirection, 1},
		{1, 8, CrossSetFromString("DFHLMNRSTX", alph), VerticalDirection, 1},
		{10, 10, CrossSetFromString("E", alph), HorizontalDirection, 11},
		{10, 10, TrivialCrossSet, VerticalDirection, 0},
	}
	for idx, tc := range testCases {
		// Compare values
		if b.GetCrossSet(tc.row, tc.col, tc.dir) != tc.crossSet {
			t.Errorf("Test=%v For row=%v col=%v, Expected cross-set to be %v, got %v",
				idx, tc.row, tc.col, tc.crossSet,
				b.GetCrossSet(tc.row, tc.col, tc.dir))
		}
		if b.GetCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				b.GetCrossScore(tc.row, tc.col, tc.dir))
		}
	}
}

func TestBoardsEqual(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	dist := lexicon.EnglishLetterDistribution()
	bag := dist.MakeBag(gd.GetAlphabet())

	b := MakeBoard(CrosswordGameBoard)
	b.SetBoardToGame(alph, VsMatt)
	b.GenAllCrossSets(gd, bag)

	c := MakeBoard(CrosswordGameBoard)
	c.SetBoardToGame(alph, VsMatt)
	c.GenAllCrossSets(gd, bag)

	if !b.equals(c) {
		log.Printf("Boards should be identical but they aren't")
	}
}

func TestPlaceMoveTiles(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	b := MakeBoard(CrosswordGameBoard)
	alph := gd.GetAlphabet()

	b.SetBoardToGame(alph, VsOxy)

	m := move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph)
	b.placeMoveTiles(m)
	for i, c := range "OXYPHENBUTAZONE" {
		if b.squares[i][0].letter.UserVisible(alph) != c {
			t.Errorf("Fail")
		}
	}
}

func TestUpdateCrossSetsForMove(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	dist := lexicon.EnglishLetterDistribution()
	bag := dist.MakeBag(gd.GetAlphabet())

	type updateCrossesForMoveTestCase struct {
		testGame        VsWho
		m               *move.Move
		userVisibleWord string
	}

	var testCases = []updateCrossesForMoveTestCase{
		{VsMatt, move.NewScoringMoveSimple(38, "K9", "TAEL", "ABD", alph), "TAEL"},
		// Test right edge of board
		{VsMatt2, move.NewScoringMoveSimple(77, "O8", "TENsILE", "", alph), "TENsILE"},
		// Test through tiles
		{VsOxy, move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph),
			"OXYPHENBUTAZONE"},
		// Test top of board, horizontal
		{VsJeremy, move.NewScoringMoveSimple(14, "1G", "S.oWED", "D?", alph), "SNoWED"},
		// Test bottom of board, horizontal
		{VsJeremy, move.NewScoringMoveSimple(11, "15F", "F..ER", "", alph), "FOYER"},
	}

	// create a move.
	for idx, tc := range testCases {
		b := MakeBoard(CrosswordGameBoard)
		b.SetBoardToGame(alph, tc.testGame)
		b.GenAllCrossSets(gd, bag)
		b.UpdateAllAnchors()
		b.PlayMove(tc.m, gd, bag)
		log.Printf(b.ToDisplayText(alph))
		// Create an identical board, but generate cross-sets for the entire
		// board after placing the letters "manually".
		c := MakeBoard(CrosswordGameBoard)
		c.SetBoardToGame(alph, tc.testGame)
		c.placeMoveTiles(tc.m)
		c.GenAllCrossSets(gd, bag)
		c.UpdateAllAnchors()

		if !b.equals(c) {
			t.Errorf("idx: %v - Boards did not equal each other", idx)
		}

		for i, c := range tc.userVisibleWord {
			row, col, vertical := tc.m.CoordsAndVertical()
			var rowInc, colInc int
			if vertical {
				rowInc = i
				colInc = 0
			} else {
				rowInc = 0
				colInc = i
			}
			uv := b.squares[row+rowInc][col+colInc].letter.UserVisible(alph)
			if uv != c {
				t.Errorf("idx: %v - Fail - expected %v (%c), got %v (%c) (row=%v col=%v)",
					idx, c, c, uv, uv, row, col+i)
			}
		}
	}

}

func TestUpdateSingleCrossSet(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	dist := lexicon.EnglishLetterDistribution()
	bag := dist.MakeBag(gd.GetAlphabet())
	b := MakeBoard(CrosswordGameBoard)
	b.SetBoardToGame(alph, VsMatt)
	b.GenAllCrossSets(gd, bag)

	b.squares[8][10].letter = 19
	b.squares[9][10].letter = 0
	b.squares[10][10].letter = 4
	b.squares[11][10].letter = 11
	fmt.Println(b.ToDisplayText(alph))
	b.GenCrossSet(7, 10, HorizontalDirection, gd, bag)
	b.Transpose()
	b.GenCrossSet(10, 7, VerticalDirection, gd, bag)
	b.Transpose()

	if b.GetCrossSet(7, 10, HorizontalDirection) != CrossSet(0) {
		t.Errorf("Expected 0, was %v",
			b.GetCrossSet(7, 10, HorizontalDirection))
	}
	if b.GetCrossSet(7, 10, VerticalDirection) != CrossSet(0) {
		t.Errorf("Expected 0, was %v",
			b.GetCrossSet(7, 10, VerticalDirection))
	}
}

func BenchmarkGenAnchorsAndCrossSets(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	dist := lexicon.EnglishLetterDistribution()
	bag := dist.MakeBag(alph)
	board := MakeBoard(CrosswordGameBoard)
	board.SetBoardToGame(alph, VsOxy)
	b.ResetTimer()

	// 38 us
	for i := 0; i < b.N; i++ {
		board.UpdateAllAnchors()
		board.GenAllCrossSets(gd, bag)
	}
}

func BenchmarkMakePlay(b *testing.B) {
	// Mostly, benchmark the progressive generation of anchors and cross-sets
	// (as opposed to generating all of them from scratch)
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	dist := lexicon.EnglishLetterDistribution()
	bag := dist.MakeBag(gd.GetAlphabet())
	board := MakeBoard(CrosswordGameBoard)
	board.SetBoardToGame(alph, VsMatt)
	board.GenAllCrossSets(gd, bag)
	board.UpdateAllAnchors()

	// create a move.
	m := move.NewScoringMove(
		38,
		alphabet.MachineWord([]alphabet.MachineLetter{19, 0, 4, 11}), // TAEL
		alphabet.MachineWord([]alphabet.MachineLetter{0, 1, 3}),
		true,
		4,
		alph,
		8, 10, "K9")

	b.ResetTimer()
	// 2.7 us; more than 10x faster than regenerating all anchors every time.
	// seems worth it.
	for i := 0; i < b.N; i++ {
		board.PlayMove(m, gd, bag)
	}

}
