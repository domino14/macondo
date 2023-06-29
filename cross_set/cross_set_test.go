package cross_set

import (
	"fmt"
	"log"
	"testing"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
	"github.com/matryer/is"
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

func GaddagFromLexicon(lex string) (*kwg.KWG, error) {
	return kwg.Get(&DefaultConfig, lex)
}

type crossSetTestCase struct {
	row      int
	col      int
	crossSet board.CrossSet
	dir      board.BoardDirection
	score    int
}

func TestGenCrossSetLoadedGame(t *testing.T) {
	is := is.New(t)

	gd, err := kwg.Get(&DefaultConfig, "NWL18")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	alph := dist.TileMapping()

	b := board.MakeBoard(board.CrosswordGameBoard)

	b.SetToGame(alph, VsMatt)
	// All horizontal for now.
	var testCases = []crossSetTestCase{
		{10, 10, board.CrossSetFromString("E", alph), board.HorizontalDirection, 11},
		{2, 4, board.CrossSetFromString("DKHLRSV", alph), board.HorizontalDirection, 9},
		{8, 7, board.CrossSetFromString("S", alph), board.HorizontalDirection, 11},
		// suffix - no hooks:
		{12, 8, board.CrossSet(0), board.HorizontalDirection, 11},
		// prefix - no hooks:
		{3, 1, board.CrossSet(0), board.HorizontalDirection, 10},
		// prefix and suffix, no path
		{6, 8, board.CrossSet(0), board.HorizontalDirection, 5},
		// More in-between
		{2, 10, board.CrossSetFromString("M", alph), board.HorizontalDirection, 2},
	}

	for _, tc := range testCases {
		GenCrossSet(b, tc.row, tc.col, tc.dir, gd, dist)
		if b.GetCrossSet(tc.row, tc.col, board.HorizontalDirection) != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				tc.row, tc.col, tc.crossSet,
				b.GetCrossSet(tc.row, tc.col, board.HorizontalDirection))
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
	crossSet    board.CrossSet
	score       int
}

func TestGenCrossSetEdges(t *testing.T) {
	is := is.New(t)

	gd, err := kwg.Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	alph := dist.TileMapping()

	b := board.MakeBoard(board.CrosswordGameBoard)

	var testCases = []crossSetEdgeTestCase{
		{0, " A", board.CrossSetFromString("ABDFHKLMNPTYZ", alph), 1},
		{1, "A", board.CrossSetFromString("ABDEGHILMNRSTWXY", alph), 1},
		{1, "R VOTED", board.CrossSetFromString("E", alph), 10},
		{1, "T UNFOLD", board.CrossSetFromString("", alph), 11},
		{7, "      A VODOU I", board.CrossSetFromString("", alph), 10},
		{2, "PA ABLE", board.CrossSetFromString("RY", alph), 10},
		{1, "W SIWYG", board.CrossSetFromString("Y", alph), 16},
		{13, "              F", board.CrossSetFromString("EIO", alph), 4},
		{14, "             F ", board.CrossSetFromString("AE", alph), 4},
		{14, "          WECH ", board.CrossSetFromString("T", alph), 12}, // phony!
		{14, "           ZZZ ", board.CrossSet(0), 30},
		{14, "       ZYZZYVA ", board.CrossSetFromString("S", alph), 43},
		{14, "        ZYZZYV ", board.CrossSetFromString("A", alph), 42}, // phony!
		{14, "       Z Z Y A ", board.CrossSetFromString("ABDEGHILMNRSTWXY", alph), 1},
		{12, "       z z Y A ", board.CrossSetFromString("E", alph), 5},
		{14, "OxYpHeNbUTAzON ", board.CrossSetFromString("E", alph), 15},
		{6, "OXYPHE BUTAZONE", board.CrossSetFromString("N", alph), 40},
		// Should still calculate score correctly despite no gaddag path.
		{0, " YHJKTKHKTLV", board.CrossSet(0), 42},
		{14, "   YHJKTKHKTLV ", board.CrossSet(0), 42},
		{6, "YHJKTK HKTLV", board.CrossSet(0), 42},
	}
	row := 4
	for _, tc := range testCases {
		b.SetRow(row, tc.rowContents, alph)
		GenCrossSet(b, row, tc.col, board.HorizontalDirection, gd, dist)
		if b.GetCrossSet(row, tc.col, board.HorizontalDirection) != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				row, tc.col, tc.crossSet,
				b.GetCrossSet(row, tc.col, board.HorizontalDirection))
		}
		if b.GetCrossScore(row, tc.col, board.HorizontalDirection) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				row, tc.col, tc.score,
				b.GetCrossScore(row, tc.col, board.HorizontalDirection))
		}
	}
}

func TestGenAllCrossSets(t *testing.T) {
	is := is.New(t)

	gd, err := kwg.Get(&DefaultConfig, "NWL18")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	alph := dist.TileMapping()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetToGame(alph, VsEd)

	GenAllCrossSets(b, gd, dist)

	var testCases = []crossSetTestCase{
		{8, 8, board.CrossSetFromString("OS", alph), board.HorizontalDirection, 8},
		{8, 8, board.CrossSetFromString("S", alph), board.VerticalDirection, 9},
		{5, 11, board.CrossSetFromString("S", alph), board.HorizontalDirection, 5},
		{5, 11, board.CrossSetFromString("AO", alph), board.VerticalDirection, 2},
		{8, 13, board.CrossSetFromString("AEOU", alph), board.HorizontalDirection, 1},
		{8, 13, board.CrossSetFromString("AEIMOUY", alph), board.VerticalDirection, 3},
		{9, 13, board.CrossSetFromString("HMNPST", alph), board.HorizontalDirection, 1},
		{9, 13, board.TrivialCrossSet, board.VerticalDirection, 0},
		{14, 14, board.TrivialCrossSet, board.HorizontalDirection, 0},
		{14, 14, board.TrivialCrossSet, board.VerticalDirection, 0},
		{12, 12, board.CrossSet(0), board.HorizontalDirection, 0},
		{12, 12, board.CrossSet(0), board.VerticalDirection, 0},
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
	b.SetToGame(alph, VsMatt)
	GenAllCrossSets(b, gd, dist)
	testCases = []crossSetTestCase{
		{8, 7, board.CrossSetFromString("S", alph), board.HorizontalDirection, 11},
		{8, 7, board.CrossSet(0), board.VerticalDirection, 12},
		{5, 11, board.CrossSetFromString("BGOPRTWX", alph), board.HorizontalDirection, 2},
		{5, 11, board.CrossSet(0), board.VerticalDirection, 15},
		{8, 13, board.TrivialCrossSet, board.HorizontalDirection, 0},
		{8, 13, board.TrivialCrossSet, board.VerticalDirection, 0},
		{11, 4, board.CrossSetFromString("DRS", alph), board.HorizontalDirection, 6},
		{11, 4, board.CrossSetFromString("CGM", alph), board.VerticalDirection, 1},
		{2, 2, board.TrivialCrossSet, board.HorizontalDirection, 0},
		{2, 2, board.CrossSetFromString("AEI", alph), board.VerticalDirection, 2},
		{7, 12, board.CrossSetFromString("AEIOY", alph), board.HorizontalDirection, 0}, // it's a blank
		{7, 12, board.TrivialCrossSet, board.VerticalDirection, 0},
		{11, 8, board.CrossSet(0), board.HorizontalDirection, 4},
		{11, 8, board.CrossSetFromString("AEOU", alph), board.VerticalDirection, 1},
		{1, 8, board.CrossSetFromString("AEO", alph), board.HorizontalDirection, 1},
		{1, 8, board.CrossSetFromString("DFHLMNRSTWX", alph), board.VerticalDirection, 1},
		{10, 10, board.CrossSetFromString("E", alph), board.HorizontalDirection, 11},
		{10, 10, board.TrivialCrossSet, board.VerticalDirection, 0},
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
	is := is.New(t)

	gd, err := kwg.Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	alph := dist.TileMapping()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetToGame(alph, VsMatt)
	GenAllCrossSets(b, gd, dist)

	c := board.MakeBoard(board.CrosswordGameBoard)
	c.SetToGame(alph, VsMatt)
	GenAllCrossSets(c, gd, dist)

	if !b.Equals(c) {
		log.Printf("Boards should be identical but they aren't")
	}
}

func TestPlaceMoveTiles(t *testing.T) {

	gd, _ := kwg.Get(&DefaultConfig, "NWL20")
	b := board.MakeBoard(board.CrosswordGameBoard)
	alph := gd.GetAlphabet()

	b.SetToGame(alph, VsOxy)

	m := move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph)
	b.PlaceMoveTiles(m)
	for i, c := range "OXYPHENBUTAZONE" {
		assert.Equal(t, string(c), b.GetLetter(i, 0).UserVisible(alph, true))
	}
}

func TestUnplaceMoveTiles(t *testing.T) {
	gd, _ := kwg.Get(&DefaultConfig, "NWL20")
	b := board.MakeBoard(board.CrosswordGameBoard)
	alph := gd.GetAlphabet()

	b.SetToGame(alph, VsOxy)

	m := move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph)
	b.PlaceMoveTiles(m)
	b.UnplaceMoveTiles(m)
	for i, c := range "..Y.HEN.UT..ON" {
		assert.Equal(t, string(c), b.GetLetter(i, 0).UserVisible(alph, true))
	}
}

type updateCrossesForMoveTestCase struct {
	testGame        board.VsWho
	m               *move.Move
	userVisibleWord string
}

func TestUpdateCrossSetsForMove(t *testing.T) {
	is := is.New(t)

	gd, err := kwg.Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	gen := GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
	alph := dist.TileMapping()

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
	for _, tc := range testCases {
		b := board.MakeBoard(board.CrosswordGameBoard)
		b.SetToGame(alph, tc.testGame)
		gen.GenerateAll(b)
		b.UpdateAllAnchors()
		b.PlayMove(tc.m, dist)
		gen.UpdateForMove(b, tc.m)
		log.Printf(b.ToDisplayText(alph))
		// Create an identical board, but generate cross-sets for the entire
		// board after placing the letters "manually".
		c := board.MakeBoard(board.CrosswordGameBoard)
		c.SetToGame(alph, tc.testGame)
		c.PlaceMoveTiles(tc.m)
		c.TestSetTilesPlayed(c.GetTilesPlayed() + tc.m.TilesPlayed())
		GenAllCrossSets(c, gd, dist)
		c.UpdateAllAnchors()

		assert.True(t, b.Equals(c))

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
			uv := b.GetLetter(row+rowInc, col+colInc).UserVisible(alph, true)
			assert.Equal(t, string(c), uv)
		}
	}
}

// func TestRestoreFromBackup(t *testing.T) {
// 	gd, _ := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
// 	alph := gd.GetAlphabet()
// 	dist := tilemapping.EnglishLetterDistribution()
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
	is := is.New(t)

	gd, err := kwg.Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	alph := dist.TileMapping()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetToGame(alph, VsMatt)
	GenAllCrossSets(b, gd, dist)

	b.SetLetter(8, 10, 19)
	b.SetLetter(9, 10, 0)
	b.SetLetter(10, 10, 4)
	b.SetLetter(11, 10, 11)
	fmt.Println(b.ToDisplayText(alph))
	GenCrossSet(b, 7, 10, board.HorizontalDirection, gd, dist)
	b.Transpose()
	GenCrossSet(b, 10, 7, board.VerticalDirection, gd, dist)
	b.Transpose()

	if b.GetCrossSet(7, 10, board.HorizontalDirection) != board.CrossSet(0) {
		t.Errorf("Expected 0, was %v",
			b.GetCrossSet(7, 10, board.HorizontalDirection))
	}
	if b.GetCrossSet(7, 10, board.VerticalDirection) != board.CrossSet(0) {
		t.Errorf("Expected 0, was %v",
			b.GetCrossSet(7, 10, board.VerticalDirection))
	}
}

// Cross-score only tests

// Copy of TestGenAllCrossSets with the cross-set bits removed
func TestGenAllCrossScores(t *testing.T) {
	is := is.New(t)

	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	alph := dist.TileMapping()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetToGame(alph, VsEd)

	GenAllCrossScores(b, dist)

	var testCases = []crossSetTestCase{
		{8, 8, board.CrossSetFromString("OS", alph), board.HorizontalDirection, 8},
		{8, 8, board.CrossSetFromString("S", alph), board.VerticalDirection, 9},
		{5, 11, board.CrossSetFromString("S", alph), board.HorizontalDirection, 5},
		{5, 11, board.CrossSetFromString("AO", alph), board.VerticalDirection, 2},
		{8, 13, board.CrossSetFromString("AEOU", alph), board.HorizontalDirection, 1},
		{8, 13, board.CrossSetFromString("AEIMOUY", alph), board.VerticalDirection, 3},
		{9, 13, board.CrossSetFromString("HMNPST", alph), board.HorizontalDirection, 1},
		{9, 13, board.TrivialCrossSet, board.VerticalDirection, 0},
		{14, 14, board.TrivialCrossSet, board.HorizontalDirection, 0},
		{14, 14, board.TrivialCrossSet, board.VerticalDirection, 0},
		{12, 12, board.CrossSet(0), board.HorizontalDirection, 0},
		{12, 12, board.CrossSet(0), board.VerticalDirection, 0},
	}

	for _, tc := range testCases {
		// Compare values
		if b.GetCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				b.GetCrossScore(tc.row, tc.col, tc.dir))
		}
	}
	// This one has more nondeterministic (in-between LR) crosssets
	b.SetToGame(alph, VsMatt)
	GenAllCrossScores(b, dist)
	testCases = []crossSetTestCase{
		{8, 7, board.CrossSetFromString("S", alph), board.HorizontalDirection, 11},
		{8, 7, board.CrossSet(0), board.VerticalDirection, 12},
		{5, 11, board.CrossSetFromString("BGOPRTWX", alph), board.HorizontalDirection, 2},
		{5, 11, board.CrossSet(0), board.VerticalDirection, 15},
		{8, 13, board.TrivialCrossSet, board.HorizontalDirection, 0},
		{8, 13, board.TrivialCrossSet, board.VerticalDirection, 0},
		{11, 4, board.CrossSetFromString("DRS", alph), board.HorizontalDirection, 6},
		{11, 4, board.CrossSetFromString("CGM", alph), board.VerticalDirection, 1},
		{2, 2, board.TrivialCrossSet, board.HorizontalDirection, 0},
		{2, 2, board.CrossSetFromString("AEI", alph), board.VerticalDirection, 2},
		{7, 12, board.CrossSetFromString("AEIOY", alph), board.HorizontalDirection, 0}, // it's a blank
		{7, 12, board.TrivialCrossSet, board.VerticalDirection, 0},
		{11, 8, board.CrossSet(0), board.HorizontalDirection, 4},
		{11, 8, board.CrossSetFromString("AEOU", alph), board.VerticalDirection, 1},
		{1, 8, board.CrossSetFromString("AEO", alph), board.HorizontalDirection, 1},
		{1, 8, board.CrossSetFromString("DFHLMNRSTX", alph), board.VerticalDirection, 1},
		{10, 10, board.CrossSetFromString("E", alph), board.HorizontalDirection, 11},
		{10, 10, board.TrivialCrossSet, board.VerticalDirection, 0},
	}
	for _, tc := range testCases {
		if b.GetCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				b.GetCrossScore(tc.row, tc.col, tc.dir))
		}
	}
}

// Copy of TestUpdateCrossSetsForMove with the CrossScoreOnlyGenerator
func TestUpdateCrossScoresForMove(t *testing.T) {
	is := is.New(t)

	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	gen := CrossScoreOnlyGenerator{Dist: dist}
	alph := dist.TileMapping()

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
	for _, tc := range testCases {
		b := board.MakeBoard(board.CrosswordGameBoard)
		b.SetToGame(alph, tc.testGame)
		gen.GenerateAll(b)
		b.UpdateAllAnchors()
		b.PlayMove(tc.m, dist)
		gen.UpdateForMove(b, tc.m)
		log.Printf(b.ToDisplayText(alph))
		// Create an identical board, but generate cross-sets for the entire
		// board after placing the letters "manually".
		c := board.MakeBoard(board.CrosswordGameBoard)
		c.SetToGame(alph, tc.testGame)
		c.PlaceMoveTiles(tc.m)
		c.TestSetTilesPlayed(c.GetTilesPlayed() + tc.m.TilesPlayed())
		GenAllCrossScores(c, dist)
		c.UpdateAllAnchors()

		assert.True(t, b.Equals(c))

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
			uv := b.GetLetter(row+rowInc, col+colInc).UserVisible(alph, true)
			assert.Equal(t, string(c), uv)
		}
	}
}

// Comparison tests between the two generators for cross-score generation

func compareCrossScores(t *testing.T, b1 *board.GameBoard, b2 *board.GameBoard) {
	dim := b1.Dim()
	dirs := []board.BoardDirection{board.HorizontalDirection, board.VerticalDirection}
	var d board.BoardDirection

	for r := 0; r < dim; r++ {
		for c := 0; c < dim; c++ {
			for _, d = range dirs {
				cs1 := b1.GetCrossScore(r, c, d)
				cs2 := b2.GetCrossScore(r, c, d)
				assert.Equal(t, cs1, cs2)
			}
		}
	}
}

func TestCompareUpdate(t *testing.T) {
	is := is.New(t)

	gd, err := kwg.Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	gen1 := GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
	gen2 := CrossScoreOnlyGenerator{Dist: dist}
	alph := dist.TileMapping()

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
	for _, tc := range testCases {
		// Run the cross set generator on b1
		b1 := board.MakeBoard(board.CrosswordGameBoard)
		b1.SetToGame(alph, tc.testGame)
		gen1.GenerateAll(b1)
		b1.UpdateAllAnchors()
		b1.PlayMove(tc.m, dist)
		gen1.UpdateForMove(b1, tc.m)

		// Run the cross score generator on b2
		b2 := board.MakeBoard(board.CrosswordGameBoard)
		b2.SetToGame(alph, tc.testGame)
		gen2.GenerateAll(b2)
		b2.UpdateAllAnchors()
		b2.PlayMove(tc.m, dist)
		gen2.UpdateForMove(b2, tc.m)

		compareCrossScores(t, b1, b2)
	}
}

func TestCompareGenAll(t *testing.T) {
	is := is.New(t)

	gd, err := kwg.Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	alph := dist.TileMapping()

	var testCases = []board.VsWho{
		VsEd,
		VsJeremy,
		VsMatt,
		VsMatt2,
		VsOxy,
	}

	for _, tc := range testCases {
		b1 := board.MakeBoard(board.CrosswordGameBoard)
		b1.SetToGame(alph, tc)
		GenAllCrossSets(b1, gd, dist)

		b2 := board.MakeBoard(board.CrosswordGameBoard)
		b2.SetToGame(alph, tc)
		GenAllCrossScores(b2, dist)

		compareCrossScores(t, b1, b2)
	}
}

// Benchmarks

func BenchmarkGenAnchorsAndCrossSets(b *testing.B) {
	is := is.New(b)

	gd, err := kwg.Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	alph := dist.TileMapping()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	bd.SetToGame(alph, VsOxy)
	b.ResetTimer()

	// 38 us
	for i := 0; i < b.N; i++ {
		bd.UpdateAllAnchors()
		GenAllCrossSets(bd, gd, dist)
	}
}

func BenchmarkMakePlay(b *testing.B) {
	// Mostly, benchmark the progressive generation of anchors and cross-sets
	// (as opposed to generating all of them from scratch)
	is := is.New(b)

	gd, err := kwg.Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	gen := GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
	alph := dist.TileMapping()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	bd.SetToGame(alph, VsMatt)
	gen.GenerateAll(bd)
	bd.UpdateAllAnchors()

	// create a move.
	m := move.NewScoringMove(
		38,
		tilemapping.MachineWord([]tilemapping.MachineLetter{19, 0, 4, 11}), // TAEL
		tilemapping.MachineWord([]tilemapping.MachineLetter{0, 1, 3}),
		true,
		4,
		alph,
		8, 10)

	b.ResetTimer()
	// 2.7 us; more than 10x faster than regenerating all anchors every time.
	// seems worth it.
	for i := 0; i < b.N; i++ {
		bd.PlayMove(m, dist)
		gen.UpdateForMove(bd, m)
	}

}
