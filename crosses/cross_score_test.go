package crosses

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
)

var DefaultConfig = config.DefaultConfig()

type crossScoreTestCase struct {
	row   int
	col   int
	dir   board.BoardDirection
	score int
}

// Cross-score only tests

// Copy of TestGenAllCrossSets with the cross-set bits removed
func TestGenAllCrossScores(t *testing.T) {
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	alph := dist.Alphabet()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetToGame(alph, board.VsEd)

	GenAllCrossScores(b, dist)

	var testCases = []crossScoreTestCase{
		{8, 8, board.HorizontalDirection, 8},
		{8, 8, board.VerticalDirection, 9},
		{5, 11, board.HorizontalDirection, 5},
		{5, 11, board.VerticalDirection, 2},
		{8, 13, board.HorizontalDirection, 1},
		{8, 13, board.VerticalDirection, 3},
		{9, 13, board.HorizontalDirection, 1},
		{9, 13, board.VerticalDirection, 0},
		{14, 14, board.HorizontalDirection, 0},
		{14, 14, board.VerticalDirection, 0},
		{12, 12, board.HorizontalDirection, 0},
		{12, 12, board.VerticalDirection, 0},
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
	b.SetToGame(alph, board.VsMatt)
	GenAllCrossScores(b, dist)
	testCases = []crossScoreTestCase{
		{8, 7, board.HorizontalDirection, 11},
		{8, 7, board.VerticalDirection, 12},
		{5, 11, board.HorizontalDirection, 2},
		{5, 11, board.VerticalDirection, 15},
		{8, 13, board.HorizontalDirection, 0},
		{8, 13, board.VerticalDirection, 0},
		{11, 4, board.HorizontalDirection, 6},
		{11, 4, board.VerticalDirection, 1},
		{2, 2, board.HorizontalDirection, 0},
		{2, 2, board.VerticalDirection, 2},
		{7, 12, board.HorizontalDirection, 0}, // it's a blank
		{7, 12, board.VerticalDirection, 0},
		{11, 8, board.HorizontalDirection, 4},
		{11, 8, board.VerticalDirection, 1},
		{1, 8, board.HorizontalDirection, 1},
		{1, 8, board.VerticalDirection, 1},
		{10, 10, board.HorizontalDirection, 11},
		{10, 10, board.VerticalDirection, 0},
	}
	for _, tc := range testCases {
		if b.GetCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				b.GetCrossScore(tc.row, tc.col, tc.dir))
		}
	}
}
