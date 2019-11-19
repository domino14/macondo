package alphabeta

import (
	"fmt"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/move"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
)

func TestRectanglesIntersect(t *testing.T) {
	is := is.New(t)
	// a play on row 3 and row 4 should not overlap
	type testcase struct {
		r1        rect
		r2        rect
		intersect bool
	}
	testcases := []testcase{
		// This looks like a 3-tile play starting at C, R = (3, 2) and
		// a 4-tile play staritng at C,R = (5, 3)
		{rect{3, 2, 5, 2}, rect{5, 3, 8, 3}, false},
		// a 3-tile play starting at C, R = (3, 2) and
		// a 4-tile play starting at C, R = (6, 2)
		{rect{3, 2, 5, 2}, rect{6, 2, 9, 2}, false},
		// same as above but the 4-tile play starts at (5, 2)
		{rect{3, 2, 5, 2}, rect{5, 2, 8, 2}, true},
		// same as above but the 4-tile play is a 3-tile play that
		// starts at 3, 3
		{rect{3, 2, 5, 2}, rect{3, 3, 5, 3}, false},
	}
	for _, tc := range testcases {
		is.True(rectanglesIntersect(tc.r1, tc.r2) == tc.intersect)
	}

}

func TestBlocks(t *testing.T) {
	is := is.New(t)
	row := "  SUP  AT ON  " // SUPS does not block ATTONE, etc.
	// U(P) should not be blocked by or block SUPINATION
	alph := alphabet.EnglishAlphabet()
	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetRow(3, row, alph)
	fmt.Println(b.ToDisplayText(alph))
	is.Fail()
}

func TestComplexBlocks(t *testing.T) {
	is := is.New(t)
	row1 := "    BETA      "
	row2 := "   HA   OSES  "
	alph := alphabet.EnglishAlphabet()
	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetRow(10, row1, alph)
	b.SetRow(11, row2, alph)
	fmt.Println(b.ToDisplayText(alph))
	// Some tricky situations:
	// HE(BETA)TE does not block (HA)LIT(OSES) (and the other way around)
	// ZA on the A in BETA should block (HA)LIT(OSES) (and vice-versa)
	// IT under HA does not block (HA)LIT(OSES)
	// ITS does block it (it actually doesn't because ELS is a word, but we
	//   are not considering that at this time)
	// BETAS/SO should be blocked by AS on the first S of OSES, or by OF from
	//   the O in OSES, or by AR/AS/RE overlapping the SE in OSES, e.g.
	// (S)HA blocks (HA)LIT(OSES)
	type testcase struct {
		stmPlay *move.Move
		otsPlay *move.Move
		blocks  bool
	}
	s := &Solver{}
	s.Init(nil, nil)
	testcases := []testcase{
		{move.NewScoringMoveSimple(0, "12D", "..LIT....", "", alph),
			move.NewScoringMoveSimple(0, "11C", "HE....TE", "", alph), false},
		{move.NewScoringMoveSimple(0, "11C", "HE....TE", "", alph),
			move.NewScoringMoveSimple(0, "12D", "..LIT....", "", alph), false},
		{move.NewScoringMoveSimple(0, "H10", "Z.", "", alph),
			move.NewScoringMoveSimple(0, "12D", "..LIT....", "", alph), true},
		{move.NewScoringMoveSimple(0, "12D", "..LIT....", "", alph),
			move.NewScoringMoveSimple(0, "H10", "Z.", "", alph), true},
	}
	for _, tc := range testcases {
		fmt.Println("trying", tc.stmPlay, "blocks", tc.otsPlay, "expect", tc.blocks)
		is.True(s.blocks(tc.stmPlay, tc.otsPlay) == tc.blocks)
	}
}
