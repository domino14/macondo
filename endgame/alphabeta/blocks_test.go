package alphabeta

import (
	"fmt"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/move"
	"github.com/matryer/is"
	"github.com/rs/zerolog/log"

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

type blocktestcase struct {
	stmPlay *move.Move
	otsPlay *move.Move
	blocks  bool
}

func TestBlocks(t *testing.T) {
	is := is.New(t)
	row := "  SUP  AT ON  " // SUPS does not block ATTONE, etc.
	// U(P) should not be blocked by or block SUPINATION
	alph := alphabet.EnglishAlphabet()
	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetRow(3, row, alph)
	fmt.Println(b.ToDisplayText(alph))

	s := &Solver{}
	s.Init(nil, nil)

	sups := move.NewScoringMoveSimple(0, "4C", "...S", "", alph)
	supination := move.NewScoringMoveSimple(0, "4C", "...IN..I..", "", alph)
	supinations := move.NewScoringMoveSimple(0, "4C", "...IN..I..S", "", alph)
	attone := move.NewScoringMoveSimple(0, "4H", "..T..E", "", alph)
	up := move.NewScoringMoveSimple(0, "E3", "U.", "", alph)
	cat := move.NewScoringMoveSimple(0, "4G", "C..", "", alph)

	testcases := []blocktestcase{
		{sups, supination, true},
		{supination, sups, true},
		{sups, supinations, true},
		{supinations, sups, true},
		{sups, attone, false},
		{attone, sups, false},
		{up, supination, false},
		{supination, up, false},
		{up, sups, false},
		{sups, up, false},
		{cat, sups, true},
		{sups, cat, true},
		{cat, attone, true},
		{attone, cat, true},
	}

	for _, tc := range testcases {
		log.Debug().Msgf("trying %v blocks %v, expect %v", tc.stmPlay, tc.otsPlay, tc.blocks)
		is.True(s.blocks(tc.stmPlay, tc.otsPlay, b) == tc.blocks)
	}
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
	// all plays should block themselves!

	s := &Solver{}
	s.Init(nil, nil)

	hebetate := move.NewScoringMoveSimple(0, "11C", "HE....TE", "", alph)
	halitoses := move.NewScoringMoveSimple(0, "12D", "..LIT....", "", alph)
	za := move.NewScoringMoveSimple(0, "H10", "Z.", "", alph)
	it := move.NewScoringMoveSimple(0, "13D", "IT", "", alph)
	its := move.NewScoringMoveSimple(0, "13D", "ITS", "", alph)
	betas := move.NewScoringMoveSimple(0, "11E", "....S", "", alph)
	so := move.NewScoringMoveSimple(0, "I11", "S.", "", alph)
	as := move.NewScoringMoveSimple(0, "J11", "A.", "", alph)
	of := move.NewScoringMoveSimple(0, "I12", ".F", "", alph)
	ar := move.NewScoringMoveSimple(0, "11J", "AR", "", alph)
	sha := move.NewScoringMoveSimple(0, "12C", "S..", "", alph)
	betas.SetDupe(so)
	so.SetDupe(betas)

	testcases := []blocktestcase{
		{halitoses, hebetate, false},
		{hebetate, halitoses, false},
		{za, halitoses, true},
		{halitoses, za, true},
		{it, halitoses, false},
		{halitoses, it, false},
		{its, halitoses, true},
		{halitoses, its, true},
		{betas, as, true},
		{as, betas, true},
		{betas, of, true},
		{of, betas, true},
		{betas, ar, true},
		{ar, betas, true},
		{betas, so, true},
		{so, betas, true},
		{so, as, true},
		{as, so, true},
		{so, of, true},
		{of, so, true},
		{so, ar, true},
		{ar, so, true},
		{sha, halitoses, true},
		{halitoses, sha, true},
		{halitoses, halitoses, true},
		{hebetate, hebetate, true},
		{za, za, true},
		{it, it, true},
		{its, its, true},
		{of, of, true},
		{so, so, true},
		{sha, sha, true},
	}
	for _, tc := range testcases {
		log.Debug().Msgf("trying %v blocks %v, expect %v", tc.stmPlay, tc.otsPlay, tc.blocks)
		is.True(s.blocks(tc.stmPlay, tc.otsPlay, b) == tc.blocks)
	}
	// is.Fail()
}
func TestMoreComplexBlocks(t *testing.T) {
	is := is.New(t)
	row1 := "    BETA      "
	row2 := "   HA   OSES  "
	row3 := "    AMEER     "
	alph := alphabet.EnglishAlphabet()
	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetRow(10, row1, alph)
	b.SetRow(11, row2, alph)
	b.SetRow(12, row3, alph)
	fmt.Println(b.ToDisplayText(alph))
	s := &Solver{}
	s.Init(nil, nil)

	hebetate := move.NewScoringMoveSimple(0, "11C", "HE....TE", "", alph)
	halitoses := move.NewScoringMoveSimple(0, "12D", "..LIT....", "", alph)
	za := move.NewScoringMoveSimple(0, "H10", "Z.", "", alph)
	it := move.NewScoringMoveSimple(0, "14F", "IT", "", alph)
	its := move.NewScoringMoveSimple(0, "14F", "ITS", "", alph)
	betas := move.NewScoringMoveSimple(0, "11E", "....S", "", alph)
	sor := move.NewScoringMoveSimple(0, "I11", "S..", "", alph)
	as := move.NewScoringMoveSimple(0, "J11", "A.", "", alph)
	sha := move.NewScoringMoveSimple(0, "12C", "S..", "", alph)
	ort := move.NewScoringMoveSimple(0, "I12", "..T", "", alph)

	betas.SetDupe(sor)
	sor.SetDupe(betas)

	testcases := []blocktestcase{
		{halitoses, hebetate, false},
		{hebetate, halitoses, false},
		{za, halitoses, true},
		{halitoses, za, true},
		{it, halitoses, true},
		{halitoses, it, true},
		{its, halitoses, true},
		{halitoses, its, true},
		{betas, as, true},
		{as, betas, true},
		{betas, ort, true},
		{ort, betas, true},
		{betas, sor, true},
		{sor, betas, true},
		{sor, as, true},
		{as, sor, true},
		{sor, ort, true},
		{ort, sor, true},
		{hebetate, ort, true},
		{ort, hebetate, true},
		{sha, halitoses, true},
		{halitoses, sha, true},
		{halitoses, halitoses, true},
		{hebetate, hebetate, true},
		{za, za, true},
		{it, it, true},
		{its, its, true},
		{ort, ort, true},
		{sor, sor, true},
		{sha, sha, true},
		{hebetate, it, false},
		{it, hebetate, false},
		{hebetate, its, false},
		{its, hebetate, false},
	}
	for _, tc := range testcases {
		log.Debug().Msgf("trying %v blocks %v, expect %v", tc.stmPlay, tc.otsPlay, tc.blocks)
		is.True(s.blocks(tc.stmPlay, tc.otsPlay, b) == tc.blocks)
	}

}
