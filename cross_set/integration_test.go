package cross_set_test

import (
	"log"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/crosses"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

var DefaultConfig = config.DefaultConfig()

type updateCrossesForMoveTestCase struct {
	testGame        board.VsWho
	m               *move.Move
	userVisibleWord string
}

func TestUpdateCrossSetsAndAnchorsForMove(t *testing.T) {
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	is.NoErr(err)
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	gen := cross_set.GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
	alph := dist.Alphabet()

	var testCases = []updateCrossesForMoveTestCase{
		{board.VsMatt, move.NewScoringMoveSimple(38, "K9", "TAEL", "ABD", alph), "TAEL"},
		// Test right edge of board
		{board.VsMatt2, move.NewScoringMoveSimple(77, "O8", "TENsILE", "", alph), "TENsILE"},
		// Test through tiles
		{board.VsOxy, move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph),
			"OXYPHENBUTAZONE"},
		// Test top of board, horizontal
		{board.VsJeremy, move.NewScoringMoveSimple(14, "1G", "S.oWED", "D?", alph), "SNoWED"},
		// Test bottom of board, horizontal
		{board.VsJeremy, move.NewScoringMoveSimple(11, "15F", "F..ER", "", alph), "FOYER"},
	}

	// create a move.
	for tidx, tc := range testCases {
		b := board.MakeBoard(board.CrosswordGameBoard)
		b.SetToGame(alph, tc.testGame)
		bcs := cross_set.MakeBoardCrossSets(b)
		a := movegen.MakeAnchors(b)
		gen.CS = bcs
		gen.GenerateAll(b)
		a.UpdateAllAnchors()

		b.PlayMove(tc.m, dist)
		gen.UpdateForMove(b, tc.m)
		a.UpdateAnchorsForMove(tc.m)
		log.Println(b.ToDisplayText(alph))
		// Create an identical board, but generate cross-sets and anchors
		// for the entire board after placing the letters "manually".
		c := board.MakeBoard(board.CrosswordGameBoard)
		c.SetToGame(alph, tc.testGame)
		cs2 := cross_set.MakeBoardCrossSets(c)
		a2 := movegen.MakeAnchors(c)
		c.PlaceMoveTiles(tc.m)
		c.TestSetTilesPlayed(c.GetTilesPlayed() + tc.m.TilesPlayed())
		cross_set.GenAllCrossSets(c, cs2, gd, dist)
		a2.UpdateAllAnchors()

		assert.True(t, b.Equals(c))
		assert.True(t, cs2.Equals(bcs))
		assert.True(t, a.Equals(a2), "failed anchor-eq: %d", tidx)

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
			uv := b.GetLetter(row+rowInc, col+colInc).UserVisible(alph)
			assert.Equal(t, c, uv)
		}
	}
}

// Copy of TestUpdateCrossSetsForMove with the CrossScoreOnlyGenerator
func TestUpdateCrossScoresForMove(t *testing.T) {
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	gen := crosses.CrossScoreOnlyGenerator{Dist: dist}
	alph := dist.Alphabet()

	var testCases = []updateCrossesForMoveTestCase{
		{board.VsMatt, move.NewScoringMoveSimple(38, "K9", "TAEL", "ABD", alph), "TAEL"},
		// Test right edge of board
		{board.VsMatt2, move.NewScoringMoveSimple(77, "O8", "TENsILE", "", alph), "TENsILE"},
		// Test through tiles
		{board.VsOxy, move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph),
			"OXYPHENBUTAZONE"},
		// Test top of board, horizontal
		{board.VsJeremy, move.NewScoringMoveSimple(14, "1G", "S.oWED", "D?", alph), "SNoWED"},
		// Test bottom of board, horizontal
		{board.VsJeremy, move.NewScoringMoveSimple(11, "15F", "F..ER", "", alph), "FOYER"},
	}

	// create a move.
	for _, tc := range testCases {
		b := board.MakeBoard(board.CrosswordGameBoard)
		b.SetToGame(alph, tc.testGame)
		a := movegen.MakeAnchors(b)

		gen.GenerateAll(b)
		a.UpdateAllAnchors()

		b.PlayMove(tc.m, dist)
		gen.UpdateForMove(b, tc.m)

		// Create an identical board, but generate cross-sets for the entire
		// board after placing the letters "manually".
		c := board.MakeBoard(board.CrosswordGameBoard)
		a2 := movegen.MakeAnchors(c)
		c.SetToGame(alph, tc.testGame)
		c.PlaceMoveTiles(tc.m)
		c.TestSetTilesPlayed(c.GetTilesPlayed() + tc.m.TilesPlayed())
		crosses.GenAllCrossScores(c, dist)
		a2.UpdateAllAnchors()

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
			uv := b.GetLetter(row+rowInc, col+colInc).UserVisible(alph)
			assert.Equal(t, c, uv)
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
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	is.NoErr(err)
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	gen1 := cross_set.GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
	gen2 := crosses.CrossScoreOnlyGenerator{Dist: dist}
	alph := dist.Alphabet()

	var testCases = []updateCrossesForMoveTestCase{
		{board.VsMatt, move.NewScoringMoveSimple(38, "K9", "TAEL", "ABD", alph), "TAEL"},
		// Test right edge of board
		{board.VsMatt2, move.NewScoringMoveSimple(77, "O8", "TENsILE", "", alph), "TENsILE"},
		// Test through tiles
		{board.VsOxy, move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph),
			"OXYPHENBUTAZONE"},
		// Test top of board, horizontal
		{board.VsJeremy, move.NewScoringMoveSimple(14, "1G", "S.oWED", "D?", alph), "SNoWED"},
		// Test bottom of board, horizontal
		{board.VsJeremy, move.NewScoringMoveSimple(11, "15F", "F..ER", "", alph), "FOYER"},
	}

	// create a move.
	for _, tc := range testCases {
		// Run the cross set generator on b1
		b1 := board.MakeBoard(board.CrosswordGameBoard)
		b1.SetToGame(alph, tc.testGame)
		gen1.CS = cross_set.MakeBoardCrossSets(b1)
		gen1.GenerateAll(b1)
		a := movegen.MakeAnchors(b1)
		a.UpdateAllAnchors()

		b1.PlayMove(tc.m, dist)
		gen1.UpdateForMove(b1, tc.m)

		// Run the cross score generator on b2
		b2 := board.MakeBoard(board.CrosswordGameBoard)
		b2.SetToGame(alph, tc.testGame)
		gen2.GenerateAll(b2)
		a2 := movegen.MakeAnchors(b2)
		a2.UpdateAllAnchors()
		b2.PlayMove(tc.m, dist)
		gen2.UpdateForMove(b2, tc.m)

		compareCrossScores(t, b1, b2)
	}
}

func TestCompareGenAll(t *testing.T) {
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	is.NoErr(err)
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	alph := dist.Alphabet()

	var testCases = []board.VsWho{
		board.VsEd,
		board.VsJeremy,
		board.VsMatt,
		board.VsMatt2,
		board.VsOxy,
	}

	for _, tc := range testCases {
		b1 := board.MakeBoard(board.CrosswordGameBoard)
		b1.SetToGame(alph, tc)
		bcs := cross_set.MakeBoardCrossSets(b1)
		cross_set.GenAllCrossSets(b1, bcs, gd, dist)

		b2 := board.MakeBoard(board.CrosswordGameBoard)
		b2.SetToGame(alph, tc)
		crosses.GenAllCrossScores(b2, dist)

		compareCrossScores(t, b1, b2)
	}
}

// Benchmarks

func BenchmarkGenAnchorsAndCrossSets(b *testing.B) {
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	is.NoErr(err)
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	alph := dist.Alphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	bd.SetToGame(alph, board.VsOxy)
	bcs := cross_set.MakeBoardCrossSets(bd)
	a := movegen.MakeAnchors(bd)

	b.ResetTimer()

	// 38 us
	for i := 0; i < b.N; i++ {
		a.UpdateAllAnchors()
		cross_set.GenAllCrossSets(bd, bcs, gd, dist)
	}
}

func BenchmarkMakePlay(b *testing.B) {
	// Mostly, benchmark the progressive generation of anchors and cross-sets
	// (as opposed to generating all of them from scratch)
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	is.NoErr(err)
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	gen := cross_set.GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
	alph := dist.Alphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	bd.SetToGame(alph, board.VsMatt)
	gen.CS = cross_set.MakeBoardCrossSets(bd)
	gen.GenerateAll(bd)
	a := movegen.MakeAnchors(bd)
	a.UpdateAllAnchors()

	// create a move.
	m := move.NewScoringMoveSimple(38, "K9", "TAEL", "ABD", alph)

	b.ResetTimer()
	// 2.7 us; more than 10x faster than regenerating all anchors every time.
	// seems worth it.
	for i := 0; i < b.N; i++ {
		bd.PlayMove(m, dist)
		gen.UpdateForMove(bd, m)
		a.UpdateAnchorsForMove(m)
	}

}
