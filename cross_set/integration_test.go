package cross_set_test

import (
	"log"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/cgboard"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

var DefaultConfig = config.DefaultConfig()

type updateCrossesForMoveTestCase struct {
	testGame        cgboard.VsWho
	m               *move.Move
	userVisibleWord string
}

func TestUpdateCrossSetsAndAnchorsForMove(t *testing.T) {
	path := filepath.Join(DefaultConfig.LexiconPath, "gaddag", "America.gaddag")
	gd, err := gaddag.LoadGaddag(path)
	if err != nil {
		t.Error(err)
	}
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	gen := cross_set.GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
	alph := dist.Alphabet()

	var testCases = []updateCrossesForMoveTestCase{
		{cgboard.VsMatt, move.NewScoringMoveSimple(38, "K9", "TAEL", "ABD", alph), "TAEL"},
		// Test right edge of board
		{cgboard.VsMatt2, move.NewScoringMoveSimple(77, "O8", "TENsILE", "", alph), "TENsILE"},
		// Test through tiles
		{cgboard.VsOxy, move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph),
			"OXYPHENBUTAZONE"},
		// Test top of board, horizontal
		{cgboard.VsJeremy, move.NewScoringMoveSimple(14, "1G", "S.oWED", "D?", alph), "SNoWED"},
		// Test bottom of board, horizontal
		{cgboard.VsJeremy, move.NewScoringMoveSimple(11, "15F", "F..ER", "", alph), "FOYER"},
	}

	// create a move.
	for tidx, tc := range testCases {
		b := cgboard.MakeBoard(cgboard.CrosswordGameBoard)
		b.SetToGame(alph, tc.testGame)
		bcs := cross_set.MakeBoardCrossSets(b)
		a := movegen.MakeAnchors(b)

		gen.GenerateAll(b, bcs)
		a.UpdateAllAnchors()

		b.PlayMove(tc.m, dist)
		gen.UpdateForMove(b, bcs, tc.m)
		a.UpdateAnchorsForMove(tc.m)
		log.Println(b.ToDisplayText(alph))
		// Create an identical board, but generate cross-sets and anchors
		// for the entire board after placing the letters "manually".
		c := cgboard.MakeBoard(cgboard.CrosswordGameBoard)
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
