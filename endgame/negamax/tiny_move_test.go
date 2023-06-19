package negamax

import (
	"testing"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
	"github.com/matryer/is"
)

func TestTinyMove(t *testing.T) {
	is := is.New(t)
	deepEndgame := "IBADAT1B7/2CAFE1OD1TRANQ/2TUT2RENIED2/3REV2YOMIM2/4RAFT1NISI2/5COR2N1x2/6LA1AGEE2/6LIAISED2/5POKY2W3/4JOWS7/V2LUZ9/ORPIN10/L1OE11/TUX12/I14 EEEEGH?/AGHNOSU 308/265 0 lex CSW19;"
	g, err := cgp.ParseCGP(&DefaultConfig, deepEndgame)
	is.NoErr(err)

	m := move.NewScoringMove(0, []tilemapping.MachineLetter{0, 21 | 0x80},
		nil, false, 1, g.Alphabet(), 5, 12)

	tm := moveToTinyMove(m)

	m2 := tinyMoveToMove(tm, g.Board())
	is.True(minimallyEqual(m, m2))
}

func TestComplexTinyMove(t *testing.T) {
	is := is.New(t)
	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetToGame(tilemapping.EnglishAlphabet(), board.VsMatt2)

	m := move.NewScoringMove(0,
		[]tilemapping.MachineLetter{5, 0, 4 | 0x80, 0, 5},
		nil, true, 3, tilemapping.EnglishAlphabet(), 9, 9)

	tm := moveToTinyMove(m)

	m2 := tinyMoveToMove(tm, b)
	is.True(minimallyEqual(m, m2))
}
