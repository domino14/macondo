package zobrist

import (
	"fmt"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/testhelpers"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/tinymove/conversions"
)

var DefaultConfig = config.DefaultConfig()

/*
func TestPlayAndUnplay(t *testing.T) {
	is := is.New(t)
	z := &Zobrist{}
	z.Initialize(15)

	endgameCGP := "1LEMNISCI2L1ER/7O1PAINT1/4A2L1RAVE2/WEDGE2Z1I1R3/4R1JAUNTEd2/4OXO2K5/2YOB3P6/3FAUNAE6/4T3GUY4/6BESTEaD2/7T2HIE2/7H4VUG/2CORMOID6/7O7/7NONIDEAL AAFIRTW/EIQSS 373/393 0 lex NWL18;"

	g, err := cgp.ParseCGP(&DefaultConfig, endgameCGP)
	is.NoErr(err)
	alph := testhelpers.EnglishAlphabet()
	h := z.Hash(g.Board().GetSquares(), tilemapping.RackFromString("AAFIRTW", alph), false)
	m := move.NewScoringMoveSimple(18, "2A", "WAR", "AFIT", alph)
	// play and unplay a move. The final hash should be the same as the beginning hash.
	h1 := z.AddMove(h, m, false)
	h2 := z.AddMove(h1, m, true)
	is.Equal(h, h2)
	is.True(h1 != h2) // extremely unlikely to collide, but this is not technically always true.
}

func TestPlayAndUnplayMoreLevels(t *testing.T) {
	is := is.New(t)
	z := &Zobrist{}
	z.Initialize(15)

	endgameCGP := "1LEMNISCI2L1ER/7O1PAINT1/4A2L1RAVE2/WEDGE2Z1I1R3/4R1JAUNTEd2/4OXO2K5/2YOB3P6/3FAUNAE6/4T3GUY4/6BESTEaD2/7T2HIE2/7H4VUG/2CORMOID6/7O7/7NONIDEAL AAFIRTW/EIQSS 373/393 0 lex NWL18;"

	g, err := cgp.ParseCGP(&DefaultConfig, endgameCGP)
	is.NoErr(err)
	alph := testhelpers.EnglishAlphabet()
	h := z.Hash(g.Board().GetSquares(), tilemapping.RackFromString("AAFIRTW", alph), false)

	passrack, err := tilemapping.ToMachineWord("EIQSS", alph)
	is.NoErr(err)
	m1 := move.NewScoringMoveSimple(18, "2A", "WAR", "AFIT", alph)
	m2 := move.NewPassMove(passrack, alph)
	m3 := move.NewScoringMoveSimple(10, "M5", ".AFT", "I", alph)
	h1 := z.AddMove(h, m1, false)
	h2 := z.AddMove(h1, m2, false)
	h3 := z.AddMove(h2, m3, false)
	// And unplay these moves in reverse order.
	is.True(h3 != h)
	h4 := z.AddMove(h3, m3, true)
	is.Equal(h2, h4)
	h5 := z.AddMove(h4, m2, true)
	is.Equal(h5, h1)
	h6 := z.AddMove(h5, m1, true)
	is.Equal(h6, h)
}
*/

func TestHashAfterMakingPlay(t *testing.T) {
	is := is.New(t)
	z := &Zobrist{}
	z.Initialize(15)

	endgameCGP := "14C/13QI/12FIE/10VEE1R/9KIT2G/8CIG1IDE/8UTA2AS/7ST1SYPh1/6JA5A1/5WOLD2BOBA/3PLOT1R1NU1EX/Y1VEIN1NOR1mOA1/UT1AT1N1L2FEH1/GUR2WIRER5/SNEEZED8 ADENOOO/AHIILMM 353/236 0 lex CSW19;"

	g, err := cgp.ParseCGP(DefaultConfig, endgameCGP)
	is.NoErr(err)
	alph := testhelpers.EnglishAlphabet()
	h := z.Hash(g.Board().GetSquares(), tilemapping.RackFromString("ADENOOO", alph), tilemapping.RackFromString("AHIILMM", alph), false, 0)

	m1 := move.NewScoringMoveSimple(8, "15J", "END", "AOOO", alph)
	tm := conversions.MoveToTinyMove(m1)
	sm := tinymove.TilePlayMove(tm, 8, 3, 3)
	h1 := z.AddMove(h, &sm, tilemapping.RackFromString("AOOO", alph), &[21]tilemapping.MachineLetter{
		5, 14, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}, true, 0, 0)

	// Actually play the move on the board.
	err = g.PlayMove(m1, false, 0)
	is.NoErr(err)

	h2 := z.Hash(g.Board().GetSquares(), tilemapping.RackFromString("AOOO", alph), tilemapping.RackFromString("AHIILMM", alph), true, 0)
	is.Equal(h1, h2)
}

func TestHashAfterMakingBlankPlay(t *testing.T) {
	is := is.New(t)
	z := &Zobrist{}
	z.Initialize(15)

	endgameCGP := "IBADAT1B7/2CAFE1OD1TRANQ/2TUT2RENIED2/3REV2YOMIM2/4RAFT1NISI2/5COR2N1x2/6LA1AGEE2/6LIAISED2/5POKY2W3/4JOWS7/V2LUZ9/ORPIN10/L1OE11/TUX12/I14 EEEEGH?/AGHNOSU 308/265 0 lex CSW21;"

	g, err := cgp.ParseCGP(DefaultConfig, endgameCGP)
	is.NoErr(err)
	alph := testhelpers.EnglishAlphabet()
	h := z.Hash(g.Board().GetSquares(), tilemapping.RackFromString("EEEEGH?", alph), tilemapping.RackFromString("AGHNOSU", alph), false, 0)

	m1 := move.NewScoringMoveSimple(0, "6M", "xu", "EEEEGH", alph)
	tm := conversions.MoveToTinyMove(m1)
	sm := tinymove.TilePlayMove(tm, 0, 1, 2)
	h1 := z.AddMove(h, &sm, tilemapping.RackFromString("EEEEGH", alph), &[21]tilemapping.MachineLetter{
		0, 21 | 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}, true, 0, 0)

	// Actually play the move on the board.
	err = g.PlayMove(m1, false, 0)
	is.NoErr(err)

	h2 := z.Hash(g.Board().GetSquares(), tilemapping.RackFromString("EEEEGH", alph), tilemapping.RackFromString("AGHNOSU", alph), true, 0)
	is.Equal(h1, h2)
}

func TestHashAfterPassing(t *testing.T) {
	is := is.New(t)
	z := &Zobrist{}
	z.Initialize(15)

	endgameCGP := "14C/13QI/12FIE/10VEE1R/9KIT2G/8CIG1IDE/8UTA2AS/7ST1SYPh1/6JA5A1/5WOLD2BOBA/3PLOT1R1NU1EX/Y1VEIN1NOR1mOA1/UT1AT1N1L2FEH1/GUR2WIRER5/SNEEZED8 ADENOOO/AHIILMM 353/236 0 lex CSW19;"

	g, err := cgp.ParseCGP(DefaultConfig, endgameCGP)
	is.NoErr(err)
	alph := testhelpers.EnglishAlphabet()
	h := z.Hash(g.Board().GetSquares(), tilemapping.RackFromString("ADENOOO", alph), tilemapping.RackFromString("AHIILMM", alph), false, 0)

	m1 := move.NewPassMove(tilemapping.RackFromString("ADENOOO", alph).TilesOn(), alph)
	sm := tinymove.PassMove()

	h1 := z.AddMove(h, &sm, tilemapping.RackFromString("ADENOOO", alph), &[21]tilemapping.MachineLetter{}, true, 1, 0)
	err = g.PlayMove(m1, false, 0)
	is.NoErr(err)

	h2 := z.Hash(g.Board().GetSquares(), tilemapping.RackFromString("ADENOOO", alph), tilemapping.RackFromString("AHIILMM", alph), true, 1)
	// Should be equal because of the number of scoreless turns.
	is.Equal(h1, h2)

	// another pass
	m2 := move.NewPassMove(tilemapping.RackFromString("AHIILMM", alph).TilesOn(), alph)
	sm2 := tinymove.PassMove()
	h3 := z.AddMove(h2, &sm2, tilemapping.RackFromString("AHIILMM", alph), &[21]tilemapping.MachineLetter{}, false, 2, 1)
	err = g.PlayMove(m2, false, 0)
	is.NoErr(err)
	// should NOT be equal to the very first hash.
	is.True(h != h3)

}

func TestHashAfterMakingAnotherPlay(t *testing.T) {
	is := is.New(t)
	z := &Zobrist{}
	z.Initialize(15)

	endgameCGP := "14C/13QI/12FIE/10VEE1R/9KIT2G/8CIG1IDE/8UTA2AS/7ST1SYPh1/6JA5AM/5WOLD2BOBA/3PLOT1R1NU1EX/Y1VEIN1NOR1mOAI/UT1AT1N1L2FEHM/GUR2WIRER4A/SNEEZED2END2L AOOO/HI 353/236 0 lex CSW19;"

	g, err := cgp.ParseCGP(DefaultConfig, endgameCGP)
	is.NoErr(err)
	alph := testhelpers.EnglishAlphabet()
	h := z.Hash(g.Board().GetSquares(), tilemapping.RackFromString("AOOO", alph), tilemapping.RackFromString("HI", alph), false, 0)
	fmt.Println(g.ToDisplayText())
	m1 := move.NewScoringMoveSimple(11, "13G", ".O.O", "AO", alph)
	tm1 := conversions.MoveToTinyMove(m1)
	sm1 := tinymove.TilePlayMove(tm1, 11, 2, 4)
	h1 := z.AddMove(h, &sm1, tilemapping.RackFromString("AO", alph),
		&[21]tilemapping.MachineLetter{
			0, 15, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		}, true, 0, 0)

	// Actually play the move on the board.
	err = g.PlayMove(m1, false, 0)
	is.NoErr(err)
	fmt.Println("game2")
	fmt.Println(g.ToDisplayText())

	h2 := z.Hash(g.Board().GetSquares(), tilemapping.RackFromString("AO", alph), tilemapping.RackFromString("HI", alph), true, 0)
	is.Equal(h1, h2)
}
