package zobrist

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/move"
	"github.com/matryer/is"
)

var DefaultConfig = config.DefaultConfig()

func TestPlayAndUnplay(t *testing.T) {
	is := is.New(t)
	z := &Zobrist{}
	z.Initialize(15)

	endgameCGP := "1LEMNISCI2L1ER/7O1PAINT1/4A2L1RAVE2/WEDGE2Z1I1R3/4R1JAUNTEd2/4OXO2K5/2YOB3P6/3FAUNAE6/4T3GUY4/6BESTEaD2/7T2HIE2/7H4VUG/2CORMOID6/7O7/7NONIDEAL AAFIRTW/EIQSS 373/393 0 lex NWL20;"

	g, err := cgp.ParseCGP(&DefaultConfig, endgameCGP)
	is.NoErr(err)
	alph := alphabet.EnglishAlphabet()
	h := z.Hash(g.Board().GetSquares(), alphabet.RackFromString("AAFIRTW", alph), false, true)
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

	endgameCGP := "1LEMNISCI2L1ER/7O1PAINT1/4A2L1RAVE2/WEDGE2Z1I1R3/4R1JAUNTEd2/4OXO2K5/2YOB3P6/3FAUNAE6/4T3GUY4/6BESTEaD2/7T2HIE2/7H4VUG/2CORMOID6/7O7/7NONIDEAL AAFIRTW/EIQSS 373/393 0 lex NWL20;"

	g, err := cgp.ParseCGP(&DefaultConfig, endgameCGP)
	is.NoErr(err)
	alph := alphabet.EnglishAlphabet()
	h := z.Hash(g.Board().GetSquares(), alphabet.RackFromString("AAFIRTW", alph), false, true)

	passrack, err := alphabet.ToMachineWord("EIQSS", alph)
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
