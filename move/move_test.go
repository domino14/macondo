package move

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/matryer/is"
)

type coordTestStruct struct {
	row      int
	col      int
	vertical bool
	output   string
}

var coordTests = []coordTestStruct{
	{0, 0, false, "1A"},
	{0, 0, true, "A1"},
	{14, 14, false, "15O"},
	{14, 14, true, "O15"},
	{9, 8, false, "10I"},
	{9, 8, true, "I10"},
	{1, 7, false, "2H"},
	{1, 7, true, "H2"},
}

func TestToBoardGameCoords(t *testing.T) {
	for _, tc := range coordTests {
		calc := ToBoardGameCoords(tc.row, tc.col, tc.vertical)
		if calc != tc.output {
			t.Errorf("For row=%v col=%v vertical=%v got %v, expected %v",
				tc.row, tc.col, tc.vertical, calc, tc.output)
		}
	}
}

func TestFromBoardGameCoords(t *testing.T) {
	for _, tc := range coordTests {
		row, col, vertical := FromBoardGameCoords(tc.output)
		if row != tc.row || col != tc.col || vertical != tc.vertical {
			t.Errorf("For coord %v expected (%v, %v, %v) got (%v, %v, %v)",
				tc.output, tc.row, tc.col, tc.vertical, row, col, vertical)
		}
	}
}

func TestEquals(t *testing.T) {
	is := is.New(t)
	m1 := NewScoringMoveSimple(35, "A7", "HELLO", "QI", alphabet.EnglishAlphabet())
	m2 := NewScoringMoveSimple(35, "A7", "HELLO", "QI", alphabet.EnglishAlphabet())
	is.True(m1.Equals(m2, false, false))
	m3 := NewScoringMoveSimple(35, "A7", "HELLO", "Q?", alphabet.EnglishAlphabet())
	is.True(!m1.Equals(m3, false, false))
}

func TestEqualsWithTransposition(t *testing.T) {
	is := is.New(t)
	m1 := NewScoringMoveSimple(66, "H8", "TERTIAL", "", alphabet.EnglishAlphabet())
	m2 := NewScoringMoveSimple(66, "8H", "TERTIAL", "", alphabet.EnglishAlphabet())
	is.True(!m1.Equals(m2, false, false))
	is.True(m1.Equals(m2, true, false))

	m3 := NewScoringMoveSimple(24, "8H", "PHEW", "", alphabet.EnglishAlphabet())
	m4 := NewScoringMoveSimple(24, "8F", "PHEW", "", alphabet.EnglishAlphabet())
	is.True(!m3.Equals(m4, true, false))

}

func TestEqualsWithLeaveIgnore(t *testing.T) {
	is := is.New(t)
	m1 := NewScoringMoveSimple(66, "H8", "WHAT", "ABC", alphabet.EnglishAlphabet())
	m2 := NewScoringMoveSimple(66, "8H", "WHAT", "F", alphabet.EnglishAlphabet())
	is.True(!m1.Equals(m2, false, false))
	is.True(!m1.Equals(m2, false, true))
	is.True(m1.Equals(m2, true, true))
	is.True(!m1.Equals(m2, true, false))
}
