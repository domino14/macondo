package alphabet

import (
	"reflect"
	"testing"
)

func TestUserVisible(t *testing.T) {
	// Initialize an alphabet.
	alph := &Alphabet{}
	alph.Init()
	alph.Update("AEROLITH")
	alph.Update("HOMEMADE")
	alph.Update("GAMODEME")
	alph.Update("XU")
	alph.Reconcile()
	expected := LetterSlice([]rune{
		'A', 'D', 'E', 'G', 'H', 'I', 'L', 'M', 'O', 'R', 'T', 'U', 'X'})
	if !reflect.DeepEqual(alph.letterSlice, expected) {
		t.Errorf("Did not equal, expected %v got %v", expected, alph.letterSlice)
	}
	mw := MachineWord([]MachineLetter{4, 8, 7, 5, 2})
	uv := mw.UserVisible(alph)
	if uv != "HOMIE" {
		t.Errorf("Did not equal, expected %v got %v", "HOMIE", uv)
	}

	mw2 := MachineWord([]MachineLetter{12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0})
	uv2 := mw2.UserVisible(alph)
	if uv2 != "XUTROMLIHGEDA" {
		t.Errorf("Did not equal, expected %v got %v", "XUTROMLIHGEDA", uv2)
	}
}
