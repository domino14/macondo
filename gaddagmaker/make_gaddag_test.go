package gaddagmaker

import (
	"reflect"
	"testing"

	"github.com/domino14/macondo/alphabet"
)

func TestGenAlphabet(t *testing.T) {
	gd := GenerateGaddag("test_files/little_spanish.txt", false, false)
	gd.SerializeElements()
	if gd.Alphabet.CurIdx() != 12 {
		t.Errorf("curIdx should be 12, is %v", gd.Alphabet.CurIdx())
	}
	if !reflect.DeepEqual(gd.Alphabet.Letters(), map[alphabet.MachineLetter]rune{
		0:  '3',
		1:  'A',
		2:  'C',
		3:  'D',
		4:  'E',
		5:  'I',
		6:  'L',
		7:  'M',
		8:  'O',
		9:  'R',
		10: 'S',
		11: 'Ñ',
	}) {
		t.Errorf("letters did not match: %v", gd.Alphabet.Letters())
	}

	expectedMap := map[rune]alphabet.MachineLetter{
		'3': 0,
		'A': 1,
		'C': 2,
		'D': 3,
		'E': 4,
		'I': 5,
		'L': 6,
		'M': 7,
		'O': 8,
		'R': 9,
		'S': 10,
		'Ñ': 11,
	}

	if !reflect.DeepEqual(gd.Alphabet.Vals(), expectedMap) {
		t.Errorf("vals did not match: got %v expected %v", gd.Alphabet.Vals(), expectedMap)
	}
}

func TestGenGaddag(t *testing.T) {
	gd := GenerateGaddag("test_files/little_spanish.txt", false, false)
	gd.SerializeElements()
	// 12 elements in the alphabet
	if gd.SerializedAlphabet[0] != 12 {
		t.Errorf("Did not match: %v", gd.SerializedAlphabet[0])
	}

	// a specific alphabet value. it's always [3] because of sorting.
	if gd.SerializedAlphabet[3] != 'C' {
		t.Errorf("Did not match: %c", gd.SerializedAlphabet[3])
	}
	// The number of unique letter sets for this lexicon.
	if gd.NumLetterSets != 7 {
		t.Errorf("Did not match: %v", gd.NumLetterSets)
	}
}

func TestGenGaddagMinimize(t *testing.T) {
	gd := GenerateGaddag("test_files/little_spanish.txt", true, false)
	gd.SerializeElements()
	// 12 elements in the alphabet
	if gd.SerializedAlphabet[0] != 12 {
		t.Errorf("Did not match: %v", gd.SerializedAlphabet[0])
	}

	// a specific alphabet value. it's always [3] because of sorting.
	if gd.SerializedAlphabet[3] != 'C' {
		t.Errorf("Did not match: %c", gd.SerializedAlphabet[3])
	}
	// The number of unique letter sets for this lexicon.
	if gd.NumLetterSets != 7 {
		t.Errorf("Did not match: %v", gd.NumLetterSets)
	}
}

// func TestGenNo(t *testing.T) {
// 	gd := GenerateGaddag("test_files/no.txt", false, false)
// 	gd.Save("no.gaddag", GaddagMagicNumber)
// }
