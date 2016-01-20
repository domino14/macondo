package gaddag

import (
	"reflect"
	"testing"
)

func TestGenAlphabet(t *testing.T) {
	gd := GenerateGaddag("test_files/little_spanish.txt", false, false)
	gd.serializeElements()
	if gd.Alphabet.curIdx != 12 {
		t.Errorf("curIdx should be 12, is %v", gd.Alphabet.curIdx)
	}
	if !reflect.DeepEqual(gd.Alphabet.letters, map[byte]rune{
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
		t.Errorf("letters did not match: %v", gd.Alphabet.letters)
	}

	if !reflect.DeepEqual(gd.Alphabet.vals, map[rune]uint32{
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
	}) {
		t.Errorf("vals did not match: %v", gd.Alphabet.vals)
	}
}

func TestGenGaddag(t *testing.T) {
	gd := GenerateGaddag("test_files/little_spanish.txt", false, false)
	gd.serializeElements()
	// 12 elements in the alphabet
	if gd.SerializedElements[0] != 12 {
		t.Errorf("Did not match: %v", gd.SerializedElements[0])
	}

	// a specific alphabet value. it's always [3] because of sorting.
	if gd.SerializedElements[3] != 'C' {
		t.Errorf("Did not match: %c", gd.SerializedElements[3])
	}
	// The number of unique letter sets for this lexicon.
	if gd.SerializedElements[13] != 7 {
		t.Errorf("Did not match: %v", gd.SerializedElements[13])
	}
}

func TestFindWordSmallSpanish(t *testing.T) {
	gd := GenerateGaddag("test_files/little_spanish.txt", false, false)
	gd.serializeElements()
	t.Logf("%#x", gd.SerializedElements)
	for _, word := range []string{"AÑO", "COMER", "COMIDA", "COMIDAS",
		"CO3AL"} {
		found := FindWord(gd.SerializedElements, word)
		if !found {
			t.Errorf("Did not find word %v :(", word)
		}
	}

}

func TestFindWordSmallEnglish(t *testing.T) {
	gd := GenerateGaddag("test_files/dogs.txt", false, false)
	gd.serializeElements()
	t.Logf("%#x", gd.SerializedElements)
	found := FindWord(gd.SerializedElements, "DOG")
	if !found {
		t.Error("Did not find DOG :(")
	}
}

func TestFindWordSmallEnglish2(t *testing.T) {
	gd := GenerateGaddag("test_files/no.txt", false, false)
	gd.serializeElements()
	t.Logf("%#x", gd.SerializedElements)
	found := FindWord(gd.SerializedElements, "NO")
	if !found {
		t.Error("Did not find NO :(")
	}
}
