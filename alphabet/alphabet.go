package alphabet

import (
	"fmt"
	"log"
	"sort"
)

const (
	// MaxAlphabetSize is the maximum size of the alphabet, and is also
	// the "code" for the separation token.
	// It should be below 64 so that it can fit in one 64-bit word.
	// Gwich'in Scrabble has 62 separate letters, including the blank.
	// Lojban has even more, but that's a weird constructed language.
	MaxAlphabetSize = 50
)

// LetterSet is a bit mask of acceptable letters, with indices from 0 to
// the maximum alphabet size.
type LetterSet uint64

type letterSlice []rune

// MachineLetter is a machine-only representation of a letter. It goes from
// 0 to the maximum alphabet size.
type MachineLetter uint8

// MachineWord is a slice of MachineLetter; it is a machine-only representation
// of a word.
type MachineWord []MachineLetter

// Alphabet defines an alphabet.
type Alphabet struct {
	// vals is a map of the actual physical letter rune (like 'A') to a
	// number representing it, from 0 to MaxAlphabetSize.

	vals map[rune]uint8
	// letters is a map of the 0 to MaxAlphabetSize value back to a letter.
	letters map[uint8]rune

	letterSlice letterSlice
	curIdx      uint8
}

func (a Alphabet) CurIdx() uint8 {
	return a.curIdx
}

// update the alphabet map.
func (a *Alphabet) Update(word string) error {
	for _, char := range word {
		if _, ok := a.vals[char]; !ok {
			a.vals[char] = a.curIdx
			a.curIdx++
		}
	}

	if a.curIdx == MaxAlphabetSize {
		return fmt.Errorf("exceeded max alphabet size")
	}
	return nil
}

// Init initializes the alphabet data structures
func (a *Alphabet) Init() {
	a.vals = make(map[rune]uint8)
	a.letters = make(map[uint8]rune)
}

// Val returns the 'value' of this rune in the alphabet; i.e a number from
// 0 to max size
func (a Alphabet) Val(r rune) (uint8, error) {
	val, ok := a.vals[r]
	if ok {
		return val, nil
	}
	return 0, fmt.Errorf("Letter %v not found in alphabet", r)
}

// Letter returns the letter that this position in the alphabet corresponds to.
func (a Alphabet) Letter(b byte) rune {
	return a.letters[b]
}

func (a Alphabet) Letters() map[uint8]rune {
	return a.letters
}

func (a Alphabet) Vals() map[rune]uint8 {
	return a.vals
}

// NumLetters returns the number of letters in this alphabet.
func (a Alphabet) NumLetters() uint8 {
	return uint8(len(a.letters))
}

func (a *Alphabet) genLetterSlice() {
	a.letterSlice = []rune{}
	for rn := range a.vals {
		a.letterSlice = append(a.letterSlice, rn)
	}
	sort.Sort(a.letterSlice)
	fmt.Println("After sorting", a.letterSlice)
	// These maps are now deterministic. Renumber them according to
	// sort order.
	for idx, rn := range a.letterSlice {
		a.vals[rn] = uint8(idx)
		a.letters[byte(idx)] = rn
	}
}

// Reconcile will take a populated alphabet, sort the glyphs, and re-index
// the numbers.
func (a *Alphabet) Reconcile() {
	fmt.Println("[DEBUG] Reconciling alphabet")
	a.genLetterSlice()
}

// Serialize serializes the alphabet into a slice of 32-bit integers.
func (a *Alphabet) Serialize() []uint32 {
	els := []uint32{}
	// Append the size first, then the individual elements.
	els = append(els, uint32(len(a.letterSlice)))
	for _, rn := range a.letterSlice {
		// Append the rune
		els = append(els, uint32(rn))
	}
	log.Println("[DEBUG] Serializing", els)
	return els
}

func (a letterSlice) Len() int           { return len(a) }
func (a letterSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a letterSlice) Less(i, j int) bool { return a[i] < a[j] }

/*

func (a ArcPtrSlice) Len() int           { return len(a) }
func (a ArcPtrSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ArcPtrSlice) Less(i, j int) bool { return a[i].Letter < a[j].Letter }
*/
