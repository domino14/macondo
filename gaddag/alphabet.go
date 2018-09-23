package gaddag

import (
	"fmt"
	"log"
	"sort"
)

const (
	// MaxAlphabetSize is the maximum size of the alphabet, and is also
	// the "code" for the separation token.
	MaxAlphabetSize = 31
)

type LetterSlice []rune

// Alphabet defines an alphabet.
// For now, don't create gaddags for alphabets with more than 31 unique
// runes. Our file format will not yet support it.
type Alphabet struct {
	// vals is a map of the actual physical letter rune (like 'A') to a
	// number representing it, from 0 to MaxAlphabetSize.
	// The vals map has uint32 values for simplicity in serialization. It's ok
	// to waste a few bytes here and there...
	vals map[rune]uint32
	// letters is a map of the 0 to MaxAlphabetSize value back to a letter.
	letters map[byte]rune

	letterSlice LetterSlice
	curIdx      uint32
	// true if alphabet contains just A through Z with no gaps in between.
	athruz bool
}

// update the alphabet map.
func (a *Alphabet) update(word string) error {
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

func (a *Alphabet) Init() {
	a.vals = make(map[rune]uint32)
	a.letters = make(map[byte]rune)
}

// Return the 'value' of this rune in the alphabet; i.e. a number from
// 0 to 31
func (a Alphabet) Val(r rune) (uint32, error) {
	val, ok := a.vals[r]
	if ok {
		return val, nil
	}
	return 0, fmt.Errorf("Letter %v not found in alphabet", r)
}

// Return the letter that this position in the alphabet corresponds to.
func (a Alphabet) Letter(b byte) rune {
	return a.letters[b]
}

// Return the number of letters in this alphabet.
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
		a.vals[rn] = uint32(idx)
		a.letters[byte(idx)] = rn
	}
}

// Reconcile will take a populated alphabet, sort the glyphs, and re-index
// the numbers.
func (a *Alphabet) Reconcile() {
	fmt.Println("[DEBUG] Reconciling alphabet")
	a.genLetterSlice()
}

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

func (a LetterSlice) Len() int           { return len(a) }
func (a LetterSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a LetterSlice) Less(i, j int) bool { return a[i] < a[j] }

/*

func (a ArcPtrSlice) Len() int           { return len(a) }
func (a ArcPtrSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ArcPtrSlice) Less(i, j int) bool { return a[i].Letter < a[j].Letter }
*/
