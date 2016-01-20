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

// This file defines an alphabet.
// For now, don't create gaddags for alphabets with more than 31 unique
// runes. Our file format will not yet support it.
// The vals map has uint32 values for simplicity in serialization. It's ok
// to waste a few bytes here and there...
type Alphabet struct {
	vals        map[rune]uint32
	letters     map[byte]rune
	letterSlice LetterSlice
	curIdx      uint32
}

// update the alphabet map.
func (a *Alphabet) update(word string) error {
	for _, char := range word {
		if _, ok := a.vals[char]; !ok {
			a.vals[char] = a.curIdx
			a.curIdx += 1
		}
	}

	if a.curIdx == MaxAlphabetSize {
		return fmt.Errorf("Exceeded max alphabet size.")
	}
	return nil
}

func (a *Alphabet) Init() {
	a.vals = make(map[rune]uint32)
	a.letters = make(map[byte]rune)
}

func (a *Alphabet) genLetterSlice() {
	a.letterSlice = []rune{}
	for rn, _ := range a.vals {
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
