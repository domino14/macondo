package alphabet

import (
	"fmt"
	"log"
	"sort"
	"unicode"
)

const (
	// MaxAlphabetSize is the maximum size of the alphabet, and is also
	// the "code" for the separation token.
	// It should be below 64 so that it can fit in one 64-bit word.
	// Gwich'in Scrabble has 62 separate letters, including the blank.
	// Lojban has even more, but that's a weird constructed language.
	MaxAlphabetSize = 50
	// SeparationMachineLetter is the "MachineLetter" corresponding to
	// the separation token. It is set at the max alphabet size.
	SeparationMachineLetter = MaxAlphabetSize
	// BlankOffset is the offset at which letters with a code >= offset
	// represent blanks.
	BlankOffset = 100
	// SeparationToken is the GADDAG separation token.
	SeparationToken = '^'
)

// LetterSet is a bit mask of acceptable letters, with indices from 0 to
// the maximum alphabet size.
type LetterSet uint64

// LetterSlice is a slice of runes. We make it a separate type for ease in
// defining sort functions on it.
type LetterSlice []rune

// MachineLetter is a machine-only representation of a letter. It goes from
// 0 to the maximum alphabet size.
type MachineLetter uint8

// Blank blankifies a letter; i.e. it adds the BlankOffset to it.
// It returns a new MachineLetter; it does not modify the one that is passed
// in!
func (ml MachineLetter) Blank() MachineLetter {
	return ml + BlankOffset
}

// UserVisible turns the passed-in machine letter into a user-visible string.
func (ml MachineLetter) UserVisible(alph *Alphabet) string {
	return MachineWord(string(ml)).UserVisible(alph)
}

// MachineWord is a string; it is a machine-only representation of a word.
// The individual runes in the string are not user-readable; they start at 0.
// We use a string instead of a []MachineLetter to give us some of the syntactic
// sugar and optimizations that Golang strings use.
type MachineWord string

// UserVisible turns the passed-in machine word into a user-visible string.
func (mw MachineWord) UserVisible(alph *Alphabet) string {
	runes := make([]rune, len(mw))
	for i, l := range mw {
		if l >= BlankOffset {
			runes[i] = unicode.ToLower(alph.Letter(MachineLetter(l - BlankOffset)))
		} else {
			runes[i] = alph.Letter(MachineLetter(l))
		}
	}
	return string(runes)
}

// Alphabet defines an alphabet.
type Alphabet struct {
	// vals is a map of the actual physical letter rune (like 'A') to a
	// number representing it, from 0 to MaxAlphabetSize.

	vals map[rune]MachineLetter
	// letters is a map of the 0 to MaxAlphabetSize value back to a letter.
	letters map[MachineLetter]rune

	letterSlice LetterSlice
	curIdx      MachineLetter
}

func (a Alphabet) CurIdx() MachineLetter {
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
	a.vals = make(map[rune]MachineLetter)
	a.letters = make(map[MachineLetter]rune)
}

// Val returns the 'value' of this rune in the alphabet; i.e a number from
// 0 to maxsize + blank offset. Takes into account blanks (lowercase letters).
func (a Alphabet) Val(r rune) (MachineLetter, error) {
	if r == SeparationToken {
		return SeparationMachineLetter, nil
	}
	val, ok := a.vals[r]
	if ok {
		return val, nil
	}
	if r == unicode.ToLower(r) {
		val, ok = a.vals[unicode.ToUpper(r)]
		if ok {
			return val + BlankOffset, nil
		}
	}
	return 0, fmt.Errorf("Letter `%v` not found in alphabet", r)
}

// Letter returns the letter that this position in the alphabet corresponds to.
func (a Alphabet) Letter(b MachineLetter) rune {
	if b == SeparationMachineLetter {
		return SeparationToken
	}
	return a.letters[b]
}

func (a Alphabet) Letters() map[MachineLetter]rune {
	return a.letters
}

func (a Alphabet) Vals() map[rune]MachineLetter {
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
		a.vals[rn] = MachineLetter(idx)
		a.letters[MachineLetter(idx)] = rn
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

// FromSlice creates an alphabet from a serialized array. It is the
// opposite of the Serialize function, except the length is implicitly passed
// in as the length of the slice.
func FromSlice(arr []uint32) *Alphabet {
	alphabet := &Alphabet{}
	alphabet.Init()
	numRunes := uint8(len(arr))

	for i := uint8(0); i < numRunes; i++ {
		alphabet.vals[rune(arr[i])] = MachineLetter(i)
		alphabet.letters[MachineLetter(i)] = rune(arr[i])
	}
	return alphabet
}

func (a LetterSlice) Len() int           { return len(a) }
func (a LetterSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a LetterSlice) Less(i, j int) bool { return a[i] < a[j] }

/*

func (a ArcPtrSlice) Len() int           { return len(a) }
func (a ArcPtrSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ArcPtrSlice) Less(i, j int) bool { return a[i].Letter < a[j].Letter }
*/
