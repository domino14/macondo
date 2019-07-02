package alphabet

import (
	"fmt"
	"sort"
	"unicode"

	"github.com/rs/zerolog/log"
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
	// BlankMachineLetter is the MachineLetter corresponding to the
	// blank. It is also set at the max alphabet size. Based on the context
	// in which it is used, it should not be confused with the
	// SeparationMachineLetter above.
	BlankMachineLetter = MaxAlphabetSize
	// BlankOffset is the offset at which letters with a code >= offset
	// represent blanks.
	BlankOffset = 100
	// SeparationToken is the GADDAG separation token.
	SeparationToken = '^'
	// EmptySquareMarker is a MachineLetter representation of an empty square
	EmptySquareMarker = MaxAlphabetSize + 1
	// PlayedThroughMarker is a MachineLetter representation of a filled-in square
	// that was played through.
	PlayedThroughMarker = MaxAlphabetSize + 2
	// ASCIIPlayedThrough is a somewhat user-friendly representation of a
	// played-through letter, used mostly for debug purposes.
	// Note that in order to actually be visible on a computer screen, we
	// should use `(`and `)` around letters already on a board.
	ASCIIPlayedThrough = '.'
	// BlankToken is the user-friendly representation of a blank.
	BlankToken = '?'
)

// LetterSet is a bit mask of acceptable letters, with indices from 0 to
// the maximum alphabet size.
type LetterSet uint64

// LetterSlice is a slice of runes. We make it a separate type for ease in
// defining sort functions on it.
type LetterSlice []rune

// MachineLetter is a machine-only representation of a letter. It goes from
// 0 to the maximum alphabet size.
type MachineLetter byte

// Blank blankifies a letter; i.e. it adds the BlankOffset to it.
// It returns a new MachineLetter; it does not modify the one that is passed
// in!
func (ml MachineLetter) Blank() MachineLetter {
	if ml < BlankOffset {
		return ml + BlankOffset
	}
	return ml
}

// IsBlanked returns true if this is the blank version of a letter.
func (ml MachineLetter) IsBlanked() bool {
	return ml >= BlankOffset
}

// Unblank is the opposite of the above function; it removes the blank offset
// from a letter.
func (ml MachineLetter) Unblank() MachineLetter {
	if ml >= BlankOffset {
		return ml - BlankOffset
	}
	return ml
}

// UserVisible turns the passed-in machine letter into a user-visible rune.
func (ml MachineLetter) UserVisible(alph *Alphabet) rune {
	if ml >= BlankOffset {
		return unicode.ToLower(alph.Letter(ml - BlankOffset))
	} else if ml == PlayedThroughMarker {
		return ASCIIPlayedThrough
	} else if ml == BlankMachineLetter {
		return '?'
	} else if ml == EmptySquareMarker {
		return ' '
	}
	return alph.Letter(ml)
}

// IsVowel returns true for vowels. Note that this needs an alphabet.
func (ml MachineLetter) IsVowel(alph *Alphabet) bool {
	ml = ml.Unblank()
	rn := alph.Letter(ml)
	switch rn {
	case 'A', 'E', 'I', 'O', 'U':
		return true
	default:
		return false
	}
}

// MachineWord is an array of MachineLetters
type MachineWord []MachineLetter

// UserVisible turns the passed-in machine word into a user-visible string.
func (mw MachineWord) UserVisible(alph *Alphabet) string {
	runes := make([]rune, len(mw))
	for i, l := range mw {
		runes[i] = l.UserVisible(alph)
	}
	return string(runes)
}

// String() returns a non-printable string version of this machineword. This
// is useful for hashing purposes.
func (mw MachineWord) String() string {
	bytes := make([]byte, len(mw))
	for i, l := range mw {
		bytes[i] = byte(l)
	}
	return string(bytes)
}

// Score returns the score of this word given the bag.
func (mw MachineWord) Score(bag *Bag) int {
	score := 0
	for _, c := range mw {
		score += bag.Score(c)
	}
	return score
}

// ToMachineWord creates a MachineWord from the given string.
func ToMachineWord(word string, alph *Alphabet) (MachineWord, error) {
	mls, err := ToMachineLetters(word, alph)
	if err != nil {
		return nil, err
	}
	return MachineWord(mls), nil
}

// ToMachineLetters creates an array of MachineLetters from the given string.
func ToMachineLetters(word string, alph *Alphabet) ([]MachineLetter, error) {
	letters := make([]MachineLetter, len(word))
	for idx, ch := range word {
		ml, err := alph.Val(ch)
		if err != nil {
			return nil, err
		}
		letters[idx] = ml
	}
	return letters, nil
}

// ToMachineOnlyString creates a non-printable string from the given word.
// This is used to make it hashable for map usage.
func ToMachineOnlyString(word string, alph *Alphabet) (string, error) {
	letters := make([]byte, len(word))
	for idx, ch := range word {
		ml, err := alph.Val(ch)
		if err != nil {
			return "", err
		}
		letters[idx] = byte(ml)
	}
	return string(letters), nil
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
	if r == BlankToken {
		return BlankMachineLetter, nil
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
	if r == ASCIIPlayedThrough {
		return PlayedThroughMarker, nil
	}
	return 0, fmt.Errorf("Letter `%c` not found in alphabet", r)
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
	log.Debug().Msgf("Serializing %v", els)
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
