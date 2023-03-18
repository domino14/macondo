package tilemapping

import (
	"fmt"
	"strings"

	"github.com/domino14/macondo/config"
	"github.com/rs/zerolog/log"
)

// A "letter" or tile is internally represented by a byte.
// The 0 value is used to represent various things:
// - an empty space on the board
// - a blank on your rack
// - a "played-through" letter on the board, when used in the description of a play.
// The letter A is represented by 1, B by 2, ... all the way to 26, for the English
// alphabet, for example.
// A blank letter is the same but with the high bit set (0x80 | ml)
const (
	// MaxAlphabetSize should be below 64 so that a letterset can be a 64-bit int.
	MaxAlphabetSize = 62
	// ASCIIPlayedThrough is a somewhat user-friendly representation of a
	// played-through letter, used mostly for debug purposes.
	// Note that in order to actually be visible on a computer screen, we
	// should use `(`and `)` around letters already on a board.
	ASCIIPlayedThrough = '.'
	// BlankToken is the user-friendly representation of a blank.
	BlankToken = '?'
)

const (
	BlankMask   = 0x80
	UnblankMask = (0x80 - 1)
)

// LetterSet is a bit mask of acceptable letters, with indices from 0 to
// the maximum alphabet size.
type LetterSet uint64

// MachineLetter is a machine-only representation of a letter. It represents a signed
// integer; negative for blank letters, 0 for blanks, positive for regular letters.
type MachineLetter byte

type MachineWord []MachineLetter

// A TileMapping contains the structures needed to map a user-visible "rune",
// like the letter B, into its "MachineLetter" counterpart (for example,
// MachineLetter(2) in the english-alphabet), and vice-versa.
type TileMapping struct {
	// vals is a map of the actual physical letter rune (like 'A') to a
	// number representing it, from 0 to MaxAlphabetSize.
	vals map[string]MachineLetter
	// letters is a map of the 0 to MaxAlphabetSize value back to a letter.
	letters [MaxAlphabetSize]string
	// maxTileLength is the maximum length of any tile for this mapping,
	// in runes. For example, Catalan's L·L tile is 3 runes long.
	maxTileLength int
}

// Init initializes the alphabet data structures
func (rm *TileMapping) Init() {
	rm.vals = make(map[string]MachineLetter)
}

// Letter returns the letter that this position in the alphabet corresponds to.
func (rm *TileMapping) Letter(b MachineLetter) string {
	if b == 0 {
		return string(BlankToken)
	}
	if b.IsBlanked() {
		return strings.ToLower(rm.letters[b.Unblank()])
	}
	return rm.letters[b]
}

// Val returns the 'value' of this rune in the alphabet.
// Takes into account blanks (lowercase letters).
func (rm *TileMapping) Val(s string) (MachineLetter, error) {
	if s == string(BlankToken) {
		return 0, nil
	}
	val, ok := rm.vals[s]
	if ok {
		return val, nil
	}
	if s == strings.ToLower(s) {
		val, ok = rm.vals[strings.ToUpper(s)]
		if ok {
			return val.Blank(), nil
		}
	}
	if s == string(ASCIIPlayedThrough) {
		return 0, nil
	}
	return 0, fmt.Errorf("letter `%v` not found in alphabet", s)
}

// UserVisible turns the passed-in machine letter into a user-visible string.
func (ml MachineLetter) UserVisible(rm *TileMapping, zeroForPlayedThrough bool) string {
	if ml == 0 {
		if zeroForPlayedThrough {
			return string(ASCIIPlayedThrough)
		}
		return string(BlankToken)
	}
	return rm.Letter(ml)
}

// Blank turns the machine letter into its blank version
func (ml MachineLetter) Blank() MachineLetter {
	return ml | BlankMask
}

// Unblank turns the machine letter into its non-blank version (if it's a blanked letter)
func (ml MachineLetter) Unblank() MachineLetter {
	return ml & UnblankMask
}

// IsBlanked returns true if the machine letter is a designated blank letter.
func (ml MachineLetter) IsBlanked() bool {
	return ml&BlankMask > 0
}

// UserVisible turns the passed-in machine word into a user-visible string.
func (mw MachineWord) UserVisible(rm *TileMapping) string {
	strs := make([]string, len(mw))
	for i, l := range mw {
		strs[i] = l.UserVisible(rm, false)
	}
	return strings.Join(strs, "")
}

// UserVisiblePlayedTiles turns the passed-in machine word into a user-visible string.
// It assumes that the MachineWord represents played tiles and not just
// tiles on a rack, so it uses the PlayedThrough character for 0.
func (mw MachineWord) UserVisiblePlayedTiles(rm *TileMapping) string {
	strs := make([]string, len(mw))
	for i, l := range mw {
		strs[i] = l.UserVisible(rm, true)
	}
	return strings.Join(strs, "")
}

// Convert the MachineLetter array into a byte array. For now, wastefully
// allocate a byte array, but maybe we can use the unsafe package in the future.
func (mw MachineWord) ToByteArr() []byte {
	bts := make([]byte, len(mw))
	for i, l := range mw {
		bts[i] = byte(l)
	}
	return bts
}

func FromByteArr(bts []byte) MachineWord {
	mls := make([]MachineLetter, len(bts))
	for i, l := range bts {
		mls[i] = MachineLetter(l)
	}
	return mls
}

// NumLetters returns the number of letters in this alphabet.
func (rm *TileMapping) NumLetters() uint8 {
	return uint8(len(rm.letters))
}

func (rm *TileMapping) Vals() map[string]MachineLetter {
	return rm.vals
}

// Score returns the score of this word given the ld.
func (mw MachineWord) Score(ld *LetterDistribution) int {
	score := 0
	for _, c := range mw {
		score += ld.Score(c)
	}
	return score
}

// IntrinsicTileIdx returns the index that this tile would have in a
// rack's LetArr.
func (ml MachineLetter) IntrinsicTileIdx() MachineLetter {
	if ml.IsBlanked() || ml == 0 {
		return 0
	}
	return ml
}

// IsPlayedTile returns true if this represents a tile that was actually
// played on the board. It has to be an assigned blank or a letter, not
// a played-through-marker.
func (ml MachineLetter) IsPlayedTile() bool {
	return ml != 0
}

func (ml MachineLetter) IsVowel(ld *LetterDistribution) bool {
	ml = ml.Unblank()
	s := ld.TileMapping().Letter(ml)

	for _, v := range ld.Vowels {
		if s == string(v) {
			return true
		}
	}
	return false
}

func ToMachineWord(word string, tm *TileMapping) (MachineWord, error) {
	mls, err := ToMachineLetters(word, tm)
	if err != nil {
		return nil, err
	}
	return MachineWord(mls), nil
}

// ToMachineLetters creates an array of MachineLetters from the given string.
func ToMachineLetters(word string, rm *TileMapping) ([]MachineLetter, error) {
	letters := []MachineLetter{}
	i := 0
	var match bool
	runes := []rune(word)
	for i < len(runes) {
		match = false
		for j := i + rm.maxTileLength; j > i; j-- {
			possibleLetter := string(runes[i:j])
			if ml, ok := rm.vals[possibleLetter]; ok {
				letters = append(letters, ml)
				i = j
				match = true
				break
			} else if ml, ok := rm.vals[strings.ToUpper(possibleLetter)]; ok {
				letters = append(letters, ml|BlankMask)
				i = j
				match = true
				break
			}
		}
		if !match {
			return nil, fmt.Errorf("cannot convert %v to MachineLetters", word)
		}
	}
	return letters, nil
}

// Reconcile will take a populated alphabet, sort the glyphs, and re-index
// the numbers.
func (rm *TileMapping) Reconcile(letters []string) {
	log.Debug().Msgf("Reconciling alphabet: %v", letters)
	// These maps are now deterministic. Renumber them according to
	// sort order.
	maxLength := 0
	for idx, letter := range letters {
		rm.vals[letter] = MachineLetter(idx)
		rm.letters[MachineLetter(idx)] = letter
		if len([]rune(letter)) > maxLength {
			maxLength = len([]rune(letter))
		}
	}
	rm.maxTileLength = maxLength
	log.Debug().
		Interface("letters", rm.letters).
		Interface("vals", rm.vals).
		Int("maxTileLength", rm.maxTileLength).
		Msg("TileMapping-letters")
}

// EnglishAlphabet returns a TileMapping that corresponds to the English
// alphabet. This function should be used for testing. In production
// we will load the alphabet from the gaddag.
func EnglishAlphabet() *TileMapping {
	cfg := config.DefaultConfig()
	ld, err := GetDistribution(&cfg, "english")
	if err != nil {
		panic(err)
	}
	return ld.TileMapping()
}
