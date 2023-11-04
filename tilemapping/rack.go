package tilemapping

import (
	"github.com/rs/zerolog/log"
)

// Rack is a machine-friendly representation of a user's rack.
type Rack struct {
	// letArr is an array of letter codes from 0 to alphabet.NumLetters.
	// The blank goes at 0.
	LetArr             []int
	numLetters         uint8
	alphabet           *TileMapping
	repr               string
	numPossibleLetters uint8
	// letterIdxs []uint8
}

// NewRack creates a brand new rack structure with an alphabet.
func NewRack(alph *TileMapping) *Rack {
	return &Rack{
		alphabet:           alph,
		LetArr:             make([]int, alph.NumLetters()),
		numPossibleLetters: alph.NumLetters(),
	}
}

// String returns a user-visible version of this rack.
func (r *Rack) String() string {
	return r.TilesOn().UserVisible(r.alphabet)
}

// Copy returns a deep copy of this rack
func (r *Rack) Copy() *Rack {
	n := &Rack{
		numLetters:         r.numLetters,
		alphabet:           r.alphabet,
		repr:               r.repr,
		numPossibleLetters: r.numPossibleLetters,
	}
	n.LetArr = make([]int, len(r.LetArr))
	copy(n.LetArr, r.LetArr)
	return n
}

func (r *Rack) CopyFrom(other *Rack) {
	r.numLetters = other.numLetters
	r.alphabet = other.alphabet
	r.repr = other.repr
	r.numPossibleLetters = other.numPossibleLetters
	if r.LetArr == nil {
		r.LetArr = make([]int, len(other.LetArr))
	}
	copy(r.LetArr, other.LetArr)
}

// RackFromString creates a Rack from a string and an alphabet
func RackFromString(rack string, a *TileMapping) *Rack {
	r := &Rack{}
	r.alphabet = a
	r.setFromStr(rack)
	return r
}

func (r *Rack) setFromStr(rack string) {
	if r.LetArr == nil {
		r.LetArr = make([]int, r.alphabet.NumLetters())
	} else {
		r.Clear()
	}
	mls, err := ToMachineLetters(rack, r.Alphabet())
	if err != nil {
		log.Error().AnErr("err", err).Msg("unable to convert rack")
		return
	}

	for _, ml := range mls {
		r.LetArr[ml]++
	}

	r.numLetters = uint8(len(mls))
	r.numPossibleLetters = r.alphabet.NumLetters()
}

// Set sets the rack from a list of machine letters
func (r *Rack) Set(mls []MachineLetter) {
	r.Clear()
	for _, ml := range mls {
		r.LetArr[ml]++
	}
	r.numLetters = uint8(len(mls))
}

func (r *Rack) Clear() {
	// Clear the rack
	for i := 0; i < int(r.Alphabet().NumLetters()); i++ {
		r.LetArr[i] = 0
	}
	r.numLetters = 0
}

func (r *Rack) Take(letter MachineLetter) {
	// this function should only be called if there is a letter on the rack
	// it doesn't check if it's there!
	r.LetArr[letter]--
	r.numLetters--
}

func (r *Rack) Has(letter MachineLetter) bool {
	return r.LetArr[letter] > 0
}

func (r *Rack) CountOf(letter MachineLetter) int {
	return r.LetArr[letter]
}

func (r *Rack) Add(letter MachineLetter) {
	r.LetArr[letter]++
	r.numLetters++
}

// TilesOn returns the MachineLetters of the rack's current tiles. It is alphabetized.
func (r *Rack) TilesOn() MachineWord {
	if r.numLetters == 0 {
		return MachineWord([]MachineLetter{})
	}
	letters := make([]MachineLetter, r.numLetters)
	r.NoAllocTilesOn(letters)

	return MachineWord(letters)
}

// NoAllocTilesOn places the tiles in the passed-in slice, and returns the number
// of letters
func (r *Rack) NoAllocTilesOn(letters []MachineLetter) int {
	ct := 0
	var i MachineLetter
	for i = 0; i < MachineLetter(r.numPossibleLetters); i++ {
		if r.LetArr[i] > 0 {
			for j := 0; j < r.LetArr[i]; j++ {
				letters[ct] = i
				ct++
			}
		}
	}

	return ct
}

// ScoreOn returns the total score of the tiles on this rack.
func (r *Rack) ScoreOn(ld *LetterDistribution) int {
	score := 0
	var i MachineLetter
	for i = 0; i < MachineLetter(r.numPossibleLetters); i++ {
		if r.LetArr[i] > 0 {
			score += ld.Score(i) * r.LetArr[i]
		}
	}
	return score
}

// NumTiles returns the current number of tiles on this rack.
func (r *Rack) NumTiles() uint8 {
	return r.numLetters
}

func (r *Rack) Empty() bool {
	return r.numLetters == 0
}

func (r *Rack) Alphabet() *TileMapping {
	return r.alphabet
}
