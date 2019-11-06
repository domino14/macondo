package alphabet

import (
	"github.com/rs/zerolog/log"
)

// Rack is a machine-friendly representation of a user's rack.
type Rack struct {
	// letArr is an array of letter codes from 0 to MaxAlphabetSize.
	// The blank can go at the MaxAlphabetSize place.
	LetArr     []int
	empty      bool
	numLetters uint8
	alphabet   *Alphabet
	repr       string
	// letterIdxs []uint8
}

// NewRack creates a brand new rack structure with an alphabet.
func NewRack(alph *Alphabet) *Rack {
	return &Rack{alphabet: alph}
}

// Hashable returns a hashable representation of this rack, that is
// not necessarily user-friendly.
func (r *Rack) Hashable() string {
	return r.TilesOn().String()
}

func (r *Rack) String() string {
	return r.TilesOn().UserVisible(r.alphabet)
}

// Copy returns a deep copy of this rack
func (r *Rack) Copy() *Rack {
	n := &Rack{
		empty:      r.empty,
		numLetters: r.numLetters,
		alphabet:   r.alphabet,
		repr:       r.repr,
	}
	n.LetArr = make([]int, len(r.LetArr))
	copy(n.LetArr, r.LetArr)
	return n
}

// RackFromString creates a Rack from a string and an alphabet
func RackFromString(rack string, a *Alphabet) *Rack {
	r := &Rack{}
	r.alphabet = a
	r.setFromStr(rack)
	return r
}

func (r *Rack) setFromStr(rack string) {
	if r.LetArr == nil {
		r.LetArr = make([]int, MaxAlphabetSize+1)
	} else {
		r.clear()
	}

	for _, c := range rack {
		ml, err := r.alphabet.Val(c)
		if err == nil {
			r.LetArr[ml]++
		} else {
			log.Error().Msgf("Rack has an illegal character: %v", string(c))
		}
	}
	if len(rack) > 0 {
		r.empty = false
	}
	r.numLetters = uint8(len(rack))
}

// Set sets the rack from a list of machine letters
func (r *Rack) Set(mls []MachineLetter) {
	if r.LetArr == nil {
		r.LetArr = make([]int, MaxAlphabetSize+1)
	} else {
		r.clear()
	}
	for _, ml := range mls {
		r.LetArr[ml]++
	}
	if len(mls) > 0 {
		r.empty = false
	}
	r.numLetters = uint8(len(mls))
}

func (r *Rack) clear() {
	// Clear the rack
	for i := 0; i < MaxAlphabetSize+1; i++ {
		r.LetArr[i] = 0
	}
	r.empty = true
	r.numLetters = 0
}

func (r *Rack) Take(letter MachineLetter) {
	// this function should only be called if there is a letter on the rack
	// it doesn't check if it's there!
	r.LetArr[letter]--
	r.numLetters--
	if r.numLetters == 0 {
		r.empty = true
	}
}

func (r *Rack) Add(letter MachineLetter) {
	r.LetArr[letter]++
	r.numLetters++
	if r.empty {
		r.empty = false
	}
}

// TilesOn returns the MachineLetters of the rack's current tiles.
func (r *Rack) TilesOn() MachineWord {
	if r.empty {
		return MachineWord([]MachineLetter{})
	}
	letters := make([]MachineLetter, r.numLetters)
	numPossibleLetters := r.alphabet.NumLetters()
	ct := 0
	var i MachineLetter
	for i = 0; i < MachineLetter(numPossibleLetters); i++ {
		if r.LetArr[i] > 0 {
			for j := 0; j < r.LetArr[i]; j++ {
				letters[ct] = i
				ct++
			}
		}
	}
	if r.LetArr[BlankMachineLetter] > 0 {
		for j := 0; j < r.LetArr[BlankMachineLetter]; j++ {
			letters[ct] = BlankMachineLetter
			ct++
		}
	}
	return MachineWord(letters)
}

// ScoreOn returns the total score of the tiles on this rack.
func (r *Rack) ScoreOn(bag *Bag) int {
	score := 0
	var i MachineLetter
	numPossibleLetters := r.alphabet.NumLetters()
	for i = 0; i < MachineLetter(numPossibleLetters); i++ {
		if r.LetArr[i] > 0 {
			score += bag.Score(i) * r.LetArr[i]
		}
	}
	return score
}

// NumTiles returns the current number of tiles on this rack.
func (r *Rack) NumTiles() uint8 {
	return r.numLetters
}

func (r *Rack) Empty() bool {
	return r.empty
}
