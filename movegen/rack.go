package movegen

import (
	"log"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/lexicon"
)

// Rack is a machine-friendly representation of a user's rack.
type Rack struct {
	// letArr is an array of letter codes from 0 to MaxAlphabetSize.
	// The blank can go at the MaxAlphabetSize place.
	LetArr     []int
	empty      bool
	numLetters uint8
	alphabet   *alphabet.Alphabet
	repr       string
	// letterIdxs []uint8
}

func (r *Rack) String() string {
	return r.TilesOn(int(r.alphabet.NumLetters())).UserVisible(r.alphabet)
}

// RackFromMachineLetters creates a rack from a list of machine letters.
func RackFromMachineLetters(mls []alphabet.MachineLetter, a *alphabet.Alphabet) *Rack {
	r := &Rack{}
	r.alphabet = a
	r.Set(mls)
	return r
}

// RackFromString creates a Rack from a string and an alphabet
func RackFromString(rack string, a *alphabet.Alphabet) *Rack {
	r := &Rack{}
	r.alphabet = a
	r.setFromStr(rack)
	return r
}

func (r *Rack) setFromStr(rack string) {
	if r.LetArr == nil {
		r.LetArr = make([]int, alphabet.MaxAlphabetSize+1)
	} else {
		r.clear()
	}

	for _, c := range rack {
		ml, err := r.alphabet.Val(c)
		if err == nil {
			r.LetArr[ml]++
		} else {
			log.Println("[ERROR] Rack has an illegal character: " + string(c))
		}
	}
	if len(rack) > 0 {
		r.empty = false
	}
	r.numLetters = uint8(len(rack))
}

// Set sets the rack from a list of machine letters
func (r *Rack) Set(mls []alphabet.MachineLetter) {
	if r.LetArr == nil {
		r.LetArr = make([]int, alphabet.MaxAlphabetSize+1)
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
	for i := 0; i < alphabet.MaxAlphabetSize+1; i++ {
		r.LetArr[i] = 0
	}
	r.empty = true
	r.numLetters = 0
}

func (r *Rack) take(letter alphabet.MachineLetter) {
	// this function should only be called if there is a letter on the rack
	// it doesn't check if it's there!
	r.LetArr[letter]--
	r.numLetters--
	if r.numLetters == 0 {
		r.empty = true
	}
}

func (r *Rack) add(letter alphabet.MachineLetter) {
	r.LetArr[letter]++
	r.numLetters++
	if r.empty {
		r.empty = false
	}
}

// TilesOn returns the MachineLetters of the rack's current tiles.
func (r *Rack) TilesOn(numPossibleLetters int) alphabet.MachineWord {
	if r.empty {
		return alphabet.MachineWord([]alphabet.MachineLetter{})
	}
	letters := make([]alphabet.MachineLetter, r.numLetters)
	ct := 0
	var i alphabet.MachineLetter
	for i = 0; i < alphabet.MachineLetter(numPossibleLetters); i++ {
		if r.LetArr[i] > 0 {
			for j := 0; j < r.LetArr[i]; j++ {
				letters[ct] = i
				ct++
			}
		}
	}
	if r.LetArr[alphabet.BlankMachineLetter] > 0 {
		for j := 0; j < r.LetArr[alphabet.BlankMachineLetter]; j++ {
			letters[ct] = alphabet.BlankMachineLetter
			ct++
		}
	}
	return alphabet.MachineWord(letters)
}

// ScoreOn returns the total score of the tiles on this rack.
func (r *Rack) ScoreOn(numPossibleLetters int, bag *lexicon.Bag) int {
	score := 0
	var i alphabet.MachineLetter
	for i = 0; i < alphabet.MachineLetter(numPossibleLetters); i++ {
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
