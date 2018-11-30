package movegen

import (
	"log"

	"github.com/domino14/macondo/alphabet"
)

// BlankCharacter is the rune that represents a blank internally
const BlankCharacter = '?'

// Rack is a machine-friendly representation of a user's rack.
type Rack struct {
	// letArr is an array of letter codes from 0 to MaxAlphabetSize.
	// The blank can go at the MaxAlphabetSize place.
	LetArr     []alphabet.MachineLetter
	empty      bool
	numLetters uint8
	alphabet   *alphabet.Alphabet
	repr       string
	// letterIdxs []uint8
}

// Initialize a rack from a string
func (r *Rack) Initialize(rack string, a *alphabet.Alphabet) {
	r.LetArr = make([]alphabet.MachineLetter, alphabet.MaxAlphabetSize+1)
	r.alphabet = a
	for _, c := range rack {
		if c != BlankCharacter {
			ml, err := a.Val(c)
			if err == nil {
				r.LetArr[ml]++
			} else {
				log.Println("[ERROR] Rack has an illegal character: " + string(c))
			}
		} else {
			r.LetArr[alphabet.BlankMachineLetter]++
		}
	}
	if len(rack) > 0 {
		r.empty = false
	}
	r.numLetters = uint8(len(rack))
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
