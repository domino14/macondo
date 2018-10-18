package movegen

import (
	"github.com/domino14/macondo/gaddag"
)

// BlankCharacter is the rune that represents a blank internally
const BlankCharacter = '?'
const BlankPos = gaddag.MaxAlphabetSize

type Rack struct {
	// letArr is an array of letter codes. Basically 0 position is A,
	// last position is blank (for the English lexicon)
	letArr     []uint8
	empty      bool
	numLetters uint8
	alphabet   *gaddag.Alphabet
	repr       string
}

// initialize a rack from a string
func (r *Rack) initialize(rack string, a *gaddag.Alphabet) {
	r.letArr = make([]uint8, gaddag.MaxAlphabetSize+1)
	r.alphabet = a
	for _, r := range rack {
		if r != BlankCharacter {
			idx, err := a.Val(r)
			if err == nil {
				r.letArr[idx]++
			} else {
				panic("Rack has an illegal character: " + r)
			}
		} else {
			r.letArr[gaddag.BlankPos]++
		}
	}
	if len(rack) > 0 {
		r.empty = false
	}
	r.numLetters = len(rack)
}

func (r *Rack) take(letter rune) {
	// this function should only be called if there is a letter on the rack
	// it doesn't check if it's there!
	if letter == BlankCharacter {
		r.letArr[gaddag.BlankPos]--
	} else {
		idx, _ := r.alphabet.Val(letter)
		r.letArr[idx]--
	}
	r.numLetters--
	if r.numLetters == 0 {
		r.empty = true
	}
}

func (r *Rack) add(letter rune) {
	if letter == BlankCharacter {
		r.letArr[gaddag.BlankPos]++
	} else {
		idx, _ := r.alphabet.Val(letter)
		r.letArr[idx]++
	}
	r.numLetters++
	if r.empty {
		r.empty = false
	}
}

func (r *Rack) contains(letter rune) bool {
	if r.empty {
		return false
	}
	if letter == BlankCharacter {
		return r.letArr[gaddag.BlankPos] > 0
	}
	idx, _ := r.alphabet.Val(letter)
	return r.letArr[idx] > 0
}
