// Package movegen contains all the move-generating functions. It makes
// heavy use of the GADDAG.
package movegen

import "github.com/domino14/macondo/gaddag"

//import "fmt"

// NOTE: THIS ONLY WORKS FOR ENGLISH. In order to rewrite for Spanish,
// and other lexica, we need to disassociate the concept of letters
// entirely until the very end (display time).
// All "letters" should just be numbers from 0 to NumTotalLetters - 1.
// The sorting can be by Unicode code point or whatever we choose.
const (
	NumTotalLetters = 27 // includes blank
	BlankPosition   = 26 // The blank is located at this position in a rack.
)

// LettersRemain returns true if there is at least one letter in the
// rack, 0 otherwise.
func LettersRemain(rack []uint8) bool {
	for i := 0; i < NumTotalLetters; i++ {
		if rack[i] > 0 {
			return true
		}
	}
	return false
}

type GordonGenerator struct {
	gaddag gaddag.SimpleGaddag
	board  GameBoard
}

// NextNodeIdx is analogous to NextArc in the Gordon paper. The main difference
// is that in Gordon, the initial state is an arc pointing to the first
// node. In our implementation of the GADDAG, the initial state is that
// first node. So we have to think in terms of the node that was pointed
// to, rather than the pointing arc. There is something slightly wrong with
// the paper as it does not seem possible to implement in exactly Gordon's way
// without running into issues. (See my notes in my `ujamaa` repo in gaddag.h)
// Note: This is a non-deterministic algorithm. However, using a 2-D table
// of nodes/arcs did not speed it up (actually it might have even been slower)
// This is probably due to larger memory usage being cache-inefficient.
func (gen GordonGenerator) NextNodeIdx(nodeIdx uint32, letter rune) uint32 {
	var i byte
	arcs := uint32(gen.gaddag.NumArcs(nodeIdx))
	if arcs == 0 {
		return 0
	}
	for i := nodeIdx + 1; i <= nodeIdx+arcs; i++ {
		idx, nextLetter := gen.gaddag.ArcToIdxLetter(gen.gaddag.Arr[i])
		if nextLetter == letter {
			return idx
		}
		if nextLetter > letter {
			// Since it's sorted alphabetically we know it won't be in the arc
			// list, so exit the loop early.
			// Note: This even applies to the SeparationToken ^, which may have
			// been serendipitously chosen to be lexicographically larger than
			// the A-Z alphabet.
			return 0
		}
	}
	return 0
}

// Gen is an implementation of the Gordon Gen function.
// pos is the offset from an anchor square.
func (gen *GordonGenerator) Gen(pos, word, rack, arc) {
	// If a letter L is already on this square, then GoOn...

}
func (gen *GordonGenerator) GoOn(pos, L, word, rack, NewArc, OldArc) {

}

// IMPORTANT NOTE: The Gordon GADDAG algorithm is somewhat inefficient because
// it goes through all letters on the rack. Then for every letter, it has to
// call the NextNodeIdx or similar function above, which has for loops that
// search for the next child.
// Instead, we need a data structure where the nodes have pointers to their
// "children" or "siblings" on the arcs; we then iterate through all the
// "siblings" and see if their letters are on the rack. This should be
// significantly faster if the data structure is fast.
