// Package movegen contains all the move-generating functions. It makes
// heavy use of the GADDAG.
// Implementation notes:
// Using Gordon's GADDAG algorithm with some minor speed changes. Similar to
// the A&J DAWG algorithm, we should not be doing "for each letter allowed
// on this square", as that is a for loop through every letter in the rack.
// Instead, we should go through the node siblings in the Gen algorithm,
// and check their presence in the cross-sets.
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
	gaddag   gaddag.SimpleGaddag
	board    GameBoard
	vertical bool // Are we generating moves vertically or not?
	// The move generator works by generating moves starting at an anchor
	// square. curAnchorRow and curAnchorCol are the 0-based coordinates
	// of the current anchor square.
	curAnchorRow uint8
	curAnchorCol uint8
}

func (gen *GordonGenerator) GenAll(rack []string, board GameBoard) {
	gen.board = board
	gen.curAnchorRow = 7
	gen.curAnchorCol = 7
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

// XXX REWRITE THIS ASAP; THIS WONT WORK FOR ANYTHING OTHER THAN ENGLISH
func crossAllowed(cross uint32, letter rune) bool {
	idx := letter - 'A'
	return cross&(1<<uint32(idx)) != 0
}

// Gen is an implementation of the Gordon Gen function.
// pos is the offset from an anchor square.
func (gen *GordonGenerator) Gen(pos uint8, word string, rack *Rack,
	nodeIdx uint32) {

	curRow := gen.curAnchorRow
	curCol := gen.curAnchorCol

	var crossSet uint32

	if gen.vertical {
		curRow += pos
	} else {
		curCol += pos
	}

	// If a letter L is already on this square, then GoOn...
	curSquare := gen.board[curRow][curCol]
	curLetter := curSquare.letter

	if gen.vertical {
		crossSet = curSquare.hcrossSet
	} else {
		crossSet = curSquare.vcrossSet
	}

	if curLetter != ' ' {
		nnIdx := NextNodeIdx(nodeIdx, curLetter)
		if nnIdx != 0 {
			gen.GoOn(pos, curLetter, word, nnIdx)
		}
	} else if !rack.empty {
		// Instead of doing the loop in the Gordon Gen algorithm, we should
		// just go through the node's children and test them independently
		// against the cross set. Note that some of these children could be
		// the SeparationToken
		arcs := uint32(gen.gaddag.NumArcs(nodeIdx))
		for i := nodeIdx + 1; i <= nodeIdx+arcs; i++ {
			nnIdx, nextLetter := gen.gaddag.ArcToIdxLetter(gen.gaddag.Arr[i])
			if nextLetter == gaddag.SeparationToken {
				break
			}
			// The letter must be on the rack AND it must be allowed in the
			// cross-set.
			if !(rack.contains(nextLetter) && crossAllowed(crossSet, nextLetter)) {
				continue
			}
			rack.take(nextLetter)
			gen.GoOn(pos, nextLetter, word, nnIdx)
			rack.add(nextLetter)
		}
		// Check for the blanks meow.
		if rack.contains(BlankCharacter) {
			// Just go through all the children; they're all acceptable if they're
			// in the cross-set.
			for i := nodeIdx + 1; i <= nodeIdx+arcs; i++ {
				nnIdx, nextLetter := gen.gaddag.ArcToIdxLetter(gen.gaddag.Arr[i])
				if nextLetter == gaddag.SeparationToken {
					break
				}
				if !crossAllowed(crossSet, nextLetter) {
					continue
				}
				rack.take(BlankCharacter)
				gen.GoOn(pos, L)

			}
		}

		gen.GoOn()
	}

}

// func (gen *GordonGenerator) GoOn(pos, L, word, rack, NewArc, OldArc) {

// }

// For future?: The Gordon GADDAG algorithm is somewhat inefficient because
// it goes through all letters on the rack. Then for every letter, it has to
// call the NextNodeIdx or similar function above, which has for loops that
// search for the next child.
// Instead, we need a data structure where the nodes have pointers to their
// "children" or "siblings" on the arcs; we then iterate through all the
// "siblings" and see if their letters are on the rack. This should be
// significantly faster if the data structure is fast.
