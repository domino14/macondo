// Package movegen contains all the move-generating functions. It makes
// heavy use of the GADDAG.
package movegen

import "github.com/domino14/gorilla/gaddag"

// LettersRemain returns true if there is at least one letter in the
// rack, 0 otherwise.
func LettersRemain(rack []uint8) bool {
	for i := 0; i < 27; i++ {
		if rack[i] > 0 {
			return true
		}
	}
	return false
}

// NextNode returns the next node index, given an initial node index and a
// letter. The letter can be the SeparationToken as well.
// This is a rewrite of the NextArc function in the original GADDAG paper.
// If the next node doesn't exist, it is 0!
func NextNode(gaddagData []uint32, nodeIdx uint32, letter byte) uint32 {
	// First check to see if an arc occurs for this letter.
	arcBitVector := gaddagData[nodeIdx]

	if letter == gaddag.SeparationToken {
		if ((1 << 26) & arcBitVector) == 0 {
			return 0
		}
	} else {
		if ((1 << (letter - 'A')) & arcBitVector) == 0 {
			return 0
		}
	}
	var idx, i uint8
	idx = 0
	for i = 'A'; i < letter; i++ {
		if ((1 << (i - 'A')) & arcBitVector) != 0 {
			idx++
		}
	}
	return gaddagData[idx+2]
}
