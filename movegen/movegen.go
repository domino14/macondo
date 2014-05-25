// Package movegen contains all the move-generating functions. It makes
// heavy use of the GADDAG.
package movegen

//import "github.com/domino14/macondo/gaddag"

//import "fmt"

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

// NextNodeIdx returns the index, in gaddagData, of the next node's arc
// bit vector, given an initial node index and a letter. The letter can
// be the SeparationToken as well. It finds the ARC corresponding to the
// letter/token and finds the node that it points to. This is a rewrite
// of the NextArc function in the original GADDAG paper. Returns 0 if
// the next node was not found.
// func NextNodeIdx(gaddagData []uint32, nodeIdx uint32, letter byte) uint32 {
// 	// First check to see if an arc occurs for this letter.
// 	arcBitVector := gaddagData[nodeIdx]
// 	nodeIdx += 1 // Start nodeIdx right before the very first arc
// 	if letter == gaddag.SeparationToken {
// 		if ((1 << (NumTotalLetters - 1)) & arcBitVector) == 0 {
// 			return 0
// 		}
// 		arcs := gaddag.NumArcs(arcBitVector)
// 		// Separation Token arc is guaranteed to be the very last arc.
// 		return gaddagData[nodeIdx+uint32(arcs)]
// 	} else {
// 		if ((1 << (letter - 'A')) & arcBitVector) == 0 {
// 			return 0
// 		}
// 	}
// 	var i byte
// 	for i = 'A'; i <= letter; i++ {
// 		if ((1 << (i - 'A')) & arcBitVector) != 0 {
// 			nodeIdx++
// 		}
// 	}
// 	return gaddagData[nodeIdx]
// }

// func NodeChildIdxs(gaddagData []uint32, nodeIdx uint32) (
// 	children []uint32, letters []byte) {
// 	arcBitVector := gaddagData[nodeIdx]

// }

// IMPORTANT NOTE: The Gordon GADDAG algorithm is somewhat inefficient because
// it goes through all letters on the rack. Then for every letter, it has to
// call the NextNodeIdx or similar function above, which has for loops that
// search for the next child.
// Instead, we need a data structure where the nodes have pointers to their
// "children" or "siblings" on the arcs; we then iterate through all the
// "siblings" and see if their letters are on the rack. This should be
// significantly faster if the data structure is fast.
