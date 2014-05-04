// Package movegen contains all the move-generating functions. It makes
// heavy use of the GADDAG.
package movegen

import "github.com/domino14/gorilla/gaddag"

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
func NextNodeIdx(gaddagData []uint32, nodeIdx uint32, letter byte) uint32 {
	// First check to see if an arc occurs for this letter.
	//fmt.Println(gaddagData[:100])
	arcBitVector := gaddagData[nodeIdx]
	//fmt.Println("Entered NextNodeIdx, searching for", string(letter),
	//	"node index:", nodeIdx, "arcs:", gaddag.ABVToString(arcBitVector))
	nodeIdx += 1 // Start nodeIdx right before the very first arc
	if letter == gaddag.SeparationToken {
		if ((1 << (NumTotalLetters - 1)) & arcBitVector) == 0 {
			//fmt.Println("   return 0, sep token")
			return 0
		}
		arcs := gaddag.NumArcs(arcBitVector)
		// Separation Token arc is guaranteed to be the very last arc.
		return gaddagData[nodeIdx+uint32(arcs)]
	} else {
		if ((1 << (letter - 'A')) & arcBitVector) == 0 {
			//fmt.Println("   return 0", letter, "not in arc bit vector")
			return 0
		}
	}
	var i byte
	for i = 'A'; i <= letter; i++ {
		if ((1 << (i - 'A')) & arcBitVector) != 0 {
			nodeIdx++
		}
	}
	//fmt.Println("   nodeIdx became", nodeIdx, "for letter", string(letter),
	///	" return", gaddagData[nodeIdx])
	return gaddagData[nodeIdx]
}
