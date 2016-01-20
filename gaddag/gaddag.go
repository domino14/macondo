// Package gaddag implements the GADDAG, a pretty cool data structure
// invented by Steven Gordon.
package gaddag

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
)

// A SimpleGaddag is just a slice of 32-bit elements.
// It is created by serializeElements in make_gaddag.go.
// Schema:
// a set of [node] [arcs...]
// Where node is a 32-bit number: LetterSet + NumArcs << NumArcsBitLoc
// Each arc is a 32-bit number: letter << LetterBitLoc + index of next node,
// where letter is from 0 to 26, and the index of the node is the index of the
// element in the SimpleGaddag array.
// If the node has no arcs, the arc array is empty.

// XXX: This serialization format needs to be changed. We need more info:
// 	- We should store unique letter sets in a separate chunk of memory.
// 	Otherwise we only have space to store a 27-letter letter set, including
// 	the separation token(27 bits + 5 bits to represent a maximum of 32 arcs)
// 	- We should store a mapping of alphabet to code. Alphabet can contain
// 	at most 31 runes right now, so code should run from 0 - 30 (separation
// 	token would be 31 in that case)

type SimpleGaddag []uint32

// SeparationToken is the GADDAG separation token.
const SeparationToken = '^'
const NumArcsBitLoc = 24
const LetterBitLoc = 24

// LoadGaddag loads a gaddag from a file and returns the slice of nodes.
func LoadGaddag(filename string) []uint32 {
	var elements uint32
	var data []uint32
	fmt.Println("Loading", filename, "...")
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	binary.Read(file, binary.LittleEndian, &elements)
	fmt.Println("Elements", elements)
	data = make([]uint32, elements)
	binary.Read(file, binary.LittleEndian, &data)
	file.Close()
	return data
}

// Finds the index of the node pointed to by this arc and
// returns it and the letter.
func (g SimpleGaddag) ArcToIdxLetter(arcIdx uint32, alphabet *Alphabet) (
	uint32, rune) {
	var rn rune
	letterCode := byte(g[arcIdx] >> LetterBitLoc)
	if letterCode == MaxAlphabetSize {
		rn = SeparationToken
	} else {
		rn = alphabet.letters[letterCode]
	}
	return g[arcIdx] & ((1 << LetterBitLoc) - 1), rn
}

// GetLetterSet gets the letter set of the node at nodeIdx.
func (g SimpleGaddag) GetLetterSet(nodeIdx uint32) uint32 {
	letterSetCode := g[nodeIdx] & ((1 << NumArcsBitLoc) - 1)
	// Look in the letter set list for this code. We use g[0] because
	// that contains the offset in `g` where the letter sets begin.
	// (See serialization code).
	return g[letterSetCode+2+g[0]]
}

// InLetterSet returns whether the `letter` is in the node at `nodeIdx`'s
// letter set.
func (g SimpleGaddag) InLetterSet(letter rune, nodeIdx uint32,
	alphabet *Alphabet) bool {
	letterSet := g.GetLetterSet(nodeIdx)
	idx, ok := alphabet.vals[letter]
	if !ok { // The ^ character, likely, when looking up single-letter words.
		return false
	}
	return letterSet&(1<<idx) != 0
}

// GetRootNodeIndex gets the index of the root node.
func (g SimpleGaddag) GetRootNodeIndex() uint32 {
	alphabetLength := g[0]
	letterSets := g[alphabetLength+1]
	return letterSets + alphabetLength + 2
}

// GetAlphabet recreates the Alphabet structure stored in this SimpleGaddag.
func (g SimpleGaddag) GetAlphabet() *Alphabet {
	alphabet := Alphabet{}
	alphabet.Init()
	// The very first element of the array is the alphabet size.
	numRunes := g[0]
	for i := uint32(0); i < numRunes; i++ {
		alphabet.vals[rune(g[i+1])] = i
		alphabet.letters[byte(i)] = rune(g[i+1])
	}
	return &alphabet
}

// Extracts the LetterSet and NumArcs from the node, and returns.
// func (g SimpleGaddag) ExtractNodeParams(nodeIdx uint32) (uint32, byte) {
// 	numArcs := byte(g[nodeIdx] >> NumArcsBitLoc)
// 	letterSet := g[nodeIdx] & ((1 << NumArcsBitLoc) - 1)
// 	return letterSet, numArcs
// }
