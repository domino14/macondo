// Package gaddag implements the GADDAG, a pretty cool data structure
// invented by Steven Gordon.
package gaddag

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
)

// A SimpleGaddag.arr is just a slice of 32-bit elements.
// It is created by serializeElements in make_gaddag.go.
// Schema:
// [alphabetlength] [letters...] (up to 31)
// [lettersetlength] [lettersets] (binary bit masks, this is why limiting
//  to 31 letters), then
// a set of [node] [arcs...]
// Where node is a 32-bit number: LetterSetIdx + (NumArcs << NumArcsBitLoc)
// Each arc is a 32-bit number: (letter << LetterBitLoc) + index of next node,
// where letter is an index from 0 to 31 into alphabet (except for 31, which
// is the SeparationToken), and the index of the node is the index of the
// element in the SimpleGaddag array.
//
// If the node has no arcs, the arc array is empty.

type SimpleGaddag struct {
	arr      []uint32
	alphabet *Alphabet
}

type SimpleDawg SimpleGaddag

// SeparationToken is the GADDAG separation token.
const SeparationToken = '^'
const NumArcsBitLoc = 24
const LetterBitLoc = 24

// LoadGaddag loads a gaddag from a file and returns the slice of nodes.
func LoadGaddag(filename string) SimpleGaddag {
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
	g := SimpleGaddag{arr: data}
	g.SetAlphabet()
	return g
}

// Finds the index of the node pointed to by this arc and
// returns it and the letter.
func (g SimpleGaddag) ArcToIdxLetter(arcIdx uint32) (
	uint32, rune) {
	var rn rune
	letterCode := byte(g.arr[arcIdx] >> LetterBitLoc)
	if letterCode == MaxAlphabetSize {
		rn = SeparationToken
	} else {
		rn = rune(g.arr[letterCode+1])
	}
	return g.arr[arcIdx] & ((1 << LetterBitLoc) - 1), rn
}

// GetLetterSet gets the letter set of the node at nodeIdx.
func (g SimpleGaddag) GetLetterSet(nodeIdx uint32) uint32 {
	letterSetCode := g.arr[nodeIdx] & ((1 << NumArcsBitLoc) - 1)
	// Look in the letter set list for this code. We use g[0] because
	// that contains the offset in `g` where the letter sets begin.
	// (See serialization code).
	return g.arr[letterSetCode+2+g.arr[0]]
}

// InLetterSet returns whether the `letter` is in the node at `nodeIdx`'s
// letter set.
func (g SimpleGaddag) InLetterSet(letter rune, nodeIdx uint32) bool {
	letterSet := g.GetLetterSet(nodeIdx)
	idx, ok := g.alphabet.vals[letter]
	if !ok { // The ^ character, likely, when looking up single-letter words.
		return false
	}
	return letterSet&(1<<idx) != 0
}

// LetterSetAsRunes returns the letter set of the node at `nodeIdx` as
// a slice of runes.
func (g SimpleGaddag) LetterSetAsRunes(nodeIdx uint32) []rune {
	letterSet := g.GetLetterSet(nodeIdx)
	runes := []rune{}
	for idx := byte(0); idx < SeparationToken; idx++ {
		if letterSet&(1<<idx) != 0 {
			runes = append(runes, g.alphabet.letters[idx])
		}
	}
	return runes
}

// GetRootNodeIndex gets the index of the root node.
func (g SimpleGaddag) GetRootNodeIndex() uint32 {
	alphabetLength := g.arr[0]
	letterSets := g.arr[alphabetLength+1]
	return letterSets + alphabetLength + 2
}

// GetAlphabet recreates the Alphabet structure stored in this SimpleGaddag,
// and stores it in g.alphabet
func (g *SimpleGaddag) SetAlphabet() {
	alphabet := Alphabet{}
	alphabet.Init()
	// The very first element of the array is the alphabet size.
	numRunes := g.arr[0]
	for i := uint32(0); i < numRunes; i++ {
		alphabet.vals[rune(g.arr[i+1])] = i
		alphabet.letters[byte(i)] = rune(g.arr[i+1])
	}
	g.alphabet = &alphabet
}

func (g SimpleGaddag) GetAlphabet() *Alphabet {
	return g.alphabet
}

// Extracts the LetterSet and NumArcs from the node, and returns.
// func (g SimpleGaddag) ExtractNodeParams(nodeIdx uint32) (uint32, byte) {
// 	numArcs := byte(g[nodeIdx] >> NumArcsBitLoc)
// 	letterSet := g[nodeIdx] & ((1 << NumArcsBitLoc) - 1)
// 	return letterSet, numArcs
// }
