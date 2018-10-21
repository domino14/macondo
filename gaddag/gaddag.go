// Package gaddag implements the GADDAG, a pretty cool data structure
// invented by Steven Gordon.
package gaddag

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
)

// SimpleGaddag is the result of loading the gaddag back into
// memory. Rather than contain an entire tree of linked nodes, arcs, etc
// it will be easier and faster to do bitwise operations on a 32-bit array.
// A SimpleGaddag.Arr is just a slice of 32-bit (or 64-bit) elements.
// It is created by serializeElements in make_gaddag.go.
// Schema:
// [alphabetlength] [letters...] (up to 60+?)
// [lettersetlength] [lettersets] (64-bit binary bit masks)
// a set of [node] [arcs...]
// Where node is a 32-bit number: LetterSetIdx + (NumArcs << NumArcsBitLoc)
// Each arc is a 32-bit number: (letter << LetterBitLoc) + index of next node,
// where letter is an index from 0 to MaxAlphabetSize into alphabet (except for
// MaxAlphabetSize, which is the SeparationToken), and the index of the node is
// the index of the element in the SimpleGaddag array.
//
// If the node has no arcs, the arc array is empty.
type SimpleGaddag struct {
	// Nodes is just a slice of 32-bit elements, the node array.
	Nodes []uint32
	// The bit-mask letter sets
	LetterSets []LetterSet
	alphabet   *Alphabet
	numLetters uint32
}

type SimpleDawg SimpleGaddag

// LoadGaddag loads a gaddag from a file and returns the slice of nodes.
func LoadGaddag(filename string) SimpleGaddag {
	var elements uint32
	var data []uint32
	fmt.Println("Loading", filename, "...")
	file, err := os.Open(filename)
	if err != nil {
		log.Println("[ERROR] Could not open gaddag", err)
		return SimpleGaddag{}
	}
	file.Read

	binary.Read(file, binary.LittleEndian, &elements)
	fmt.Println("Elements", elements)
	data = make([]uint32, elements)
	binary.Read(file, binary.LittleEndian, &data)
	file.Close()
	g := SimpleGaddag{Arr: data}
	g.SetAlphabet()
	return g
}

// ArcToIdxLetter finds the index of the node pointed to by this arc and
// returns it and the letter.
func (g SimpleGaddag) ArcToIdxLetter(arcIdx uint32) (uint32, MachineLetter) {
	var rn rune
	letterCode := byte(g.Arr[arcIdx] >> LetterBitLoc)
	if letterCode == MaxAlphabetSize {
		rn = SeparationToken
	} else {
		rn = rune(g.Arr[letterCode+1])
	}
	return g.Arr[arcIdx] & NodeIdxBitMask, rn
}

// GetLetterSet gets the letter set of the node at nodeIdx.
func (g SimpleGaddag) GetLetterSet(nodeIdx uint32) uint32 {
	letterSetCode := g.Arr[nodeIdx] & LetterSetBitMask
	// Look in the letter set list for this code. We use g[0] because
	// that contains the offset in `g` where the letter sets begin.
	// (See serialization code).
	// Note: for some reason g.numLetters seems to be a little slower
	// than g.Arr[0].
	return g.Arr[letterSetCode+2+g.Arr[0]]
}

// InLetterSet returns whether the `letter` is in the node at `nodeIdx`'s
// letter set.
func (g SimpleGaddag) InLetterSet(letter MachineLetter, nodeIdx uint32) bool {
	if letter == MaxAlphabetSize {
		return false
	}
	letterSet := g.GetLetterSet(nodeIdx)
	return letterSet&(1<<letter) != 0
}

// LetterSetAsRunes returns the letter set of the node at `nodeIdx` as
// a slice of runes.
// func (g SimpleGaddag) LetterSetAsRunes(nodeIdx uint32) []rune {
// 	letterSet := g.GetLetterSet(nodeIdx)
// 	runes := []rune{}
// 	for idx := byte(0); idx < SeparationToken; idx++ {
// 		if letterSet&(1<<idx) != 0 {
// 			runes = append(runes, g.alphabet.letters[idx])
// 		}
// 	}
// 	return runes
// }

func (g SimpleGaddag) NumArcs(nodeIdx uint32) byte {
	return byte(g.Arr[nodeIdx] >> NumArcsBitLoc)
}

// GetRootNodeIndex gets the index of the root node.
func (g SimpleGaddag) GetRootNodeIndex() uint32 {
	return 0
}

// SetAlphabet recreates the Alphabet structure stored in this SimpleGaddag,
// and stores it in g.alphabet
func (g *SimpleGaddag) SetAlphabet() {
	alphabet := Alphabet{}
	alphabet.Init()
	// The very first element of the array is the alphabet size.
	numRunes := g.Arr[0]
	athruz := false
	runeCt := uint32(65)
	for i := uint32(0); i < numRunes; i++ {
		alphabet.vals[rune(g.Arr[i+1])] = i
		alphabet.letters[byte(i)] = rune(g.Arr[i+1])
		if runeCt == g.Arr[i+1] {
			runeCt++
		}
	}
	if runeCt == uint32(91) {
		athruz = true
	}
	alphabet.athruz = athruz
	g.alphabet = &alphabet
	g.numLetters = numRunes
	log.Printf("Alphabet athruz is %v", alphabet.athruz)
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
