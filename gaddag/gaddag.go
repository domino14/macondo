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
// TODO: document schema here.
type SimpleGaddag []uint32

// SeparationToken is the GADDAG separation token.
const SeparationToken = '^'
const NumArcsBitLoc = 27
const LetterBitLoc = 24

// LoadGaddag loads a gaddag from a file and returns a pointer to its
// root node.
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
func (g SimpleGaddag) ArcToIdxLetter(arcIdx uint32) (uint32, byte) {
	letter := byte(g[arcIdx] >> LetterBitLoc)
	if letter == 26 {
		// XXX: hard-code, fix
		letter = SeparationToken
	} else {
		letter += 'A'
	}
	return g[arcIdx] & ((1 << LetterBitLoc) - 1), letter
}

// Extracts the LetterSet and NumArcs from the node, and returns.
func (g SimpleGaddag) ExtractNodeParams(nodeIdx uint32) (uint32, byte) {
	numArcs := byte(g[nodeIdx] >> NumArcsBitLoc)
	letterSet := g[nodeIdx] & ((1 << NumArcsBitLoc) - 1)
	return letterSet, numArcs
}
