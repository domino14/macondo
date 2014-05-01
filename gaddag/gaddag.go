// Package gaddag implements the GADDAG, a pretty cool data structure
// invented by Steven Gordon.
package gaddag

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
)

// A Gaddag has Elements elements in the array Data. Each element is
// a uint32.
// Data is laid-out as follows:
//      NODE_ARC_BITVECTOR NODE_LETTERSET [NODE_ARCS...]
//
// The number of NODE_ARCS is the number of 1-bits in NODE_ARC_BITVECTOR.
//
// NODE_ARC_BITVECTOR is a bit vector with the lowest bit (0) corresponding
// to A and the highest bit (26) corresponding to the GADDAG SEPARATION_TOKEN.
//
// NODE_LETTERSET is also a bit vector with the lowest bit (0) corresponding
// to A and the highest bit (25) corresponding to Z.
//
// Each of the uint32s in NODE_ARCS is the index, in Elements, of the next
// node. They are ordered according to NODE_ARC_BITVECTOR, from low bit to
// high bit.
type Gaddag struct {
	Elements uint32
	Nodes    uint32
	Data     []uint32
}

// LoadGaddag loads a gaddag from a file and returns a pointer to its
// root node.
func LoadGaddag(filename string) []uint32 {
	fmt.Println("Loading", filename, "...")
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	gaddag := Gaddag{}
	binary.Read(file, binary.LittleEndian, &gaddag.Elements)
	binary.Read(file, binary.LittleEndian, &gaddag.Nodes)
	fmt.Println("Elements, nodes", gaddag.Elements, gaddag.Nodes)
	gaddag.Data = make([]uint32, gaddag.Elements)
	binary.Read(file, binary.LittleEndian, &gaddag.Data)
	file.Close()
	return gaddag.Data
}

// func ContainsLetter
// func NextArc
