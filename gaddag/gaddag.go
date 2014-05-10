// Package gaddag implements the GADDAG, a pretty cool data structure
// invented by Steven Gordon.
package gaddag

// import (
// 	"encoding/binary"
// 	"fmt"
// 	"log"
// 	"os"
// )

// SeparationToken is the GADDAG separation token.
const SeparationToken = '^'

// // A Gaddag has Elements elements in the array Data. Each element is
// // a uint32.
// // Data is laid-out as follows:
// //      NODE_ARC_BITVECTOR NODE_LETTERSET [NODE_ARCS...]
// //
// // The number of NODE_ARCS is the number of 1-bits in NODE_ARC_BITVECTOR.
// //
// // NODE_ARC_BITVECTOR is a bit vector with the lowest bit (0)
// // corresponding to A and the highest bit (26) corresponding to the
// // GADDAG SEPARATION_TOKEN.
// //
// // NODE_LETTERSET is also a bit vector with the lowest bit (0)
// // corresponding to A and the highest bit (25) corresponding to Z.
// //
// // Each of the uint32s in NODE_ARCS is the index, in Elements, of the
// // next node's NODE_ARC_BITVECTOR. They are ordered according to
// // NODE_ARC_BITVECTOR, from low bit to high bit.
// type GaddagNode struct {
// 	FirstChild   *GaddagNode
// 	NextSibling  *GaddagNode
// 	ArcBitVector uint32
// 	LetterSet    uint32
// }

// // NumArcs counts the number of 1-bits in the arc bit vector.
// func NumArcs(arcBitVector uint32) uint8 {
// 	var count uint8
// 	for arcBitVector > 0 {
// 		if (arcBitVector & 1) == 1 {
// 			count++
// 		}
// 		arcBitVector >>= 1
// 	}
// 	return count
// }

// // convertNodeData - The data array in the GADDAG has node indices
// // instead of element indices, since I made this with
// // github.com/domino14/ujamaa. This function converts all node indices
// // to element indices. At some point I will make a GADDAG maker in Go
// // and output the GADDAG in proper format.
// func convertNodeData(data []uint32, numNodes uint32) {
// 	// A map where the keys are node indices and the values are element
// 	// indices.
// 	var nodeIdx uint32
// 	var nArcs uint8
// 	nodeElMap := make(map[uint32]uint32)
// 	// consts for state machine
// 	const (
// 		_BIT_VECTOR = iota
// 		_LETTER_SET = iota
// 		_NODE_IDXS  = iota
// 	)
// 	state := _BIT_VECTOR
// 	for idx, val := range data {
// 		switch state {
// 		case _BIT_VECTOR:
// 			nodeElMap[nodeIdx] = uint32(idx)
// 			state = _LETTER_SET
// 			nArcs = NumArcs(val)
// 			if nArcs > 27 || nArcs < 0 {
// 				panic("WTF")
// 			}
// 		case _LETTER_SET:
// 			if nArcs == 0 {
// 				state = _BIT_VECTOR
// 				nodeIdx++
// 			} else {
// 				state = _NODE_IDXS
// 			}
// 		case _NODE_IDXS:
// 			nArcs--
// 			if nArcs == 0 {
// 				state = _BIT_VECTOR
// 				nodeIdx++
// 			}
// 		}
// 	}
// 	if nodeIdx != numNodes {
// 		fmt.Println(nodeIdx, numNodes)
// 		panic("Didn't count right!")
// 	}
// 	// Now that we have built up our map, let's start again and overwrite
// 	// the data array in the proper spots.
// 	state = _BIT_VECTOR
// 	for idx, val := range data {
// 		switch state {
// 		case _BIT_VECTOR:
// 			state = _LETTER_SET
// 			nArcs = NumArcs(val)
// 		case _LETTER_SET:
// 			if nArcs == 0 {
// 				state = _BIT_VECTOR
// 			} else {
// 				state = _NODE_IDXS
// 			}
// 		case _NODE_IDXS:
// 			nArcs--
// 			if nArcs == 0 {
// 				state = _BIT_VECTOR
// 			}
// 			data[idx] = nodeElMap[data[idx]]
// 		}
// 	}

// }

// // LoadGaddag loads a gaddag from a file and returns a pointer to its
// // root node.
// func LoadGaddag(filename string) []uint32 {
// 	var elements, nodes uint32
// 	var data []uint32
// 	fmt.Println("Loading", filename, "...")
// 	file, err := os.Open(filename)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	binary.Read(file, binary.LittleEndian, &elements)
// 	binary.Read(file, binary.LittleEndian, &nodes)
// 	fmt.Println("Elements, nodes", elements, nodes)
// 	data = make([]uint32, elements)
// 	binary.Read(file, binary.LittleEndian, &data)
// 	file.Close()
// 	convertNodeData(data, nodes)
// 	fmt.Println("Converted and processed GADDAG.")
// 	return data
// }

// // ContainsLetter returns a boolean indicating whether the node
// // contains the letter in its letterset.
// // gaddagData is basically the Gaddag.Data slice
// // nodeIdx is the index in Gaddag.Data of the node's NODE_ARC_BITVECTOR
// // For now letter is the ASCII representation of an uppercase letter.
// func ContainsLetter(gaddagData []uint32, nodeIdx uint32, letter byte) bool {
// 	// nodeIdx + 1 is the index of NODE_LETTERSET
// 	return (gaddagData[nodeIdx+1] & (1 << (letter - 'A'))) != 0
// }

// // ABVToString converts the arc bit vector to a string, for debugging.
// func ABVToString(bitVector uint32) string {
// 	s := ""
// 	for i := uint8(0); i < 26; i++ {
// 		if (bitVector & (1 << i)) != 0 {
// 			s += string(i + 'A')
// 		}
// 	}
// 	if (bitVector & (1 << 26)) != 0 {
// 		s += string(SeparationToken)
// 	}
// 	return s
// }
