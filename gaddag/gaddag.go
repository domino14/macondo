// Package gaddag implements the GADDAG, a pretty cool data structure
// invented by Steven Gordon.
package gaddag

import (
	"encoding/binary"
	"log"
	"os"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddagmaker"
)

// SimpleNode is what makes up the nodes in the SimpleGaddag. Note: these
// don't correspond to the "states" in the Gordon paper. Instead, states and
// arcs are combined into one structure. We create a map of arcs in order
// to speed up access.
type SimpleNode struct {
	// Val is the actual 32-bit value stored in the node.
	// We don't need to break it out here.
	Val uint32
	// Arcs contains a map for nodes that are "states". For nodes that are arcs,
	// or for states that have no arcs coming out of them, this map would be nil.
	// The uint32 is simply the index of the destination node in the slice of
	// SimpleNodes.
	Arcs map[alphabet.MachineLetter]uint32
}

// SimpleGaddag is the result of loading the gaddag back into
// memory. Rather than contain an entire tree of linked nodes, arcs, etc
// it will be easier and faster to do bitwise operations on a 32-bit array.
// A SimpleGaddag.Nodes is just a slice of 32-bit elements.
// It is created by serializeElements in make_gaddag.go.
// File Schema:
// [4-byte magic number]
// [alphabetlength] [letters...] (up to 60+?)
// [lettersetlength] [lettersets] (64-bit binary bit masks)
// a set of [node] [arcs...]
// Where node is a 32-bit number: LetterSetIdx + (NumArcs << NumArcsBitLoc)
// Each arc is a 32-bit number: (letter << LetterBitLoc) + index of next node,
// where letter is an index from 0 to MaxAlphabetSize into alphabet (except for
// MaxAlphabetSize, which is the SeparationToken), and the index of the node is
// the index of the element in the SimpleGaddag.Nodes array.
//
// If the node has no arcs, the arc array is empty.
type SimpleGaddag struct {
	// Nodes is just a slice of 32-bit elements, the node array.
	Nodes []SimpleNode
	// The bit-mask letter sets
	LetterSets []alphabet.LetterSet
	alphabet   *alphabet.Alphabet
}

// A SimpleDawg is just a SimpleGaddag with fewer paths.
type SimpleDawg SimpleGaddag

// Ensure the magic number matches.
func compareMagic(bytes [4]uint8) bool {
	cast := string(bytes[:])
	return cast == gaddagmaker.GaddagMagicNumber || cast == gaddagmaker.DawgMagicNumber
}

// LoadGaddag loads a gaddag from a file and returns the slice of nodes.
func LoadGaddag(filename string) SimpleGaddag {
	log.Println("Loading", filename, "...")
	file, err := os.Open(filename)
	if err != nil {
		log.Println("[ERROR] Could not open gaddag", err)
		return SimpleGaddag{}
	}
	var magicStr [4]uint8
	binary.Read(file, binary.BigEndian, &magicStr)

	if !compareMagic(magicStr) {
		log.Println("[ERROR] Magic number does not match")
		return SimpleGaddag{}
	}

	var alphabetSize, lettersetSize, nodeSize uint32

	binary.Read(file, binary.BigEndian, &alphabetSize)
	log.Println("[DEBUG] Alphabet size: ", alphabetSize)
	alphabetArr := make([]uint32, alphabetSize)
	binary.Read(file, binary.BigEndian, &alphabetArr)

	binary.Read(file, binary.BigEndian, &lettersetSize)
	log.Println("[DEBUG] LetterSet size: ", lettersetSize)
	letterSets := make([]alphabet.LetterSet, lettersetSize)
	binary.Read(file, binary.BigEndian, letterSets)

	binary.Read(file, binary.BigEndian, &nodeSize)
	log.Println("[DEBUG] Nodes size: ", nodeSize)
	nodes := make([]uint32, nodeSize)
	binary.Read(file, binary.BigEndian, &nodes)
	file.Close()

	g := SimpleGaddag{Nodes: toNodeSlice(nodes), LetterSets: letterSets,
		alphabet: alphabet.FromSlice(alphabetArr)}
	return g
}

// Turn the slice of ints into a nodes and arcs connected structure
func toNodeSlice(nodes []uint32) []SimpleNode {
	simpleNodes := make([]SimpleNode, len(nodes))
	isState := true
	var curNodeIdx uint32
	curArcsToGo := uint32(0)
	for idx := range simpleNodes {
		simpleNodes[idx].Val = nodes[idx]
		if isState {
			// States have a map of arcs (possibly nil)
			curArcsToGo = nodes[idx] >> gaddagmaker.NumArcsBitLoc
			if curArcsToGo > 0 {
				simpleNodes[idx].Arcs = make(map[alphabet.MachineLetter]uint32)
				curNodeIdx = uint32(idx)
				isState = false
			}
		} else {
			// This is an arc. Modify the "current node"'s arc map.
			dest, ml := nodes[idx]&gaddagmaker.NodeIdxBitMask,
				alphabet.MachineLetter(nodes[idx]>>gaddagmaker.LetterBitLoc)
			simpleNodes[curNodeIdx].Arcs[ml] = dest
			curArcsToGo--
			if curArcsToGo == 0 {
				isState = true
			}
		}
	}
	return simpleNodes
}

// ArcToIdxLetter finds the index of the node pointed to by this arc and
// returns it and the letter.
func (g SimpleGaddag) ArcToIdxLetter(arcIdx uint32) (uint32, alphabet.MachineLetter) {
	// log.Printf("[DEBUG] ArcToIdxLetter called with %v", arcIdx)
	letterCode := alphabet.MachineLetter(g.Nodes[arcIdx].Val >> gaddagmaker.LetterBitLoc)
	return g.Nodes[arcIdx].Val & gaddagmaker.NodeIdxBitMask, letterCode
}

// GetLetterSet gets the letter set of the node at nodeIdx.
func (g SimpleGaddag) GetLetterSet(nodeIdx uint32) alphabet.LetterSet {
	letterSetCode := g.Nodes[nodeIdx].Val & gaddagmaker.LetterSetBitMask
	return g.LetterSets[letterSetCode]
}

// InLetterSet returns whether the `letter` is in the node at `nodeIdx`'s
// letter set.
func (g SimpleGaddag) InLetterSet(letter alphabet.MachineLetter, nodeIdx uint32) bool {
	if letter == alphabet.SeparationMachineLetter {
		return false
	}
	letterSet := g.GetLetterSet(nodeIdx)
	return letterSet&(1<<letter) != 0
}

// LetterSetAsRunes returns the letter set of the node at `nodeIdx` as
// a slice of runes.
func (g SimpleGaddag) LetterSetAsRunes(nodeIdx uint32) []rune {
	letterSet := g.GetLetterSet(nodeIdx)
	runes := []rune{}
	for idx := alphabet.MachineLetter(0); idx < alphabet.MaxAlphabetSize; idx++ {
		if letterSet&(1<<idx) != 0 {
			runes = append(runes, g.alphabet.Letter(idx))
		}
	}
	return runes
}

// NumArcs is simply the number of arcs for the given node.
func (g SimpleGaddag) NumArcs(nodeIdx uint32) byte {
	if g.Nodes[nodeIdx].Arcs == nil {
		return 0
	}
	return byte(len(g.Nodes[nodeIdx].Arcs))
}

// GetRootNodeIndex gets the index of the root node.
func (g SimpleGaddag) GetRootNodeIndex() uint32 {
	return 0
}

func (g SimpleGaddag) GetAlphabet() *alphabet.Alphabet {
	return g.alphabet
}

// Extracts the LetterSet and NumArcs from the node, and returns.
// func (g SimpleGaddag) ExtractNodeParams(nodeIdx uint32) (uint32, byte) {
// 	numArcs := byte(g[nodeIdx] >> NumArcsBitLoc)
// 	letterSet := g[nodeIdx] & ((1 << NumArcsBitLoc) - 1)
// 	return letterSet, numArcs
// }
