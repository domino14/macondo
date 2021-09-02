// Package gaddag implements the GADDAG, a pretty cool data structure
// invented by Steven Gordon.
package gaddag

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddagmaker"
)

// SimpleGaddag is the result of loading the gaddag back into
// memory. Rather than contain an entire tree of linked nodes, arcs, etc
// it will be easier and faster to do bitwise operations on a 32-bit array.
// A SimpleGaddag.Nodes is just a slice of 32-bit elements.
// It is created by serializeElements in make_gaddag.go.
// File Schema:
// [4-byte magic number]
// [1-byte length (LX_LEN)]
// [utf-8 encoded bytes with lexicon name, length of bytes being LX_LEN]
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
	nodes []uint32
	// The bit-mask letter sets
	letterSets  []alphabet.LetterSet
	alphabet    *alphabet.Alphabet
	lexiconName string
}

// Ensure the magic number matches.
func compareMagicGaddag(bytes [4]uint8) bool {
	cast := string(bytes[:])
	return cast == gaddagmaker.GaddagMagicNumber
}

func GaddagToSimpleGaddag(g *gaddagmaker.Gaddag) *SimpleGaddag {
	g.SerializeElements()
	// Write to buffer
	buf := new(bytes.Buffer)
	g.Write(buf)
	// Then read from the bytes
	readBuf := bytes.NewBuffer(buf.Bytes())
	nodes, letterSets, alphabetArr, lexName := loadCommonDagStructure(readBuf)

	sg := &SimpleGaddag{nodes: nodes, letterSets: letterSets,
		alphabet:    alphabet.FromSlice(alphabetArr),
		lexiconName: string(lexName)}
	return sg
}

func ScanGaddag(data io.Reader) (*SimpleGaddag, error) {
	var magicStr [4]uint8
	binary.Read(data, binary.BigEndian, &magicStr)

	if !compareMagicGaddag(magicStr) {
		return nil, errors.New("magic number does not match gaddag")
	}

	nodes, letterSets, alphabetArr, lexName := loadCommonDagStructure(data)

	g := &SimpleGaddag{nodes: nodes, letterSets: letterSets,
		// Need to mape lexname to distribution
		alphabet:    alphabet.FromSlice(alphabetArr),
		lexiconName: string(lexName)}
	return g, nil
}

// ArcToIdxLetter finds the index of the node pointed to by this arc and
// returns it and the letter.
func (g *SimpleGaddag) ArcToIdxLetter(arcIdx uint32) (uint32, alphabet.MachineLetter) {
	letterCode := alphabet.MachineLetter(g.nodes[arcIdx] >> gaddagmaker.LetterBitLoc)
	return g.nodes[arcIdx] & gaddagmaker.NodeIdxBitMask, letterCode
}

// GetLetterSet gets the letter set of the node at nodeIdx.
func (g *SimpleGaddag) GetLetterSet(nodeIdx uint32) alphabet.LetterSet {
	letterSetCode := g.nodes[nodeIdx] & gaddagmaker.LetterSetBitMask
	return g.letterSets[letterSetCode]
}

// InLetterSet returns whether the `letter` is in the node at `nodeIdx`'s
// letter set.
func (g *SimpleGaddag) InLetterSet(letter alphabet.MachineLetter, nodeIdx uint32) bool {
	if letter == alphabet.SeparationMachineLetter {
		return false
	}
	ltc := letter
	if letter >= alphabet.BlankOffset {
		ltc = letter - alphabet.BlankOffset
	}
	letterSet := g.GetLetterSet(nodeIdx)
	return letterSet&(1<<ltc) != 0
}

// LetterSetAsRunes returns the letter set of the node at `nodeIdx` as
// a slice of runes.
func (g *SimpleGaddag) LetterSetAsRunes(nodeIdx uint32) []rune {
	letterSet := g.GetLetterSet(nodeIdx)
	runes := []rune{}
	for idx := alphabet.MachineLetter(0); idx < alphabet.MaxAlphabetSize; idx++ {
		if letterSet&(1<<idx) != 0 {
			runes = append(runes, g.alphabet.Letter(idx))
		}
	}
	return runes
}

// Note: This commented-out implementation makes the whole thing _slower_
// even though it should be O(1) lookups. Tells you something about speed
// of maps.
// func (g SimpleGaddag) NextNodeIdx(nodeIdx uint32, letter alphabet.MachineLetter) uint32 {
// 	if g.Nodes[nodeIdx].Arcs == nil {
// 		return 0
// 	}
// 	// Note: This will automatically return the zero value if the letter is not found.
// 	return g.Nodes[nodeIdx].Arcs[letter]
// }

// NextNodeIdx is analogous to NextArc in the Gordon paper. The main difference
// is that in Gordon, the initial state is an arc pointing to the first
// node. In our implementation of the GADDAG, the initial state is that
// first node. So we have to think in terms of the node that was pointed
// to, rather than the pointing arc. There is something slightly wrong with
// the paper as it does not seem possible to implement in exactly Gordon's way
// without running into issues. (See my notes in my `ujamaa` repo in gaddag.h)
func (g *SimpleGaddag) NextNodeIdx(nodeIdx uint32, letter alphabet.MachineLetter) uint32 {
	numArcs := g.NumArcs(nodeIdx)
	for i := nodeIdx + 1; i <= uint32(numArcs)+nodeIdx; i++ {
		ml := alphabet.MachineLetter(g.nodes[i] >> gaddagmaker.LetterBitLoc)
		if letter == ml {
			return g.nodes[i] & gaddagmaker.NodeIdxBitMask
		}
		// The gaddagmaker sorts SeparationToken ('^') < Norwegian 'Ø'.
		// Since SeparationMachineLetter (50) > 30, this is invalid:
		// if ml > letter {
		// 	// Since the arcs are sorted by machine letter, break if
		// 	// we hit one that is bigger than what we are looking for.
		// 	break
		// }
	}
	return 0
}

// NumArcs is simply the number of arcs for the given node.
func (g *SimpleGaddag) NumArcs(nodeIdx uint32) byte {
	// if g.Nodes[nodeIdx].Arcs == nil {
	// 	return 0
	// }
	// return byte(len(g.Nodes[nodeIdx].Arcs))
	return byte(g.nodes[nodeIdx] >> gaddagmaker.NumArcsBitLoc)
}

// GetRootNodeIndex gets the index of the root node.
func (g *SimpleGaddag) GetRootNodeIndex() uint32 {
	return 0
}

// GetAlphabet returns the alphabet for this gaddag.
func (g *SimpleGaddag) GetAlphabet() *alphabet.Alphabet {
	return g.alphabet
}

// LexiconName returns the name of the lexicon.
func (g *SimpleGaddag) LexiconName() string {
	return g.lexiconName
}

// Extracts the LetterSet and NumArcs from the node, and returns.
// func (g SimpleGaddag) ExtractNodeParams(nodeIdx uint32) (uint32, byte) {
// 	numArcs := byte(g[nodeIdx] >> NumArcsBitLoc)
// 	letterSet := g[nodeIdx] & ((1 << NumArcsBitLoc) - 1)
// 	return letterSet, numArcs
// }

func (g *SimpleGaddag) Type() GenericDawgType {
	return TypeGaddag
}

func (g *SimpleGaddag) Nodes() []uint32 {
	return g.nodes
}
