package gaddag

import "github.com/domino14/macondo/alphabet"

type WordGraph interface {
	GetRootNodeIndex() uint32
	NextNodeIdx(nodeIdx uint32, letter alphabet.MachineLetter) uint32
	InLetterSet(letter alphabet.MachineLetter, nodeIdx uint32) bool
	GetAlphabet() *alphabet.Alphabet
	GetLetterSet(nodeIdx uint32) alphabet.LetterSet
	IterateSiblings(nodeIdx uint32, cb func(ml alphabet.MachineLetter, nn uint32))
}
