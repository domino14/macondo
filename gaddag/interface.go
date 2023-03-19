package gaddag

import (
	"github.com/domino14/macondo/tilemapping"
)

type WordGraph interface {
	GetRootNodeIndex() uint32
	NextNodeIdx(nodeIdx uint32, letter tilemapping.MachineLetter) uint32
	InLetterSet(letter tilemapping.MachineLetter, nodeIdx uint32) bool
	GetAlphabet() *tilemapping.TileMapping
	GetLetterSet(nodeIdx uint32) tilemapping.LetterSet
	IterateSiblings(nodeIdx uint32, cb func(ml tilemapping.MachineLetter, nn uint32))
	LexiconName() string
	// For speed expose these...
	IsEnd(nodeIdx uint32) bool
	ArcIndex(nodeIdx uint32) uint32
	Tile(nodeIdx uint32) uint8
	Accepts(nodeIdx uint32) bool
}
