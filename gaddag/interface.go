package gaddag

import (
	"github.com/domino14/word-golib/tilemapping"
)

type WordGraph interface {
	GetRootNodeIndex() uint32
	NextNodeIdx(nodeIdx uint32, letter tilemapping.MachineLetter) uint32
	InLetterSet(letter tilemapping.MachineLetter, nodeIdx uint32) bool
	GetAlphabet() *tilemapping.TileMapping
	GetLetterSet(nodeIdx uint32) tilemapping.LetterSet
	LexiconName() string
	// For speed expose these...
	IsEnd(nodeIdx uint32) bool
	ArcIndex(nodeIdx uint32) uint32
	Tile(nodeIdx uint32) uint8
	Accepts(nodeIdx uint32) bool
}

// GetLetterAndExtensionSets returns both the letter set (accepting letters) and
// the extension set (all possible continuations) at the given node.
// This is used for shadow-based move generation to compute leftx/rightx.
// The extension set includes all letters that have arcs from this node,
// while the letter set only includes letters that form valid words (accept).
// Following Magpie's kwg_get_letter_sets pattern.
func GetLetterAndExtensionSets(g WordGraph, nodeIdx uint32) (letterSet, extSet uint64) {
	for i := nodeIdx; ; i++ {
		t := g.Tile(i)
		// Set bit for this tile, but exclude the separation token (tile 0)
		// The expression (1 << t) ^ (1 if t==0 else 0) clears bit 0 for separation token
		var bit uint64
		if t != 0 {
			bit = uint64(1) << t
		}
		extSet |= bit
		if g.Accepts(i) {
			letterSet |= bit
		}
		if g.IsEnd(i) {
			break
		}
	}
	return
}
