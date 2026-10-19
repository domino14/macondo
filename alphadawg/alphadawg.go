// Package alphadawg implements the query primitives for an "alpha DAWG" --
// a word graph whose entries are the alphagrams of a lexicon rather than its
// words. CARE, RACE and ACRE all collapse to the single entry ACER, so a
// multiset of tiles is a legal WordSmog word exactly when its sorted form is
// in the graph.
//
// The on-disk format is a .kad file ("Kurnia Alpha Dawg"), which is byte-wise
// a standard DawgOnly KWG: one little-endian uint32 per node, no header. The
// only structural difference from a .kwg is that node 0's arc points at the
// DAWG root and node 1 is an ordinary node (a .kwg has a GADDAG root at node
// 1), so kwg.KWG.GetRootNodeIndex must NOT be used here -- see rootNodeIndex.
//
// Because every stored entry is sorted and sibling lists are in strictly
// ascending tile order, all of these operations walk a tally (a per-letter
// histogram) through the graph in ascending letter order. No permutation
// search is ever needed.
//
// These functions were originally developed in wolges and ported here via
// magpie's src/ent/kwg_alpha.h. See
// https://github.com/andy-k/wolges/blob/main/details.txt
package alphadawg

import (
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

// Tally is a per-letter histogram of tiles. Index 0 counts undesignated
// blanks; index i counts copies of MachineLetter(i). Callers keep these on
// the stack, so every function here takes a pointer and none of them retain
// it.
type Tally [tilemapping.MaxAlphabetSize + 1]uint8

// AddWord tallies the letters of word, unblanking designated blanks. Blanks
// on a board or in a formed word are already designated, so they count as the
// letter they represent.
func (t *Tally) AddWord(word tilemapping.MachineWord) {
	for _, ml := range word {
		t[ml.Unblank()]++
	}
}

// rootNodeIndex returns the index of the DAWG root's child list. For a
// DawgOnly graph that is node 0's arc index.
func rootNodeIndex(k *kwg.KWG) uint32 {
	return k.ArcIndex(0)
}

// seek descends into nodeIdx's child list and returns the index of the child
// holding tile, or 0 if there is none. Note that it dereferences the node it
// is given, which is why callers can start at node index 0 (the root pointer
// node) without special-casing it.
func seek(nodes []uint32, nodeIdx uint32, tile uint8) uint32 {
	nodeIdx = nodes[nodeIdx] & kwg.KWGNodeArcMask
	if nodeIdx == 0 {
		return 0
	}
	for {
		node := nodes[nodeIdx]
		if uint8(node>>kwg.KWGNodeTileShift) == tile {
			return nodeIdx
		}
		if node&kwg.KWGNodeIsEndBit != 0 {
			return 0
		}
		nodeIdx++
	}
}

// completes reports whether descending through nodeIdx and then placing every
// remaining tile in t from nextLetter upwards lands on an accepting node.
func completes(nodes []uint32, nodeIdx uint32, t *Tally, distSize int, nextLetter int) bool {
	for letter := nextLetter; letter < distSize; letter++ {
		for k := uint8(0); k < t[letter]; k++ {
			nodeIdx = seek(nodes, nodeIdx, uint8(letter))
			if nodeIdx == 0 {
				return false
			}
		}
	}
	return nodes[nodeIdx]&kwg.KWGNodeAcceptsBit != 0
}

// Accepts reports whether the tallied tiles form a valid alphagram. Blanks
// (index 0) are ignored, so this is the right call when every tile is already
// designated -- board letters and formed words. Use AcceptsWithBlanks when
// the tally may still hold undesignated blanks.
func Accepts(k *kwg.KWG, t *Tally, distSize int) bool {
	// Start at node 0, the root pointer node; seek dereferences it. Start at
	// letter 1 to skip the blank slot.
	return completes(k.Nodes(), 0, t, distSize, 1)
}

// AcceptsWithBlanks reports whether the tallied tiles form a valid alphagram
// under some designation of the undesignated blanks in t[0].
func AcceptsWithBlanks(k *kwg.KWG, t *Tally, distSize int) bool {
	if t[0] == 0 {
		return Accepts(k, t, distSize)
	}
	var designated Tally = *t
	designated[0]--
	for letter := 1; letter < distSize; letter++ {
		designated[letter]++
		if AcceptsWithBlanks(k, &designated, distSize) {
			return true
		}
		designated[letter]--
	}
	return false
}

// ComputeCrossSet returns the bitset of letters X for which the tallied tiles
// plus one X form a valid alphagram. Bit i is set for MachineLetter(i); bit 0
// is never set (macondo handles blanks separately in the move generator, and
// its classic cross sets don't use bit 0 either).
//
// This is a single descent: walking the tally in ascending letter order, every
// sibling whose tile sorts before the current tally letter is a candidate
// insertion point, and after the whole tally is consumed the remaining child
// list holds the candidates that sort after everything.
func ComputeCrossSet(k *kwg.KWG, t *Tally, distSize int) uint64 {
	var crossSet uint64
	nodes := k.Nodes()
	nodeIdx := rootNodeIndex(k)
	if nodeIdx == 0 {
		return 0
	}
	for letter := 1; letter < distSize; letter++ {
		for n := uint8(0); n < t[letter]; n++ {
			for {
				node := nodes[nodeIdx]
				tile := int(node >> kwg.KWGNodeTileShift)
				if tile > letter {
					// Sibling lists are sorted, so nothing further can match.
					return crossSet
				}
				if tile < letter {
					// Candidate insertion before this tally letter.
					if completes(nodes, nodeIdx, t, distSize, letter) {
						crossSet |= 1 << uint(tile)
					}
					if node&kwg.KWGNodeIsEndBit != 0 {
						return crossSet
					}
					nodeIdx++
					continue
				}
				next := node & kwg.KWGNodeArcMask
				if next == 0 {
					return crossSet
				}
				nodeIdx = next
				break
			}
		}
	}
	// Candidates that sort after every tallied letter.
	for {
		node := nodes[nodeIdx]
		if completes(nodes, nodeIdx, t, distSize, distSize) {
			crossSet |= 1 << uint(node>>kwg.KWGNodeTileShift)
		}
		if node&kwg.KWGNodeIsEndBit != 0 {
			return crossSet
		}
		nodeIdx++
	}
}
