package movegen

import (
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
)

// leaveMap enables O(1) leave value lookup during move generation.
// Each tile on the rack gets a unique bit in a bitmask. As tiles are
// played/unplayed during recursiveGen, the bitmask is updated with a
// single AND/OR. The leave value is just values[currentIndex].
//
// Populated during exchange generation using incremental KWG traversal
// (no per-subset sort + KWG walk). Ported from magpie's leave_map.h.
type leaveMap struct {
	values          [1 << game.RackTileLimit]float64
	letterBaseIndex [tilemapping.MaxAlphabetSize + 1]int
	currentIndex    int
	totalTiles      int
	initialized     bool
}

// init assigns bit positions to rack tiles and sets the initial index
// to all-bits-set (full rack = empty leave).
func (lm *leaveMap) init(rack *tilemapping.Rack) {
	lm.initialized = false
	currentBase := 0
	for ml := tilemapping.MachineLetter(0); int(ml) < len(rack.LetArr); ml++ {
		count := rack.LetArr[ml]
		if count > 0 {
			lm.letterBaseIndex[ml] = currentBase
			currentBase += count
		}
	}
	lm.totalTiles = currentBase
	if lm.totalTiles == 0 || lm.totalTiles > game.RackTileLimit {
		return
	}
	lm.currentIndex = (1 << lm.totalTiles) - 1 // all bits set = full rack
	lm.initialized = true
}

// takeLetter updates the index when a tile is removed from the rack.
// numberOnRack is the count AFTER removal.
func (lm *leaveMap) takeLetter(ml tilemapping.MachineLetter, numberOnRack int) {
	base := lm.letterBaseIndex[ml]
	lm.currentIndex &= ^(1 << (base + numberOnRack))
}

// addLetter updates the index when a tile is added back to the rack.
// numberOnRack is the count BEFORE addition (i.e., current count on rack).
func (lm *leaveMap) addLetter(ml tilemapping.MachineLetter, numberOnRack int) {
	base := lm.letterBaseIndex[ml]
	lm.currentIndex |= 1 << (base + numberOnRack)
}

// currentValue returns the leave value for the current rack state.
func (lm *leaveMap) currentValue() float64 {
	return lm.values[lm.currentIndex]
}

const unfoundIndex int32 = -1

// incrementNodeToML scans siblings at nodeIndex to find tile ml,
// updating wordIndex to account for skipped subtrees.
func incrementNodeToML(kwg *kwg.KWG, nodeIndex uint32, wordIndex int32, ml tilemapping.MachineLetter) (uint32, int32) {
	if nodeIndex == 0 {
		return 0, unfoundIndex
	}
	wIdx := wordIndex
	for {
		if kwg.Tile(nodeIndex) == uint8(ml) {
			return nodeIndex, wIdx
		}
		if kwg.IsEnd(nodeIndex) {
			return 0, unfoundIndex
		}
		wIdx += kwg.WordCountAt(nodeIndex) - kwg.WordCountAt(nodeIndex+1)
		nodeIndex++
	}
}

// followArc follows the arc from nodeIndex to its children,
// incrementing wordIndex by 1 if the current node accepts.
func followArc(kwg *kwg.KWG, nodeIndex uint32, wordIndex int32) (uint32, int32) {
	if nodeIndex == 0 {
		return 0, unfoundIndex
	}
	nextWord := wordIndex
	if kwg.Accepts(nodeIndex) {
		nextWord++
	}
	return kwg.ArcIndex(nodeIndex), nextWord
}

// populateLeaveMap enumerates all rack subsets using incremental KWG
// traversal and stores leave values. Also records best leave per size.
func (gen *GordonGenerator) populateLeaveMap(rack *tilemapping.Rack) {
	klv := gen.klv
	if klv == nil {
		return
	}
	leaveKWG := klv.KWG()
	rootIdx := leaveKWG.ArcIndex(0)
	gen.populateLeaveMapIncremental(rack, leaveKWG, klv, rootIdx, 0, 0)
}

// populateLeaveMapIncremental walks the leave KWG incrementally as tiles
// are added to the leave, avoiding per-subset sort + full KWG walk.
// Matches magpie's generate_exchange_moves with incremental KLV indices.
func (gen *GordonGenerator) populateLeaveMapIncremental(
	rack *tilemapping.Rack,
	leaveKWG *kwg.KWG,
	klv *equity.KLV,
	nodeIndex uint32,
	wordIndex int32,
	ml tilemapping.MachineLetter,
) {
	for int(ml) < len(rack.LetArr) && rack.LetArr[ml] == 0 {
		ml++
	}
	if int(ml) == len(rack.LetArr) {
		// Leaf: store leave value using the tracked word index
		numOnRack := int(rack.NumTiles())
		var val float64
		if numOnRack > 0 && wordIndex != unfoundIndex {
			val = klv.LeaveValueByIndex(wordIndex - 1)
		}
		gen.leavemap.values[gen.leavemap.currentIndex] = val
		if val > gen.shadow.bestLeaves[numOnRack] {
			gen.shadow.bestLeaves[numOnRack] = val
		}
		return
	}

	// First recurse without taking any of this letter
	gen.populateLeaveMapIncremental(rack, leaveKWG, klv, nodeIndex, wordIndex, ml+1)

	// Then try taking 1, 2, ..., count copies into the leave
	numthis := rack.LetArr[ml]
	curNode := nodeIndex
	curWord := wordIndex
	for i := 0; i < numthis; i++ {
		rack.Take(ml)
		gen.leavemap.takeLetter(ml, rack.LetArr[ml])

		// Advance in the leave KWG
		curNode, curWord = incrementNodeToML(leaveKWG, curNode, curWord, ml)
		if curNode == 0 {
			// Dead path — fill zeros for remaining subsets
			gen.populateLeaveMapZero(rack, ml+1)
			for j := i; j >= 0; j-- {
				gen.leavemap.addLetter(ml, rack.LetArr[ml])
				rack.Add(ml)
			}
			return
		}
		childNode, childWord := followArc(leaveKWG, curNode, curWord)
		gen.populateLeaveMapIncremental(rack, leaveKWG, klv, childNode, childWord, ml+1)
	}
	for i := 0; i < numthis; i++ {
		gen.leavemap.addLetter(ml, rack.LetArr[ml])
		rack.Add(ml)
	}
}

// populateLeaveMapZero fills in 0 values for all remaining subsets
// when the KWG path is dead.
func (gen *GordonGenerator) populateLeaveMapZero(rack *tilemapping.Rack, ml tilemapping.MachineLetter) {
	for int(ml) < len(rack.LetArr) && rack.LetArr[ml] == 0 {
		ml++
	}
	if int(ml) == len(rack.LetArr) {
		gen.leavemap.values[gen.leavemap.currentIndex] = 0
		return
	}
	gen.populateLeaveMapZero(rack, ml+1)
	numthis := rack.LetArr[ml]
	for i := 0; i < numthis; i++ {
		rack.Take(ml)
		gen.leavemap.takeLetter(ml, rack.LetArr[ml])
		gen.populateLeaveMapZero(rack, ml+1)
	}
	for i := 0; i < numthis; i++ {
		gen.leavemap.addLetter(ml, rack.LetArr[ml])
		rack.Add(ml)
	}
}
