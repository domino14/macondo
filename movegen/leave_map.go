package movegen

import (
	"github.com/domino14/word-golib/tilemapping"

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

// populateLeaveMap enumerates all rack subsets and stores equity-adjusted
// leave values. Uses KLV.LeaveValue per subset (sort + KWG walk).
// TODO: port magpie's incremental KWG traversal for ~2% speedup.
func (gen *GordonGenerator) populateLeaveMap(rack *tilemapping.Rack) {
	gen.populateLeaveMapRecursive(rack, 0)
}

func (gen *GordonGenerator) populateLeaveMapRecursive(rack *tilemapping.Rack, ml tilemapping.MachineLetter) {
	for int(ml) < len(rack.LetArr) && rack.LetArr[ml] == 0 {
		ml++
	}
	if int(ml) == len(rack.LetArr) {
		numOnRack := int(rack.NumTiles())
		var val float64
		if gen.tilesInBag > 0 {
			// Build leave from current rack state and look up value
			var leave [game.RackTileLimit]tilemapping.MachineLetter
			n := 0
			for lml := tilemapping.MachineLetter(0); int(lml) < len(rack.LetArr); lml++ {
				for j := 0; j < rack.LetArr[lml]; j++ {
					leave[n] = lml
					n++
				}
			}
			val = gen.klv.LeaveValue(leave[:n])
			// Bake in pre-endgame peg adjustment
			tilesPlayed := gen.leavemap.totalTiles - numOnRack
			bagPlusSeven := gen.tilesInBag - tilesPlayed + 7
			if bagPlusSeven >= 0 && bagPlusSeven < len(gen.pegValues) {
				val += gen.pegValues[bagPlusSeven]
			}
		} else {
			// Endgame: compute adjustment instead of leave value
			if numOnRack > 0 {
				leaveScore := 0
				for lml := tilemapping.MachineLetter(0); int(lml) < len(rack.LetArr); lml++ {
					leaveScore += rack.LetArr[lml] * gen.letterDistribution.Score(lml)
				}
				val = float64(-endgameNonOutplayLeavePenaltyMultiplier*leaveScore) - endgameNonOutplayConstantPenalty
			} else {
				val = float64(2 * gen.oppRackScore)
			}
		}
		gen.leavemap.values[gen.leavemap.currentIndex] = val
		if val > gen.shadow.bestLeaves[numOnRack] {
			gen.shadow.bestLeaves[numOnRack] = val
		}
		return
	}
	gen.populateLeaveMapRecursive(rack, ml+1)
	numthis := rack.LetArr[ml]
	for i := 0; i < numthis; i++ {
		rack.Take(ml)
		gen.leavemap.takeLetter(ml, rack.LetArr[ml])
		gen.populateLeaveMapRecursive(rack, ml+1)
	}
	for i := 0; i < numthis; i++ {
		gen.leavemap.addLetter(ml, rack.LetArr[ml])
		rack.Add(ml)
	}
}
