package movegen

import (
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

// populateLeaveMap enumerates all rack subsets using incremental KWG
// traversal matching magpie's generate_exchange_moves. The KWG walk
// tracks nodeIndex/wordIndex as tiles are added to the leave.
//
// The leave map itself lives in macondo/leavemap; this file holds
// only the populator (which is tightly coupled to GordonGenerator's
// klv, peg, and shadow state).
func (gen *GordonGenerator) populateLeaveMap(rack *tilemapping.Rack) {
	klv := gen.klv
	leaveKWG := klv.KWG()
	rootIdx := leaveKWG.ArcIndex(0)
	gen.populateLeaveMapIncremental(rack, leaveKWG, rootIdx, 0, 0)
}

func (gen *GordonGenerator) populateLeaveMapIncremental(
	rack *tilemapping.Rack,
	leaveKWG *kwg.KWG,
	nodeIndex uint32,
	wordIndex int32,
	ml tilemapping.MachineLetter,
) {
	for int(ml) < len(rack.LetArr) && rack.LetArr[ml] == 0 {
		ml++
	}
	if int(ml) == len(rack.LetArr) {
		// Leaf: compute equity-adjusted leave value.
		numOnRack := int(rack.NumTiles())
		var rawLeave float64
		var val float64
		if gen.tilesInBag > 0 {
			if numOnRack > 0 && wordIndex >= 0 {
				rawLeave = gen.klv.LeaveValueByIndex(wordIndex - 1)
			}
			val = rawLeave
			tilesPlayed := gen.leavemap.TotalTiles - numOnRack
			bagPlusSeven := gen.tilesInBag - tilesPlayed + 7
			if bagPlusSeven >= 0 && bagPlusSeven < len(gen.pegValues) {
				val += gen.pegValues[bagPlusSeven]
			}
		} else {
			if numOnRack > 0 {
				leaveScore := 0
				for lml := tilemapping.MachineLetter(0); int(lml) < len(rack.LetArr); lml++ {
					leaveScore += rack.LetArr[lml] * gen.letterDistribution.Score(lml)
				}
				val = float64(-endgameNonOutplayLeavePenaltyMultiplier*leaveScore) - endgameNonOutplayConstantPenalty
			} else {
				val = float64(2 * gen.oppRackScore)
			}
			rawLeave = val
		}
		gen.leavemap.Values[gen.leavemap.CurrentIndex] = val
		// Update bestLeaves from raw leave (no peg) — only reachable indices.
		if rawLeave > gen.shadow.bestLeaves[numOnRack] {
			gen.shadow.bestLeaves[numOnRack] = rawLeave
		}
		return
	}

	// Remove all copies of ml from rack first (leave has 0 copies).
	numthis := rack.LetArr[ml]
	for i := 0; i < numthis; i++ {
		rack.Take(ml)
		gen.leavemap.TakeLetter(ml, rack.LetArr[ml])
	}

	// Recurse with 0 copies of ml in leave — KWG state unchanged.
	gen.populateLeaveMapIncremental(rack, leaveKWG, nodeIndex, wordIndex, ml+1)

	// Add copies back one at a time, advancing KWG for each.
	// curNode/curWord persist across iterations (matching magpie where
	// node_index and word_index carry over through increment + follow_arc).
	curNode := nodeIndex
	curWord := wordIndex
	for i := 0; i < numthis; i++ {
		gen.leavemap.AddLetter(ml, rack.LetArr[ml])
		rack.Add(ml)

		// incrementNodeToML: find sibling with tile ml
		curNode, curWord = incrementNodeToML(leaveKWG, curNode, curWord, ml)
		if curNode == 0 {
			// Dead path — fill zeros for remaining subsets
			for j := i + 1; j < numthis; j++ {
				gen.leavemap.AddLetter(ml, rack.LetArr[ml])
				rack.Add(ml)
			}
			gen.populateLeaveMapZero(rack, ml+1)
			return
		}

		// followArc: always increment wordIndex by 1, move to children.
		// Both curNode and curWord persist to the NEXT iteration so the
		// second copy of the same tile searches at the child level.
		curNode = leaveKWG.ArcIndex(curNode)
		curWord = curWord + 1

		gen.populateLeaveMapIncremental(rack, leaveKWG, curNode, curWord, ml+1)
	}
}

// populateLeaveMapZero fills 0 values for remaining subsets when KWG path is dead.
func (gen *GordonGenerator) populateLeaveMapZero(rack *tilemapping.Rack, ml tilemapping.MachineLetter) {
	for int(ml) < len(rack.LetArr) && rack.LetArr[ml] == 0 {
		ml++
	}
	if int(ml) == len(rack.LetArr) {
		numOnRack := int(rack.NumTiles())
		var val float64
		if gen.tilesInBag > 0 {
			// Leave not in KLV (dead path) — leave value 0, apply peg for recorder.
			tilesPlayed := gen.leavemap.TotalTiles - numOnRack
			bagPlusSeven := gen.tilesInBag - tilesPlayed + 7
			if bagPlusSeven >= 0 && bagPlusSeven < len(gen.pegValues) {
				val = gen.pegValues[bagPlusSeven]
			}
		} else if numOnRack > 0 {
			leaveScore := 0
			for lml := tilemapping.MachineLetter(0); int(lml) < len(rack.LetArr); lml++ {
				leaveScore += rack.LetArr[lml] * gen.letterDistribution.Score(lml)
			}
			val = float64(-endgameNonOutplayLeavePenaltyMultiplier*leaveScore) - endgameNonOutplayConstantPenalty
		} else {
			val = float64(2 * gen.oppRackScore)
		}
		gen.leavemap.Values[gen.leavemap.CurrentIndex] = val
		// Dead path: rawLeave = 0 for mid-game, endgame adjustment otherwise
		var rawLeave float64
		if gen.tilesInBag <= 0 {
			rawLeave = val
		}
		if rawLeave > gen.shadow.bestLeaves[numOnRack] {
			gen.shadow.bestLeaves[numOnRack] = rawLeave
		}
		return
	}
	gen.populateLeaveMapZero(rack, ml+1)
	numthis := rack.LetArr[ml]
	for i := 0; i < numthis; i++ {
		gen.leavemap.AddLetter(ml, rack.LetArr[ml])
		rack.Add(ml)
		gen.populateLeaveMapZero(rack, ml+1)
	}
	for i := 0; i < numthis; i++ {
		rack.Take(ml)
		gen.leavemap.TakeLetter(ml, rack.LetArr[ml])
	}
}

// incrementNodeToML scans siblings to find tile ml, updating wordIndex
// for skipped subtrees. Matches magpie's increment_node_to_ml.
func incrementNodeToML(kwg *kwg.KWG, nodeIndex uint32, wordIndex int32, ml tilemapping.MachineLetter) (uint32, int32) {
	if nodeIndex == 0 {
		return 0, -1
	}
	wIdx := wordIndex
	for {
		if kwg.Tile(nodeIndex) == uint8(ml) {
			return nodeIndex, wIdx
		}
		if kwg.IsEnd(nodeIndex) {
			return 0, -1
		}
		wIdx += kwg.WordCountAt(nodeIndex) - kwg.WordCountAt(nodeIndex+1)
		nodeIndex++
	}
}
