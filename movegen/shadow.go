package movegen

import (
	"math"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/word-golib/tilemapping"
)

// Shadow generation algorithm for move generation optimization.
// This is based on Magpie's shadow algorithm, which provides tight upper bounds
// on play equity to enable early pruning during move generation.

// playDirection returns the direction of play (Horizontal when playing horizontally,
// Vertical when playing vertically). This is the direction for extension sets.
func (gen *GordonGenerator) playDirection() board.BoardDirection {
	if gen.vertical {
		return board.VerticalDirection
	}
	return board.HorizontalDirection
}

// initShadowState prepares shadow state from rack (no allocations)
func (gen *GordonGenerator) initShadowState(rack *tilemapping.Rack) {
	// Compute rack cross-set (bitmask of letters we have)
	gen.rackCrossSet = 0
	for ml := 0; ml < len(rack.LetArr); ml++ {
		if rack.LetArr[ml] > 0 {
			gen.rackCrossSet |= uint64(1) << ml
		}
	}

	// Compute descending tile scores
	gen.setDescendingTileScores(rack)

	// Cache rack size
	gen.numLettersOnRack = int(rack.NumTiles())
}

// setDescendingTileScores fills descendingTileScores in descending score order.
// Uses insertion sort since rack size is at most 7.
func (gen *GordonGenerator) setDescendingTileScores(rack *tilemapping.Rack) {
	idx := 0
	// Iterate through all letters in the rack
	for ml := 0; ml < len(rack.LetArr); ml++ {
		count := int(rack.LetArr[ml])
		if count == 0 {
			continue
		}
		score := gen.tileScores[ml]
		// Insert 'count' copies of this score in descending order
		for c := 0; c < count; c++ {
			// Insertion sort: find position and shift
			insertIdx := idx
			for insertIdx > 0 && gen.descendingTileScores[insertIdx-1] < score {
				gen.descendingTileScores[insertIdx] = gen.descendingTileScores[insertIdx-1]
				insertIdx--
			}
			gen.descendingTileScores[insertIdx] = score
			idx++
		}
	}
	// Zero remaining slots
	for i := idx; i < 7; i++ {
		gen.descendingTileScores[i] = 0
	}
	// Save copy for full rack restoration
	copy(gen.fullRackDescTileScores[:], gen.descendingTileScores[:])
}

// genShadow performs shadow generation for all anchors.
// It estimates the maximum possible equity for each anchor, then heapifies
// the anchors for processing in descending equity order.
// If bestLeavesAlreadyComputed is true, bestLeaves was already computed during
// exchange generation (optimization to avoid double enumeration).
func (gen *GordonGenerator) genShadow(rack *tilemapping.Rack, bestLeavesAlreadyComputed bool) {
	// Compute best leaves for equity upper bounds (must be done before shadow)
	// Skip if already computed during exchange generation
	if !bestLeavesAlreadyComputed {
		gen.computeBestLeaves(rack)
	}

	gen.initShadowState(rack)
	gen.anchorCount = 0

	// Shadow horizontal direction
	gen.vertical = false
	gen.shadowByOrientation(rack, board.HorizontalDirection)

	// Shadow vertical direction
	gen.board.Transpose()
	gen.vertical = true
	gen.shadowByOrientation(rack, board.VerticalDirection)
	gen.board.Transpose()

	// Heapify anchors by equity (in-place, no allocation)
	gen.heapifyAnchors()
}

// shadowByOrientation processes all rows in one direction
func (gen *GordonGenerator) shadowByOrientation(rack *tilemapping.Rack, dir board.BoardDirection) {
	for row := 0; row < gen.boardDim; row++ {
		gen.curRowIdx = row
		gen.lastAnchorCol = -1

		for col := 0; col < gen.boardDim; col++ {
			if gen.board.IsAnchor(row, col, dir) {
				gen.shadowPlayForAnchor(rack, col)
				gen.lastAnchorCol = col
				// If this is a playthrough anchor (tile on this square),
				// advance lastAnchorCol past the tile
				if gen.board.HasLetter(row, col) {
					gen.lastAnchorCol = col + 1
				}
			}
		}
	}
}

// shadowPlayForAnchor computes max equity for one anchor
func (gen *GordonGenerator) shadowPlayForAnchor(rack *tilemapping.Rack, col int) {
	// Reset shadow state for this anchor
	gen.currentLeftCol = col
	gen.currentRightCol = col
	gen.curAnchorCol = col
	gen.shadowTilesPlayed = 0

	sqIdx := gen.board.GetSqIdx(gen.curRowIdx, col)
	playDir := gen.playDirection()

	// Get extension sets for this anchor (using play direction, not cross-set direction)
	gen.anchorLeftExtSet = gen.board.GetLeftExtSetIdx(sqIdx, playDir)
	gen.anchorRightExtSet = gen.board.GetRightExtSetIdx(sqIdx, playDir)

	// Reset multiplier tracking
	gen.numUnrestrictedMuls = 0
	gen.lastWordMultiplier = 1
	for i := range gen.descendingEffLetterMuls {
		gen.descendingEffLetterMuls[i] = 0
		gen.descendingCrossWordMuls[i] = 0
	}

	// Reset tile scores to full rack
	copy(gen.descendingTileScores[:], gen.fullRackDescTileScores[:])

	// Reset score accumulators
	gen.shadowMainwordRestrictedScore = 0
	gen.shadowPerpAdditionalScore = 0
	gen.shadowWordMultiplier = 1

	// Reset result tracking
	gen.highestShadowEquity = 0
	gen.highestShadowScore = 0

	// Save rack state
	gen.shadowRackCopy = *rack
	gen.rackCrossSetCopy = gen.rackCrossSet

	// Perform shadow
	gen.shadowStart(rack)

	if gen.shadowTilesPlayed == 0 {
		return // No valid plays possible from this anchor
	}

	// Record anchor result
	gen.anchors[gen.anchorCount] = Anchor{
		HighestPossibleEquity: gen.highestShadowEquity,
		HighestPossibleScore:  int16(gen.highestShadowScore),
		Row:                   uint8(gen.curRowIdx),
		Col:                   uint8(col),
		LastAnchorCol:         int8(gen.lastAnchorCol),
		Vertical:              gen.vertical,
	}
	gen.anchorCount++

	// Restore rack state
	*rack = gen.shadowRackCopy
	gen.rackCrossSet = gen.rackCrossSetCopy
}

// shadowStart dispatches to playthrough or non-playthrough shadow
func (gen *GordonGenerator) shadowStart(rack *tilemapping.Rack) {
	currentLetter := gen.board.GetLetter(gen.curRowIdx, gen.currentLeftCol)
	if currentLetter == 0 {
		gen.shadowStartNonplaythrough(rack)
	} else {
		gen.shadowStartPlaythrough(rack, currentLetter)
	}
}

// shadowStartNonplaythrough handles empty anchor squares
func (gen *GordonGenerator) shadowStartNonplaythrough(rack *tilemapping.Rack) {
	// Play first tile on the empty anchor square
	sqIdx := gen.board.GetSqIdx(gen.curRowIdx, gen.curAnchorCol)
	csDir := gen.crossDirection()
	crossSet := gen.board.GetCrossSetIdx(sqIdx, csDir)

	// Check if we can play anything here
	possibleLetters := uint64(crossSet) & gen.rackCrossSet
	if possibleLetters == 0 {
		return
	}

	// Get multipliers for this square
	letterMul := gen.board.GetLetterMultiplier(sqIdx)
	wordMul := gen.board.GetWordMultiplier(sqIdx)
	crossScore := gen.board.GetCrossScoreIdx(sqIdx, csDir)

	// Try to restrict to a single tile if cross-set forces it
	if gen.tryRestrictTile(possibleLetters, letterMul, wordMul, crossScore, gen.curAnchorCol) {
		// Successfully restricted - score already accumulated
	} else {
		// Multiple tiles possible - add unrestricted multipliers
		gen.insertUnrestrictedMultiplier(letterMul, wordMul, gen.curAnchorCol)
	}

	gen.shadowTilesPlayed++
	gen.shadowWordMultiplier *= wordMul

	// Record single-tile horizontal play (for uniqueness)
	gen.shadowRecord(rack)

	// Extend left
	gen.nonplaythroughShadowPlayLeft(rack)
}

// shadowStartPlaythrough handles occupied anchor squares
func (gen *GordonGenerator) shadowStartPlaythrough(rack *tilemapping.Rack, currentLetter tilemapping.MachineLetter) {
	// Traverse existing tiles on board from right to left, accumulating scores
	col := gen.curAnchorCol
	for col >= 0 && gen.board.HasLetter(gen.curRowIdx, col) {
		ml := gen.board.GetLetter(gen.curRowIdx, col)
		score := gen.tileScores[ml.Unblank()]
		gen.shadowMainwordRestrictedScore += score
		col--
	}
	gen.currentLeftCol = col + 1

	// Continue left from here
	gen.playthroughShadowPlayLeft(rack)
}

// nonplaythroughShadowPlayLeft extends left from anchor (non-playthrough case)
// Follows Magpie's pattern: call shadowPlayRight at the start of each iteration,
// before checking if we can extend left further.
func (gen *GordonGenerator) nonplaythroughShadowPlayLeft(rack *tilemapping.Rack) {
	col := gen.curAnchorCol

	for {
		// First, try extending right from current position
		possibleTilesRight := gen.anchorRightExtSet & gen.rackCrossSet
		if possibleTilesRight != 0 {
			gen.shadowPlayRight(rack)
		}
		gen.anchorRightExtSet = board.TrivialCrossSet

		// Check exit conditions for left extension
		if col == 0 || col == gen.lastAnchorCol+1 || gen.shadowTilesPlayed >= gen.numLettersOnRack {
			break
		}

		// Now try to extend left
		col--

		// Check if this square has a tile already (playthrough)
		if gen.board.HasLetter(gen.curRowIdx, col) {
			ml := gen.board.GetLetter(gen.curRowIdx, col)
			gen.shadowMainwordRestrictedScore += gen.tileScores[ml.Unblank()]
			continue
		}

		sqIdx := gen.board.GetSqIdx(gen.curRowIdx, col)
		csDir := gen.crossDirection()
		playDir := gen.playDirection()
		crossSet := gen.board.GetCrossSetIdx(sqIdx, csDir)
		leftExtSet := gen.board.GetLeftExtSetIdx(sqIdx, playDir)

		// Check if we can play anything here
		possibleLetters := uint64(crossSet) & gen.rackCrossSet & leftExtSet
		if possibleLetters == 0 {
			break
		}

		letterMul := gen.board.GetLetterMultiplier(sqIdx)
		wordMul := gen.board.GetWordMultiplier(sqIdx)
		crossScore := gen.board.GetCrossScoreIdx(sqIdx, csDir)

		if gen.tryRestrictTile(possibleLetters, letterMul, wordMul, crossScore, col) {
			// Successfully restricted
		} else {
			gen.insertUnrestrictedMultiplier(letterMul, wordMul, col)
		}

		gen.shadowTilesPlayed++
		gen.shadowWordMultiplier *= wordMul

		// Record play
		gen.shadowRecord(rack)
	}

	gen.currentLeftCol = col
}

// playthroughShadowPlayLeft extends left from playthrough tiles
// Follows Magpie's pattern: call shadowPlayRight at the start of each iteration,
// before checking if we can extend left further.
func (gen *GordonGenerator) playthroughShadowPlayLeft(rack *tilemapping.Rack) {
	col := gen.currentLeftCol

	for {
		// First, try extending right from current position
		possibleTilesRight := gen.anchorRightExtSet & gen.rackCrossSet
		if possibleTilesRight != 0 {
			gen.shadowPlayRight(rack)
		}
		gen.anchorRightExtSet = board.TrivialCrossSet

		// Check exit conditions for left extension
		if col == 0 || col == gen.lastAnchorCol+1 || gen.shadowTilesPlayed >= gen.numLettersOnRack {
			break
		}

		// Check if we can extend left
		possibleTilesLeft := gen.anchorLeftExtSet & gen.rackCrossSet
		gen.anchorLeftExtSet = board.TrivialCrossSet
		if possibleTilesLeft == 0 {
			break
		}

		// Now try to extend left
		col--

		// Check if this square has a tile already (playthrough)
		if gen.board.HasLetter(gen.curRowIdx, col) {
			ml := gen.board.GetLetter(gen.curRowIdx, col)
			gen.shadowMainwordRestrictedScore += gen.tileScores[ml.Unblank()]
			continue
		}

		sqIdx := gen.board.GetSqIdx(gen.curRowIdx, col)
		csDir := gen.crossDirection()
		crossSet := gen.board.GetCrossSetIdx(sqIdx, csDir)

		possibleLetters := uint64(crossSet) & possibleTilesLeft
		if possibleLetters == 0 {
			break
		}

		letterMul := gen.board.GetLetterMultiplier(sqIdx)
		wordMul := gen.board.GetWordMultiplier(sqIdx)
		crossScore := gen.board.GetCrossScoreIdx(sqIdx, csDir)

		gen.shadowPerpAdditionalScore += crossScore * wordMul
		gen.shadowWordMultiplier *= wordMul

		if gen.tryRestrictTile(possibleLetters, letterMul, wordMul, crossScore, col) {
			// Successfully restricted
		} else {
			gen.insertUnrestrictedMultiplier(letterMul, wordMul, col)
		}

		gen.shadowTilesPlayed++

		// Record play
		gen.shadowRecord(rack)
	}

	gen.currentLeftCol = col
}

// shadowPlayRight extends right from current position
func (gen *GordonGenerator) shadowPlayRight(rack *tilemapping.Rack) {
	// Save state for restoration
	savedTilesPlayed := gen.shadowTilesPlayed
	savedMainword := gen.shadowMainwordRestrictedScore
	savedPerp := gen.shadowPerpAdditionalScore
	savedWordMul := gen.shadowWordMultiplier
	savedLastWordMul := gen.lastWordMultiplier
	savedNumUnrest := gen.numUnrestrictedMuls
	copy(gen.descTileScoresCopy[:], gen.descendingTileScores[:])
	copy(gen.descEffLetterMulsCopy[:], gen.descendingEffLetterMuls[:])
	copy(gen.descCrossWordMulsCopy[:], gen.descendingCrossWordMuls[:])
	copy(gen.descLetterMulsCopy[:], gen.descendingLetterMuls[:])

	col := gen.curAnchorCol + 1

	for col < gen.boardDim && gen.shadowTilesPlayed < gen.numLettersOnRack {
		// Check if this square has a tile already
		if gen.board.HasLetter(gen.curRowIdx, col) {
			ml := gen.board.GetLetter(gen.curRowIdx, col)
			gen.shadowMainwordRestrictedScore += gen.tileScores[ml.Unblank()]
			col++
			continue
		}

		sqIdx := gen.board.GetSqIdx(gen.curRowIdx, col)
		csDir := gen.crossDirection()
		playDir := gen.playDirection()
		crossSet := gen.board.GetCrossSetIdx(sqIdx, csDir)
		leftExtSet := gen.board.GetLeftExtSetIdx(sqIdx, playDir)

		// For rightward extension, also check rightExtSet at anchor
		possibleLetters := uint64(crossSet) & gen.rackCrossSet & leftExtSet
		if gen.shadowTilesPlayed == 0 {
			possibleLetters &= gen.anchorRightExtSet
		}
		if possibleLetters == 0 {
			break
		}

		letterMul := gen.board.GetLetterMultiplier(sqIdx)
		wordMul := gen.board.GetWordMultiplier(sqIdx)
		crossScore := gen.board.GetCrossScoreIdx(sqIdx, csDir)

		if gen.tryRestrictTile(possibleLetters, letterMul, wordMul, crossScore, col) {
			// Successfully restricted
		} else {
			gen.insertUnrestrictedMultiplier(letterMul, wordMul, col)
		}

		gen.shadowTilesPlayed++
		gen.shadowWordMultiplier *= wordMul

		gen.shadowRecord(rack)

		col++
	}

	gen.currentRightCol = col - 1

	// Restore state
	gen.shadowTilesPlayed = savedTilesPlayed
	gen.shadowMainwordRestrictedScore = savedMainword
	gen.shadowPerpAdditionalScore = savedPerp
	gen.shadowWordMultiplier = savedWordMul
	gen.lastWordMultiplier = savedLastWordMul
	gen.numUnrestrictedMuls = savedNumUnrest
	copy(gen.descendingTileScores[:], gen.descTileScoresCopy[:])
	copy(gen.descendingEffLetterMuls[:], gen.descEffLetterMulsCopy[:])
	copy(gen.descendingCrossWordMuls[:], gen.descCrossWordMulsCopy[:])
	copy(gen.descendingLetterMuls[:], gen.descLetterMulsCopy[:])
}

// shadowRecord records shadow result for current position
func (gen *GordonGenerator) shadowRecord(rack *tilemapping.Rack) {
	if gen.shadowTilesPlayed == 0 {
		return
	}

	// Recalculate effective multipliers if word multiplier changed
	gen.maybeRecalculateEffectiveMultipliers()

	// Calculate score: mainword * wordMul + perp + unrestricted tiles
	mainwordScore := gen.shadowMainwordRestrictedScore

	// Add unrestricted tile scores (greedy: pair best tiles with best multipliers)
	unrestrictedScore := 0
	for i := 0; i < gen.numUnrestrictedMuls && i < gen.shadowTilesPlayed; i++ {
		tileScore := gen.descendingTileScores[i]
		effMul := int(gen.descendingEffLetterMuls[i])
		unrestrictedScore += tileScore * effMul
	}

	totalScore := mainwordScore*gen.shadowWordMultiplier + gen.shadowPerpAdditionalScore + unrestrictedScore

	// Add bingo bonus if 7 tiles played
	if gen.shadowTilesPlayed >= 7 {
		totalScore += 50 // Standard bingo bonus
	}

	// Compute equity upper bound = score + best possible leave value for this leave size
	leaveSize := gen.numLettersOnRack - gen.shadowTilesPlayed
	if leaveSize < 0 {
		leaveSize = 0
	}
	if leaveSize >= len(gen.bestLeaves) {
		leaveSize = len(gen.bestLeaves) - 1
	}
	equity := float64(totalScore) + gen.bestLeaves[leaveSize]

	if equity > gen.highestShadowEquity {
		gen.highestShadowEquity = equity
	}
	if totalScore > gen.highestShadowScore {
		gen.highestShadowScore = totalScore
	}
}

// tryRestrictTile attempts to restrict to a single tile (when cross-set forces it).
// Returns true if successfully restricted, false if multiple tiles possible.
func (gen *GordonGenerator) tryRestrictTile(possibleLetters uint64, letterMul, wordMul int, crossScore int, col int) bool {
	// Check if cross-set is a single bit (forces one letter)
	// A number is a power of 2 (single bit) if n & (n-1) == 0 and n != 0
	// But we need to exclude bit 0 (blank) from this check
	nonblankLetters := possibleLetters & ^uint64(1)
	if nonblankLetters != 0 && (nonblankLetters&(nonblankLetters-1)) == 0 {
		// Single letter forced (excluding blank)
		// Find which letter it is
		ml := 0
		for ; ml < 64; ml++ {
			if (nonblankLetters & (uint64(1) << ml)) != 0 {
				break
			}
		}

		score := gen.tileScores[ml]
		gen.shadowMainwordRestrictedScore += score * letterMul

		// Add cross-word score if there's a perpendicular word
		if crossScore > 0 {
			gen.shadowPerpAdditionalScore += (score*letterMul + crossScore) * wordMul
		}

		// Remove this tile from descending scores (shift down)
		gen.removeTileFromDescending(score)

		return true
	}
	return false
}

// insertUnrestrictedMultiplier adds multiplier info in sorted order
func (gen *GordonGenerator) insertUnrestrictedMultiplier(letterMul, wordMul int, col int) {
	effMul := uint16(letterMul * gen.shadowWordMultiplier * wordMul)

	// Insert in descending order
	insertIdx := gen.numUnrestrictedMuls
	for insertIdx > 0 && gen.descendingEffLetterMuls[insertIdx-1] < effMul {
		gen.descendingEffLetterMuls[insertIdx] = gen.descendingEffLetterMuls[insertIdx-1]
		gen.descendingCrossWordMuls[insertIdx] = gen.descendingCrossWordMuls[insertIdx-1]
		gen.descendingLetterMuls[insertIdx] = gen.descendingLetterMuls[insertIdx-1]
		insertIdx--
	}
	gen.descendingEffLetterMuls[insertIdx] = effMul
	gen.descendingCrossWordMuls[insertIdx] = uint16(wordMul<<8) | uint16(col&0xff)
	gen.descendingLetterMuls[insertIdx] = uint8(letterMul)
	gen.numUnrestrictedMuls++
}

// maybeRecalculateEffectiveMultipliers recalculates effective letter multipliers
// when the word multiplier has changed since they were last computed.
// This is needed because effective_mul = letter_mul * word_mul * crossword_mul,
// and word_mul accumulates as we traverse the row.
func (gen *GordonGenerator) maybeRecalculateEffectiveMultipliers() {
	if gen.lastWordMultiplier == gen.shadowWordMultiplier {
		return
	}
	gen.lastWordMultiplier = gen.shadowWordMultiplier

	// Recalculate and re-sort effective letter multipliers
	// We need to recalculate based on the current shadowWordMultiplier
	for i := 0; i < gen.numUnrestrictedMuls; i++ {
		letterMul := int(gen.descendingLetterMuls[i])
		crossWordMul := int(gen.descendingCrossWordMuls[i] >> 8)
		newEffMul := uint16(letterMul*gen.shadowWordMultiplier + crossWordMul)
		gen.descendingEffLetterMuls[i] = newEffMul
	}

	// Re-sort in descending order (insertion sort for small arrays)
	for i := 1; i < gen.numUnrestrictedMuls; i++ {
		key := gen.descendingEffLetterMuls[i]
		keyCW := gen.descendingCrossWordMuls[i]
		keyLM := gen.descendingLetterMuls[i]
		j := i - 1
		for j >= 0 && gen.descendingEffLetterMuls[j] < key {
			gen.descendingEffLetterMuls[j+1] = gen.descendingEffLetterMuls[j]
			gen.descendingCrossWordMuls[j+1] = gen.descendingCrossWordMuls[j]
			gen.descendingLetterMuls[j+1] = gen.descendingLetterMuls[j]
			j--
		}
		gen.descendingEffLetterMuls[j+1] = key
		gen.descendingCrossWordMuls[j+1] = keyCW
		gen.descendingLetterMuls[j+1] = keyLM
	}
}

// removeTileFromDescending removes a tile score from the descending array
func (gen *GordonGenerator) removeTileFromDescending(score int) {
	// Find and remove the score
	for i := 0; i < 7; i++ {
		if gen.descendingTileScores[i] == score {
			// Shift remaining elements left
			for j := i; j < 6; j++ {
				gen.descendingTileScores[j] = gen.descendingTileScores[j+1]
			}
			gen.descendingTileScores[6] = 0
			return
		}
	}
}

// recordScoringPlaysFromAnchors processes anchors in equity order and generates
// actual moves. This is the second phase of shadow-based generation.
func (gen *GordonGenerator) recordScoringPlaysFromAnchors(rack *tilemapping.Rack) {
	gen.tilesPlayed = 0

	for gen.anchorCount > 0 {
		anchor := gen.popAnchor()

		// Early termination: if best found move has equity >= max possible from
		// this anchor, we can skip all remaining anchors (they have lower max equity)
		if !gen.winner.IsEmpty() && gen.winner.Equity() >= anchor.HighestPossibleEquity {
			break
		}

		// Set up generator state for this anchor
		gen.curRowIdx = int(anchor.Row)
		gen.curAnchorCol = int(anchor.Col)
		gen.lastAnchorCol = int(anchor.LastAnchorCol)
		gen.vertical = anchor.Vertical

		// Handle board transposition for vertical anchors
		if anchor.Vertical {
			if !gen.board.IsTransposed() {
				gen.board.Transpose()
			}
		} else {
			if gen.board.IsTransposed() {
				gen.board.Transpose()
			}
		}

		// Get rightx for filtering during generation
		sqIdx := gen.board.GetSqIdx(gen.curRowIdx, int(anchor.Col))
		playDir := gen.playDirection()
		gen.anchorRightExtSet = gen.board.GetRightExtSetIdx(sqIdx, playDir)

		// Generate actual moves from this anchor using the standard recursive algorithm
		gen.recursiveGen(int(anchor.Col), rack, gen.gaddag.GetRootNodeIndex(),
			int(anchor.Col), int(anchor.Col), !anchor.Vertical, 0, 0, 1)
	}

	// Ensure board is not transposed when we're done
	if gen.board.IsTransposed() {
		gen.board.Transpose()
	}
}

// computeBestLeaves computes the maximum leave value for each leave size (0-7).
// This is used for shadow equity upper bounds. We need an upper bound on leave
// equity to ensure shadow pruning doesn't incorrectly eliminate good moves.
// This uses the same enumeration pattern as exchange generation.
func (gen *GordonGenerator) computeBestLeaves(rack *tilemapping.Rack) {
	// Initialize to very negative values
	for i := range gen.bestLeaves {
		gen.bestLeaves[i] = -math.MaxFloat64
	}

	// Find a leave calculator among the equity calculators
	var leaveCalc equity.Leaves
	for _, calc := range gen.equityCalculators {
		if lc, ok := calc.(equity.Leaves); ok {
			leaveCalc = lc
			break
		}
	}

	// If no leave calculator, leave values are 0
	if leaveCalc == nil {
		for i := range gen.bestLeaves {
			gen.bestLeaves[i] = 0
		}
		return
	}

	// Enumerate all possible leaves (same pattern as exchange generation)
	// and track best leave value for each leave size
	gen.computeBestLeavesFromExchanges(rack, leaveCalc, 0)
}
