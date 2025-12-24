package movegen

import (
	"fmt"
	"math"
	"os"

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

	fmt.Fprintf(os.Stderr, "[SHADOW] Generated %d total anchors after heapify\n", gen.anchorCount)
	// Show all row=13 horizontal anchors
	for i := 0; i < gen.anchorCount; i++ {
		a := gen.anchors[i]
		if a.Row == 13 && !a.Vertical {
			fmt.Fprintf(os.Stderr, "[SHADOW] Anchor[%d]: row=%d col=%d vert=%v maxEq=%.2f maxScore=%d\n",
				i, a.Row, a.Col, a.Vertical, a.HighestPossibleEquity, a.HighestPossibleScore)
		}
	}
}

// shadowByOrientation processes all rows in one direction
func (gen *GordonGenerator) shadowByOrientation(rack *tilemapping.Rack, dir board.BoardDirection) {
	for row := 0; row < gen.boardDim; row++ {
		gen.curRowIdx = row
		gen.lastAnchorCol = -1

		for col := 0; col < gen.boardDim; col++ {
			// Debug logging - print entire rows 12 and 13 at start
			if (row == 12 || row == 13) && !gen.vertical && col == 0 {
				fmt.Fprintf(os.Stderr, "[SHADOW-DEBUG] Row %d tiles: ", row)
				for c := 0; c < gen.boardDim; c++ {
					if gen.board.HasLetter(row, c) {
						letter := gen.board.GetLetter(row, c)
						fmt.Fprintf(os.Stderr, "[%d:%d] ", c, letter)
					} else {
						fmt.Fprintf(os.Stderr, "[%d:.] ", c)
					}
				}
				fmt.Fprintf(os.Stderr, "\n")
			}
			// Debug anchor processing for rows 12 and 13
			if (row == 12 || row == 13) && !gen.vertical && gen.board.IsAnchor(row, col, dir) {
				boardNotationRow := row + 1
				boardNotationCol := string(rune('A' + col))
				fmt.Fprintf(os.Stderr, "[SHADOW-DEBUG] Detected anchor at row=%d col=%d (%d%s) horizontal\n",
					row, col, boardNotationRow, boardNotationCol)
			}

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
	gen.maxShadowTilesPlayed = 0

	// Save rack state
	gen.shadowRackCopy = *rack
	gen.rackCrossSetCopy = gen.rackCrossSet

	// Perform shadow
	gen.shadowStart(rack)

	// Check if any valid plays were found during shadow exploration.
	// Note: shadowTilesPlayed may be 0 after backtracking, so check maxShadowTilesPlayed instead.
	if gen.maxShadowTilesPlayed == 0 {
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

	// Initialize perpendicular score (Magpie does this BEFORE restricting tiles)
	gen.shadowPerpAdditionalScore = crossScore * wordMul

	// Temporarily set word multiplier to 0 for single-tile recording (matches Magpie)
	gen.shadowWordMultiplier = 0

	// Try to restrict to a single tile if cross-set forces it
	if gen.tryRestrictTile(possibleLetters, letterMul, wordMul, crossScore, gen.curAnchorCol) {
		// Successfully restricted - score already accumulated
	} else {
		// Multiple tiles possible - add unrestricted multipliers
		gen.insertUnrestrictedMultiplier(letterMul, wordMul, gen.curAnchorCol, crossScore)
	}

	gen.shadowTilesPlayed++

	// Record single-tile horizontal play (for uniqueness) with word_multiplier = 0
	if !gen.vertical {
		gen.shadowRecord(rack)
	}

	// Now set the actual word multiplier
	gen.shadowWordMultiplier = wordMul

	// Recalculate effective multipliers (matches Magpie)
	gen.maybeRecalculateEffectiveMultipliers()

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
	for {
		// First, try extending right from current position
		possibleTilesRight := gen.anchorRightExtSet & gen.rackCrossSet
		if possibleTilesRight != 0 {
			gen.shadowPlayRight(rack)
		}
		gen.anchorRightExtSet = board.TrivialCrossSet

		// Check exit conditions for left extension
		if gen.currentLeftCol == 0 || gen.currentLeftCol == gen.lastAnchorCol+1 ||
			gen.shadowTilesPlayed >= gen.numLettersOnRack {
			return
		}

		// Check if we can extend left
		possibleTilesLeft := gen.anchorLeftExtSet & gen.rackCrossSet
		if possibleTilesLeft == 0 {
			return
		}
		gen.anchorLeftExtSet = board.TrivialCrossSet

		// Extend left (matches Magpie)
		gen.currentLeftCol--
		gen.shadowTilesPlayed++

		sqIdx := gen.board.GetSqIdx(gen.curRowIdx, gen.currentLeftCol)
		csDir := gen.crossDirection()
		crossSet := gen.board.GetCrossSetIdx(sqIdx, csDir)

		// Check possible letters at this position
		possibleLetters := uint64(crossSet) & possibleTilesLeft
		if possibleLetters == 0 {
			return
		}

		letterMul := gen.board.GetLetterMultiplier(sqIdx)
		wordMul := gen.board.GetWordMultiplier(sqIdx)
		crossScore := gen.board.GetCrossScoreIdx(sqIdx, csDir)

		gen.shadowWordMultiplier *= wordMul

		if gen.tryRestrictTile(possibleLetters, letterMul, wordMul, crossScore, gen.currentLeftCol) {
			// Successfully restricted
		} else {
			gen.insertUnrestrictedMultiplier(letterMul, wordMul, gen.currentLeftCol, crossScore)
		}

		// Record play
		gen.shadowRecord(rack)
	}
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
			gen.insertUnrestrictedMultiplier(letterMul, wordMul, col, crossScore)
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
		// Handle blanks: In Macondo, cross-sets don't include the blank bit (bit 0).
		// If we have a blank, we can designate it as any letter in the crossSet.
		hasBlank := (gen.rackCrossSet & 1) != 0
		if hasBlank {
			possibleLetters |= uint64(crossSet) & leftExtSet
		}
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
			if gen.curRowIdx == 13 && !gen.vertical && gen.curAnchorCol == 10 {
				fmt.Fprintf(os.Stderr, "[SHADOW-RIGHT] row=13 anchor=10 extended right to col=%d, mainword now=%d\n",
					col, gen.shadowMainwordRestrictedScore)
			}
		} else {
			gen.insertUnrestrictedMultiplier(letterMul, wordMul, col, crossScore)
		}

		gen.shadowTilesPlayed++
		gen.shadowWordMultiplier *= wordMul

		gen.shadowRecord(rack)

		col++
	}

	gen.currentRightCol = col - 1

	// Restore state
	gen.shadowTilesPlayed = savedTilesPlayed
	// Debug: check if we're resetting a non-zero mainword score
	if gen.curRowIdx == 13 && !gen.vertical && gen.shadowMainwordRestrictedScore != savedMainword {
		fmt.Fprintf(os.Stderr, "[SHADOW-RESTORE] row=13 resetting mainword from %d to %d (savedMainword=%d)\n",
			gen.shadowMainwordRestrictedScore, savedMainword, savedMainword)
	}
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

	// Debug for row 13
	if gen.curRowIdx == 13 && !gen.vertical && gen.curAnchorCol == 10 {
		fmt.Fprintf(os.Stderr, "[SHADOW-RECORD] row=13 col=10 tilesPlayed=%d mainwordScore=%d at entry\n",
			gen.shadowTilesPlayed, mainwordScore)
	}

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

	// Add endgame adjustment for upper bound calculation
	if gen.game != nil && gen.game.Bag().TilesRemaining() == 0 {
		if leaveSize == 0 {
			// Going out - add maximum possible endgame bonus (2 * opponent rack score)
			oppRack := gen.game.RackFor(gen.game.NextPlayer())
			if oppRack != nil {
				endgameBonus := 2 * float64(oppRack.ScoreOn(gen.letterDistribution))
				equity += endgameBonus
			}
		}
	}

	// Debug logging for row 13 horizontal anchors
	if gen.curRowIdx == 13 && !gen.vertical && (gen.curAnchorCol == 10 || gen.curAnchorCol == 11 || gen.curAnchorCol == 12 || gen.curAnchorCol == 13) {
		fmt.Fprintf(os.Stderr, "[SHADOW-EQUITY] row=13 col=%d: tilesPlayed=%d totalScore=%d (main=%d*%d + perp=%d + unres=%d) leaveSize=%d bestLeave=%.2f equity=%.2f\n",
			gen.curAnchorCol, gen.shadowTilesPlayed, totalScore,
			gen.shadowMainwordRestrictedScore, gen.shadowWordMultiplier,
			gen.shadowPerpAdditionalScore, unrestrictedScore,
			leaveSize, gen.bestLeaves[leaveSize], equity)
	}

	if equity > gen.highestShadowEquity {
		gen.highestShadowEquity = equity
	}
	if totalScore > gen.highestShadowScore {
		gen.highestShadowScore = totalScore
	}
	if gen.shadowTilesPlayed > gen.maxShadowTilesPlayed {
		gen.maxShadowTilesPlayed = gen.shadowTilesPlayed
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
		lsm := score * letterMul
		gen.shadowMainwordRestrictedScore += lsm

		// Debug for row 13
		if gen.curRowIdx == 13 && !gen.vertical {
			fmt.Fprintf(os.Stderr, "[SHADOW-RESTRICT] row=13 col=%d restricted to letter=%d score=%d letterMul=%d added=%d\n",
				col, ml, score, letterMul, lsm)
		}

		// Add tile contribution to perpendicular score if there's a cross word
		// (cross_score * word_mul is already in shadowPerpAdditionalScore from shadowStartNonplaythrough)
		// Magpie: gen->shadow_perpendicular_additional_score += (lsm * this_word_multiplier) & (-(int)is_cross_word)
		if crossScore > 0 {
			gen.shadowPerpAdditionalScore += lsm * wordMul
		}

		// Remove this tile from descending scores (shift down)
		gen.removeTileFromDescending(score)

		return true
	}
	return false
}

// insertUnrestrictedMultiplier adds multiplier info in sorted order.
// crossScore > 0 indicates there's a perpendicular word at this square.
// The effective multiplier formula (matching Magpie) is:
//   effective_mul = shadow_word_mul * letter_mul + letter_mul * word_mul * is_cross_word
// where is_cross_word = 1 if crossScore > 0, else 0.
func (gen *GordonGenerator) insertUnrestrictedMultiplier(letterMul, wordMul int, col int, crossScore int) {
	// Compute cross-word contribution: letter_mul * word_mul if there's a cross word, else 0
	isCrossWord := 0
	if crossScore > 0 {
		isCrossWord = 1
	}
	crossWordContribution := letterMul * wordMul * isCrossWord

	// Effective multiplier = main word contribution + cross word contribution
	// Main word contribution = shadow_word_mul * letter_mul (applied via shadowWordMultiplier)
	// But we need to account for THIS square's word_mul too
	effMul := uint16(letterMul*gen.shadowWordMultiplier*wordMul + crossWordContribution)

	// Insert in descending order
	insertIdx := gen.numUnrestrictedMuls
	for insertIdx > 0 && gen.descendingEffLetterMuls[insertIdx-1] < effMul {
		gen.descendingEffLetterMuls[insertIdx] = gen.descendingEffLetterMuls[insertIdx-1]
		gen.descendingCrossWordMuls[insertIdx] = gen.descendingCrossWordMuls[insertIdx-1]
		gen.descendingLetterMuls[insertIdx] = gen.descendingLetterMuls[insertIdx-1]
		insertIdx--
	}
	gen.descendingEffLetterMuls[insertIdx] = effMul
	// Store cross-word contribution (not just wordMul) for recalculation
	gen.descendingCrossWordMuls[insertIdx] = uint16(crossWordContribution<<8) | uint16(col&0xff)
	gen.descendingLetterMuls[insertIdx] = uint8(letterMul)
	gen.numUnrestrictedMuls++
}

// maybeRecalculateEffectiveMultipliers recalculates effective letter multipliers
// when the word multiplier has changed since they were last computed.
// Formula: effective_mul = letter_mul * shadow_word_mul + cross_word_contribution
// where cross_word_contribution = letter_mul * word_mul * is_cross_word (pre-computed and stored)
func (gen *GordonGenerator) maybeRecalculateEffectiveMultipliers() {
	if gen.lastWordMultiplier == gen.shadowWordMultiplier {
		return
	}
	gen.lastWordMultiplier = gen.shadowWordMultiplier

	// Recalculate and re-sort effective letter multipliers
	// cross_word_contribution is stored in descendingCrossWordMuls (already includes letterMul * wordMul * isCrossWord)
	for i := 0; i < gen.numUnrestrictedMuls; i++ {
		letterMul := int(gen.descendingLetterMuls[i])
		crossWordContribution := int(gen.descendingCrossWordMuls[i] >> 8)
		// effective_mul = letter_mul * shadow_word_mul + cross_word_contribution
		newEffMul := uint16(letterMul*gen.shadowWordMultiplier + crossWordContribution)
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

		fmt.Fprintf(os.Stderr, "[SHADOW] Processing anchor row=%d col=%d vert=%v maxEq=%.2f maxScore=%d winner=%s winnerEq=%.2f\n",
			anchor.Row, anchor.Col, anchor.Vertical, anchor.HighestPossibleEquity, anchor.HighestPossibleScore,
			gen.winner.ShortDescription(), gen.winner.Equity())

		// Early termination: if best found move has equity >= max possible from
		// this anchor, we can skip all remaining anchors (they have lower max equity)
		if !gen.winner.IsEmpty() && gen.winner.Equity() >= anchor.HighestPossibleEquity {
			fmt.Fprintf(os.Stderr, "[SHADOW] PRUNING: winnerEq=%.2f >= anchorMaxEq=%.2f\n",
				gen.winner.Equity(), anchor.HighestPossibleEquity)
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

		fmt.Fprintf(os.Stderr, "[SHADOW] After gen: winner=%s winnerEq=%.2f\n",
			gen.winner.ShortDescription(), gen.winner.Equity())
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
