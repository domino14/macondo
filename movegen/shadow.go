// Package movegen - shadow.go implements the shadow play algorithm for
// best-first move finding. The algorithm was originally developed in wolges
// (https://github.com/andy-k/wolges/blob/main/details.txt) and ported to
// macondo from magpie's implementation.
//
// Shadow computes an upper bound on the score achievable from each anchor
// square, then uses a max-heap to process anchors in descending order. This
// allows early termination once we can prove no remaining anchor can beat
// the best move found so far.
package movegen

import (
	"fmt"
	"math"
	"sort"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tinymove"
)

const (
	maxRackSize       = 7
	initialLastAnchor = 100 // sentinel larger than any board dim
)

// Anchor represents a board anchor with its shadow-computed upper bound.
type Anchor struct {
	HighestPossibleEquity float64
	HighestPossibleScore  int
	Row                   int
	Col                   int
	LastAnchorCol         int
	Dir                   board.BoardDirection
}

// AnchorHeap is a max-heap of anchors ordered by highestPossibleEquity.
type AnchorHeap struct {
	anchors []Anchor
}

func (h *AnchorHeap) reset() {
	h.anchors = h.anchors[:0]
}

func (h *AnchorHeap) addUnheaped(row, col, lastAnchorCol int, dir board.BoardDirection,
	equity float64, score int) {
	h.anchors = append(h.anchors, Anchor{
		HighestPossibleEquity: equity,
		HighestPossibleScore:  score,
		Row:                   row,
		Col:                   col,
		LastAnchorCol:         lastAnchorCol,
		Dir:                   dir,
	})
}

func (h *AnchorHeap) heapifyAll() {
	n := len(h.anchors)
	for i := n/2 - 1; i >= 0; i-- {
		h.heapifyDown(i)
	}
}

func (h *AnchorHeap) heapifyDown(parent int) {
	n := len(h.anchors)
	for {
		best := parent
		left := parent*2 + 1
		right := parent*2 + 2
		if left < n && h.anchors[left].HighestPossibleEquity > h.anchors[best].HighestPossibleEquity {
			best = left
		}
		if right < n && h.anchors[right].HighestPossibleEquity > h.anchors[best].HighestPossibleEquity {
			best = right
		}
		if best == parent {
			break
		}
		h.anchors[parent], h.anchors[best] = h.anchors[best], h.anchors[parent]
		parent = best
	}
}

func (h *AnchorHeap) extractMax() Anchor {
	max := h.anchors[0]
	n := len(h.anchors) - 1
	h.anchors[0] = h.anchors[n]
	h.anchors = h.anchors[:n]
	if n > 0 {
		h.heapifyDown(0)
	}
	return max
}

func (h *AnchorHeap) len() int {
	return len(h.anchors)
}

// shadowState holds all the mutable state used during shadow computation.
// It's embedded in GordonGenerator to avoid allocations.
type shadowState struct {
	anchorHeap AnchorHeap

	// Current shadow play state
	currentLeftCol  int
	currentRightCol int

	// Extension sets for the current anchor
	anchorLeftExtSet  uint64
	anchorRightExtSet uint64

	// Scoring accumulators
	shadowMainwordRestrictedScore int
	shadowPerpAdditionalScore     int
	shadowWordMultiplier          int

	// Highest values found for current anchor
	highestShadowEquity float64
	highestShadowScore  int
	maxTilesToPlay      int

	// Descending tile scores for the rack
	fullRackDescTileScores [maxRackSize]int
	descTileScores         [maxRackSize]int
	descTileScoresCopy     [maxRackSize]int

	// Unrestricted multiplier tracking. When a tile placement is NOT
	// restricted to a single letter (multiple possibilities), we track
	// the multipliers and compute the max via inner product with
	// descending tile scores.
	numUnrestrictedMuls int
	lastWordMultiplier  int

	// Effective letter multipliers in descending order.
	// effective = wordMul*letterMul + crossWordMul
	descEffLetterMuls     [maxRackSize]int
	descEffLetterMulsCopy [maxRackSize]int

	// Cross-word multipliers with column tracking for recalculation.
	descCrossWordMuls     [maxRackSize]crossWordMul
	descCrossWordMulsCopy [maxRackSize]crossWordMul

	// Rack state for shadow
	shadowRack     [tilemapping.MaxAlphabetSize + 1]int
	shadowRackCopy [tilemapping.MaxAlphabetSize + 1]int
	rackCrossSet   uint64 // bitmask of tiles on rack
	numLettersOnRack int

	// Whether shadow is enabled for this generation
	shadowEnabled bool
}

type crossWordMul struct {
	multiplier int
	col        int
}

// rackPossible returns the set of letters that the rack can place on a square
// given its cross-set. If the rack has a blank, any letter in the cross-set
// is possible (since blank can represent it).
func (s *shadowState) rackPossible(crossSet uint64) uint64 {
	possible := crossSet & s.rackCrossSet
	if s.shadowRack[0] > 0 {
		possible |= crossSet // blank can represent any letter
	}
	return possible
}

// initShadow sets up shadow state for a generation pass.
func (gen *GordonGenerator) initShadow(rack *tilemapping.Rack) {
	gen.shadow.anchorHeap.reset()

	// Build rack cross set and descending tile scores
	gen.shadow.rackCrossSet = 0
	gen.shadow.numLettersOnRack = int(rack.NumTiles())

	var scores []int
	for ml := tilemapping.MachineLetter(0); int(ml) < len(rack.LetArr); ml++ {
		gen.shadow.shadowRack[ml] = rack.LetArr[ml]
		if rack.LetArr[ml] > 0 {
			gen.shadow.rackCrossSet |= uint64(1) << ml
			s := gen.letterDistribution.Score(ml)
			for j := 0; j < rack.LetArr[ml]; j++ {
				scores = append(scores, s)
			}
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(scores)))
	for i := range gen.shadow.fullRackDescTileScores {
		if i < len(scores) {
			gen.shadow.fullRackDescTileScores[i] = scores[i]
		} else {
			gen.shadow.fullRackDescTileScores[i] = 0
		}
	}
}

// shadowPlayForAnchor computes the shadow (upper bound score) for a given anchor.
// The board is already in the correct orientation (transposed for vertical).
// csDir is the cross-set direction to check.
func (gen *GordonGenerator) shadowPlayForAnchor(row, col, lastAnchorCol int, dir board.BoardDirection) {
	s := &gen.shadow
	sqIdx := gen.board.GetSqIdx(row, col)

	// When generating horizontal moves, check vertical cross-sets and vice versa
	var csDir board.BoardDirection
	if dir == board.HorizontalDirection {
		csDir = board.VerticalDirection
	} else {
		csDir = board.HorizontalDirection
	}

	// Set up state
	s.currentLeftCol = col
	s.currentRightCol = col
	s.anchorLeftExtSet = gen.board.GetLeftExtSetIdx(sqIdx, csDir)
	s.anchorRightExtSet = gen.board.GetRightExtSetIdx(sqIdx, csDir)

	if s.anchorLeftExtSet|s.anchorRightExtSet == 0 {
		return
	}

	// Reset multiplier tracking
	s.numUnrestrictedMuls = 0
	for i := range s.descEffLetterMuls {
		s.descEffLetterMuls[i] = 0
	}
	s.lastWordMultiplier = 1

	// Reset tile scores
	s.descTileScores = s.fullRackDescTileScores

	// Reset scoring
	s.shadowMainwordRestrictedScore = 0
	s.shadowPerpAdditionalScore = 0
	s.shadowWordMultiplier = 1

	// Reset shadow results
	s.highestShadowEquity = 0
	s.highestShadowScore = 0
	s.maxTilesToPlay = 0

	gen.tilesPlayed = 0
	gen.curAnchorCol = col

	// Copy rack for restoration
	copy(s.shadowRackCopy[:], s.shadowRack[:])
	origRackCrossSet := s.rackCrossSet

	curLetter := gen.board.GetLetter(row, col)
	if curLetter == 0 {
		gen.shadowStartNonplaythrough(row, dir, csDir)
	} else {
		gen.shadowStartPlaythrough(row, col, lastAnchorCol, curLetter, dir, csDir)
	}

	// Restore rack
	copy(s.shadowRack[:], s.shadowRackCopy[:])
	s.rackCrossSet = origRackCrossSet

	if s.maxTilesToPlay == 0 {
		return
	}

	// Use score as equity for KWG-only shadow (no leave values yet)
	s.anchorHeap.addUnheaped(row, col, lastAnchorCol, dir,
		float64(s.highestShadowScore), s.highestShadowScore)
}

// shadowStartNonplaythrough handles shadow for an anchor on an empty square.
func (gen *GordonGenerator) shadowStartNonplaythrough(row int, dir, csDir board.BoardDirection) {
	s := &gen.shadow
	col := s.currentLeftCol
	sqIdx := gen.board.GetSqIdx(row, col)

	crossSet := uint64(gen.board.GetCrossSetIdx(sqIdx, csDir))
	possibleLetters := s.rackPossible(crossSet)
	if possibleLetters == 0 {
		return
	}

	lm := gen.board.GetLetterMultiplier(sqIdx)
	wm := gen.board.GetWordMultiplier(sqIdx)
	cs := gen.board.GetCrossScoreIdx(sqIdx, csDir)
	isCrossWord := cs > 0 || crossSet != board.TrivialCrossSet

	// Set word multiplier to 0 initially since we're recording a single tile
	s.shadowWordMultiplier = 0
	s.shadowPerpAdditionalScore = cs * wm

	if !gen.shadowTryRestrict(possibleLetters, lm, wm, col, isCrossWord) {
		gen.shadowInsertUnrestricted(col, lm, wm, isCrossWord)
	}
	gen.tilesPlayed++

	// Record single-tile play only in horizontal (avoid vertical duplicates)
	if dir == board.HorizontalDirection {
		gen.shadowRecord()
	}

	s.shadowWordMultiplier = wm
	gen.shadowMaybeRecalcEffMuls()

	// Now try extending left
	gen.shadowNonplaythroughPlayLeft(row, dir == board.HorizontalDirection, csDir)
}

// shadowStartPlaythrough handles shadow for an anchor on a filled square.
func (gen *GordonGenerator) shadowStartPlaythrough(row, col, lastAnchorCol int,
	curLetter tilemapping.MachineLetter, dir, csDir board.BoardDirection) {
	s := &gen.shadow

	// Traverse the full length of existing tiles leftward
	for {
		s.shadowMainwordRestrictedScore += gen.letterDistribution.Score(curLetter)
		if s.currentLeftCol == 0 || s.currentLeftCol == lastAnchorCol+1 {
			break
		}
		s.currentLeftCol--
		curLetter = gen.board.GetLetter(row, s.currentLeftCol)
		if curLetter == 0 {
			s.currentLeftCol++
			break
		}
	}

	gen.shadowPlaythroughPlayLeft(row, dir == board.HorizontalDirection, csDir)
}

// shadowNonplaythroughPlayLeft extends shadow leftward from a nonplaythrough anchor.
func (gen *GordonGenerator) shadowNonplaythroughPlayLeft(row int, isUnique bool, csDir board.BoardDirection) {
	s := &gen.shadow

	for {
		// Try extending right first
		possibleRight := s.rackPossible(s.anchorRightExtSet)
		if possibleRight != 0 {
			gen.shadowPlayRight(row, isUnique, csDir)
		}
		s.anchorRightExtSet = board.TrivialCrossSet

		if s.currentLeftCol == 0 ||
			s.currentLeftCol == gen.lastAnchorCol+1 ||
			gen.tilesPlayed >= s.numLettersOnRack {
			return
		}

		possibleLeft := s.rackPossible(s.anchorLeftExtSet)
		if possibleLeft == 0 {
			return
		}
		s.anchorLeftExtSet = board.TrivialCrossSet

		s.currentLeftCol--
		gen.tilesPlayed++

		sqIdx := gen.board.GetSqIdx(row, s.currentLeftCol)
		crossSet := uint64(gen.board.GetCrossSetIdx(sqIdx, csDir))
		possibleHere := s.rackPossible(crossSet)
		if possibleHere == 0 {
			return
		}
		lm := gen.board.GetLetterMultiplier(sqIdx)
		wm := gen.board.GetWordMultiplier(sqIdx)
		cs := gen.board.GetCrossScoreIdx(sqIdx, csDir)
		isCrossWord := cs > 0 || crossSet != board.TrivialCrossSet

		s.shadowPerpAdditionalScore += cs * wm
		s.shadowWordMultiplier *= wm

		if !gen.shadowTryRestrict(possibleHere, lm, wm, s.currentLeftCol, isCrossWord) {
			gen.shadowInsertUnrestricted(s.currentLeftCol, lm, wm, isCrossWord)
		}

		gen.shadowRecord()
	}
}

// shadowPlaythroughPlayLeft extends shadow leftward from a playthrough anchor.
func (gen *GordonGenerator) shadowPlaythroughPlayLeft(row int, isUnique bool, csDir board.BoardDirection) {
	s := &gen.shadow

	for {
		// Try extending right
		possibleRight := s.rackPossible(s.anchorRightExtSet)
		if possibleRight != 0 {
			gen.shadowPlayRight(row, isUnique, csDir)
		}
		s.anchorRightExtSet = board.TrivialCrossSet

		possibleLeft := s.rackPossible(s.anchorLeftExtSet)
		s.anchorLeftExtSet = board.TrivialCrossSet

		if s.currentLeftCol == 0 ||
			s.currentLeftCol == gen.lastAnchorCol+1 ||
			gen.tilesPlayed >= s.numLettersOnRack {
			break
		}
		if possibleLeft == 0 {
			break
		}

		s.currentLeftCol--
		gen.tilesPlayed++

		sqIdx := gen.board.GetSqIdx(row, s.currentLeftCol)
		crossSet := uint64(gen.board.GetCrossSetIdx(sqIdx, csDir))

		possibleHere := s.rackPossible(crossSet)
		if possibleHere == 0 {
			break
		}

		lm := gen.board.GetLetterMultiplier(sqIdx)
		wm := gen.board.GetWordMultiplier(sqIdx)
		cs := gen.board.GetCrossScoreIdx(sqIdx, csDir)
		isCrossWord := cs > 0 || crossSet != board.TrivialCrossSet

		s.shadowPerpAdditionalScore += cs * wm
		s.shadowWordMultiplier *= wm

		if !gen.shadowTryRestrict(possibleHere, lm, wm, s.currentLeftCol, isCrossWord) {
			gen.shadowInsertUnrestricted(s.currentLeftCol, lm, wm, isCrossWord)
		}

		if crossSet == board.TrivialCrossSet {
			isUnique = true
		}

		if gen.tilesPlayed > 0 && (isUnique || gen.tilesPlayed > 1) {
			gen.shadowRecord()
		}
	}
}

// shadowPlayRight extends the shadow rightward from the current position.
// Matches magpie's shadow_play_right structure: the main loop iterates over
// EMPTY squares (placing tiles), and after each placement, scans past any
// consecutive playthrough tiles (adding their scores without using rack tiles).
func (gen *GordonGenerator) shadowPlayRight(row int, isUnique bool, csDir board.BoardDirection) {
	s := &gen.shadow

	// Save state for restoration
	origMainRestricted := s.shadowMainwordRestrictedScore
	origPerpScore := s.shadowPerpAdditionalScore
	origWordMul := s.shadowWordMultiplier
	origTilesPlayed := gen.tilesPlayed
	origRightCol := s.currentRightCol
	origNumUnrestricted := s.numUnrestrictedMuls

	s.descTileScoresCopy = s.descTileScores
	copy(s.descCrossWordMulsCopy[:], s.descCrossWordMuls[:])
	copy(s.descEffLetterMulsCopy[:], s.descEffLetterMuls[:])
	origRackCrossSet := s.rackCrossSet
	var rackCopy [tilemapping.MaxAlphabetSize + 1]int
	copy(rackCopy[:], s.shadowRack[:])
	restrictedAny := false
	changedMuls := false

	dim := gen.boardDim

	for s.currentRightCol < dim-1 && gen.tilesPlayed < s.numLettersOnRack {
		s.currentRightCol++
		gen.tilesPlayed++

		sqIdx := gen.board.GetSqIdx(row, s.currentRightCol)
		crossSet := uint64(gen.board.GetCrossSetIdx(sqIdx, csDir))
		leftExtHere := gen.board.GetLeftExtSetIdx(sqIdx, csDir)

		// Possible letters: must be in cross-set, left extension set,
		// anchor right extension set, and on the rack (or blank).
		possibleHere := s.rackPossible(crossSet & leftExtHere & s.anchorRightExtSet)
		// After first use, right extension set becomes trivial
		s.anchorRightExtSet = board.TrivialCrossSet

		if possibleHere == 0 {
			break
		}

		lm := gen.board.GetLetterMultiplier(sqIdx)
		wm := gen.board.GetWordMultiplier(sqIdx)
		cs := gen.board.GetCrossScoreIdx(sqIdx, csDir)
		isCrossWord := cs > 0 || crossSet != board.TrivialCrossSet

		s.shadowPerpAdditionalScore += cs * wm
		s.shadowWordMultiplier *= wm

		if gen.shadowTryRestrict(possibleHere, lm, wm, s.currentRightCol, isCrossWord) {
			restrictedAny = true
		} else {
			gen.shadowInsertUnrestricted(s.currentRightCol, lm, wm, isCrossWord)
			changedMuls = true
		}

		if crossSet == board.TrivialCrossSet {
			isUnique = true
		}

		// Scan past consecutive playthrough tiles to the right
		for s.currentRightCol+1 < dim {
			nextLetter := gen.board.GetLetter(row, s.currentRightCol+1)
			if nextLetter == 0 {
				break
			}
			s.shadowMainwordRestrictedScore += gen.letterDistribution.Score(nextLetter)
			s.currentRightCol++
		}

		if gen.tilesPlayed > 0 && (isUnique || gen.tilesPlayed > 1) {
			gen.shadowMaybeRecalcEffMuls()
			gen.shadowRecord()
		}
	}

	// Restore all state
	s.shadowMainwordRestrictedScore = origMainRestricted
	s.shadowPerpAdditionalScore = origPerpScore
	s.shadowWordMultiplier = origWordMul

	if restrictedAny {
		copy(s.shadowRack[:], rackCopy[:])
		s.rackCrossSet = origRackCrossSet
		s.descTileScores = s.descTileScoresCopy
	}
	if changedMuls {
		s.numUnrestrictedMuls = origNumUnrestricted
		copy(s.descCrossWordMuls[:], s.descCrossWordMulsCopy[:])
		copy(s.descEffLetterMuls[:], s.descEffLetterMulsCopy[:])
	}

	s.currentRightCol = origRightCol
	gen.tilesPlayed = origTilesPlayed
	gen.shadowMaybeRecalcEffMuls()
}

// shadowRecord computes the upper bound score for the current shadow state
// and updates the highest shadow score if it's better.
func (gen *GordonGenerator) shadowRecord() {
	s := &gen.shadow

	// Compute unrestricted tiles score: inner product of descending tile scores
	// and descending effective letter multipliers
	tilesPlayedScore := 0
	for i := 0; i < maxRackSize; i++ {
		tilesPlayedScore += s.descTileScores[i] * s.descEffLetterMuls[i]
	}

	bingoBonus := 0
	if gen.tilesPlayed == maxRackSize {
		bingoBonus = 50
	}

	score := tilesPlayedScore +
		s.shadowMainwordRestrictedScore*s.shadowWordMultiplier +
		s.shadowPerpAdditionalScore +
		bingoBonus

	if score > s.highestShadowScore {
		s.highestShadowScore = score
	}
	if gen.tilesPlayed > s.maxTilesToPlay {
		s.maxTilesToPlay = gen.tilesPlayed
	}
}

// shadowTryRestrict attempts to restrict a tile placement to a single letter.
// Returns true if exactly one letter is possible (restricted), false otherwise.
func (gen *GordonGenerator) shadowTryRestrict(possibleLetters uint64, letterMul, wordMul, col int, isCrossWord bool) bool {
	s := &gen.shadow

	// Check if exactly one bit is set
	if possibleLetters == 0 || possibleLetters&(possibleLetters-1) != 0 {
		return false
	}

	// Find the single set bit
	ml := tilemapping.MachineLetter(0)
	tmp := possibleLetters
	for tmp&1 == 0 {
		tmp >>= 1
		ml++
	}

	// Remove from rack
	s.shadowRack[ml]--
	if s.shadowRack[ml] == 0 {
		s.rackCrossSet &= ^possibleLetters
	}

	tileScore := gen.letterDistribution.Score(ml)
	gen.removeFromDescTileScores(tileScore)

	lsm := tileScore * letterMul
	s.shadowMainwordRestrictedScore += lsm

	if isCrossWord {
		s.shadowPerpAdditionalScore += lsm * wordMul
	}

	return true
}

// removeFromDescTileScores finds and removes a score from the descending list.
func (gen *GordonGenerator) removeFromDescTileScores(score int) {
	s := &gen.shadow
	total := 0
	for i := 0; i < maxRackSize; i++ {
		if s.descTileScores[i] > 0 {
			total++
		}
	}
	newSize := total - 1
	for i := newSize; i >= 0; i-- {
		if s.descTileScores[i] == score {
			for j := i; j < newSize; j++ {
				s.descTileScores[j] = s.descTileScores[j+1]
			}
			s.descTileScores[newSize] = 0
			return
		}
	}
}

// shadowInsertUnrestricted adds an unrestricted multiplier entry.
func (gen *GordonGenerator) shadowInsertUnrestricted(col, letterMul, wordMul int, isCrossWord bool) {
	s := &gen.shadow
	gen.shadowMaybeRecalcEffMuls()

	effectiveCrossWordMul := 0
	if isCrossWord {
		effectiveCrossWordMul = letterMul * wordMul
	}

	// Insert cross-word multiplier in descending order
	idx := s.numUnrestrictedMuls
	for idx > 0 && s.descCrossWordMuls[idx-1].multiplier < effectiveCrossWordMul {
		s.descCrossWordMuls[idx] = s.descCrossWordMuls[idx-1]
		idx--
	}
	s.descCrossWordMuls[idx] = crossWordMul{multiplier: effectiveCrossWordMul, col: col}

	// Insert effective letter multiplier in descending order
	mainWordMul := s.shadowWordMultiplier * letterMul
	effLetterMul := mainWordMul + effectiveCrossWordMul
	idx = s.numUnrestrictedMuls
	for idx > 0 && s.descEffLetterMuls[idx-1] < effLetterMul {
		s.descEffLetterMuls[idx] = s.descEffLetterMuls[idx-1]
		idx--
	}
	s.descEffLetterMuls[idx] = effLetterMul

	s.numUnrestrictedMuls++
}

// shadowMaybeRecalcEffMuls recalculates effective letter multipliers when
// the word multiplier changes.
func (gen *GordonGenerator) shadowMaybeRecalcEffMuls() {
	s := &gen.shadow
	if s.lastWordMultiplier == s.shadowWordMultiplier {
		return
	}
	s.lastWordMultiplier = s.shadowWordMultiplier

	origNum := s.numUnrestrictedMuls
	s.numUnrestrictedMuls = 0

	for i := 0; i < origNum; i++ {
		xwMul := s.descCrossWordMuls[i].multiplier
		col := s.descCrossWordMuls[i].col

		sqIdx := gen.board.GetSqIdx(gen.curRowIdx, col)
		letterMul := gen.board.GetLetterMultiplier(sqIdx)
		effLetterMul := s.shadowWordMultiplier*letterMul + xwMul

		// Insert in descending order
		idx := s.numUnrestrictedMuls
		for idx > 0 && s.descEffLetterMuls[idx-1] < effLetterMul {
			s.descEffLetterMuls[idx] = s.descEffLetterMuls[idx-1]
			idx--
		}
		s.descEffLetterMuls[idx] = effLetterMul
		s.numUnrestrictedMuls++
	}
}

// genShadow computes shadow values for all anchors in both directions.
// It uses Transpose() to match the existing movegen convention: horizontal
// moves operate on the untransposed board, vertical moves on the transposed board.
// Anchors are stored in the coordinates of their respective orientation.
func (gen *GordonGenerator) genShadow(rack *tilemapping.Rack) {
	gen.initShadow(rack)
	dim := gen.boardDim

	// Horizontal
	gen.vertical = false
	for row := 0; row < dim; row++ {
		gen.curRowIdx = row
		gen.lastAnchorCol = initialLastAnchor
		for col := 0; col < dim; col++ {
			if gen.board.IsAnchor(row, col, board.HorizontalDirection) {
				gen.shadowPlayForAnchor(row, col, gen.lastAnchorCol, board.HorizontalDirection)
				gen.lastAnchorCol = col
				if gen.board.HasLetter(row, col) {
					gen.lastAnchorCol++
				}
			}
		}
	}

	// Vertical - transpose board so we can use the same row/col iteration
	gen.board.Transpose()
	gen.vertical = true
	for row := 0; row < dim; row++ {
		gen.curRowIdx = row
		gen.lastAnchorCol = initialLastAnchor
		for col := 0; col < dim; col++ {
			if gen.board.IsAnchor(row, col, board.VerticalDirection) {
				gen.shadowPlayForAnchor(row, col, gen.lastAnchorCol, board.VerticalDirection)
				gen.lastAnchorCol = col
				if gen.board.HasLetter(row, col) {
					gen.lastAnchorCol++
				}
			}
		}
	}
	gen.board.Transpose()

	gen.shadow.anchorHeap.heapifyAll()
}

// genRecordScoringPlaysFromAnchors processes anchors in best-first order,
// generating actual moves via recursive generation. Stops when no remaining
// anchor can beat the best move found so far.
func (gen *GordonGenerator) genRecordScoringPlaysFromAnchors(rack *tilemapping.Rack) {
	gen.tilesPlayed = 0
	gd := gen.gaddag

	// Track current orientation to minimize transpose calls.
	// Anchors were stored in the coordinates of their orientation
	// (horizontal = untransposed, vertical = transposed).
	currentlyTransposed := false

	for gen.shadow.anchorHeap.len() > 0 {
		if gen.quitEarly {
			break
		}

		anchor := gen.shadow.anchorHeap.extractMax()

		// Check if we can stop: no remaining anchor can beat our best
		if gen.winner != nil && !gen.winner.IsEmpty() {
			bestScore := float64(gen.winner.Score())
			if len(gen.equityCalculators) > 0 {
				bestScore = gen.winner.Equity()
			}
			if anchor.HighestPossibleEquity < bestScore {
				if currentlyTransposed {
					gen.board.Transpose()
				}
				return
			}
		}

		needTransposed := anchor.Dir == board.VerticalDirection
		if needTransposed != currentlyTransposed {
			gen.board.Transpose()
			currentlyTransposed = needTransposed
		}
		gen.vertical = needTransposed

		// Anchor row/col are already in the correct coordinate system
		gen.curRowIdx = anchor.Row
		gen.curAnchorCol = anchor.Col
		gen.lastAnchorCol = anchor.LastAnchorCol

		gen.recursiveGen(anchor.Col, rack, gd.GetRootNodeIndex(),
			anchor.Col, anchor.Col, !gen.vertical, 0, 0, 1)
	}

	if currentlyTransposed {
		gen.board.Transpose()
	}
}

// ShadowAnchorCount returns the number of anchors in the shadow heap (for testing).
func (gen *GordonGenerator) ShadowAnchorCount() int {
	return len(gen.shadow.anchorHeap.anchors)
}

// RunShadowOnly runs the shadow pass and returns the anchors sorted by
// descending highest possible score. For testing.
func (gen *GordonGenerator) RunShadowOnly(rack *tilemapping.Rack) []Anchor {
	gen.genShadow(rack)
	// Extract and sort by descending score
	result := make([]Anchor, len(gen.shadow.anchorHeap.anchors))
	copy(result, gen.shadow.anchorHeap.anchors)
	sort.Slice(result, func(i, j int) bool {
		return result[i].HighestPossibleScore > result[j].HighestPossibleScore
	})
	// Export fields
	exported := make([]Anchor, len(result))
	for i, a := range result {
		exported[i] = Anchor{
			HighestPossibleEquity: a.HighestPossibleEquity,
			HighestPossibleScore:  a.HighestPossibleScore,
			Row:                   a.Row,
			Col:                   a.Col,
			LastAnchorCol:         a.LastAnchorCol,
			Dir:                   a.Dir,
		}
	}
	return exported
}

// PrintShadowAnchors runs the shadow pass and prints all anchors with their
// upper bounds. For debugging.
func (gen *GordonGenerator) PrintShadowAnchors(rack *tilemapping.Rack) {
	gen.genShadow(rack)
	fmt.Printf("Shadow anchors (%d):\n", gen.shadow.anchorHeap.len())
	// Print in heap order (not sorted, but max is at top)
	for _, a := range gen.shadow.anchorHeap.anchors {
		dir := "H"
		if a.Dir == board.VerticalDirection {
			dir = "V"
		}
		fmt.Printf("  (%d,%d) %s score=%d equity=%.1f lastAnchor=%d\n",
			a.Row, a.Col, dir, a.HighestPossibleScore, a.HighestPossibleEquity, a.LastAnchorCol)
	}
}

// GenAllWithShadow generates all moves using shadow for best-first ordering.
// Only useful when we need the best move (not all moves).
func (gen *GordonGenerator) GenAllWithShadow(rack *tilemapping.Rack, addExchange bool) []*move.Move {
	gen.winner.SetEmpty()
	gen.quitEarly = false
	gen.plays = gen.plays[:0]
	for i := range gen.topNPlays {
		gen.topNPlays[i].SetEquity(math.Inf(-1))
		gen.topNPlays[i].SetEmpty()
	}

	ptr := SmallPlaySlicePool.Get().(*[]tinymove.SmallMove)
	gen.smallPlays = *ptr
	gen.smallPlays = gen.smallPlays[0:0]

	// Run shadow to rank anchors
	gen.genShadow(rack)

	// Generate moves from anchors in best-first order
	gen.genRecordScoringPlaysFromAnchors(rack)

	// Pass handling
	if (len(gen.plays) == 0 && len(gen.smallPlays) == 0) || gen.genPass {
		gen.playRecorder(gen, rack, 0, 0, move.MoveTypePass, 0)
	} else if len(gen.plays) > 1 {
		switch gen.sortingParameter {
		case SortByScore:
			sort.Slice(gen.plays, func(i, j int) bool {
				return gen.plays[i].Score() > gen.plays[j].Score()
			})
		case SortByNone:
			break
		}
	}

	if addExchange {
		gen.generateExchangeMoves(rack, 0, 0)
	}
	*ptr = gen.smallPlays

	if gen.maxTopMovesSize != 0 {
		delfrom := -1
		for i := 0; i < gen.maxTopMovesSize; i++ {
			if gen.topNPlays[i].IsEmpty() {
				delfrom = i
				break
			}
		}
		if delfrom != -1 {
			return gen.topNPlays[:delfrom]
		}
		return gen.topNPlays
	}

	return gen.plays
}
