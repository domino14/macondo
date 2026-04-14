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
	"math"
	"slices"
	"sort"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/wmp"
)


const (
	// Endgame equity constants (see also equity/endgame.go).
	endgameNonOutplayLeavePenaltyMultiplier = 2
	endgameNonOutplayConstantPenalty        = 10.0
)

// Anchor represents a board anchor with its shadow-computed upper bound.
//
// The WMP-specific fields (TilesToPlay, PlaythroughBlocks, WordLength,
// LeftmostStartCol, RightmostStartCol) are populated only when WMP is
// active and the anchor came out of the per-(blocks, tiles) WMP anchor
// table. They mirror MAGPIE's Anchor extension fields used by
// wmp_move_gen_add_anchors / wordmap_gen.
//
// Layout-packed: 8 (equity) + 4 (score) + 9 × 1 (uint8 fields) = 21
// bytes, padded to 24 with 8-byte alignment. Down from 80 bytes when
// every field was a Go `int`. Smaller anchors mean cheaper heap
// operations: heapifyDown swaps cost ~3× less and the heap fits more
// entries per cache line. The sentinel value for LastAnchorCol is
// math.MaxUint8 (255) instead of the 100 used previously.
//
// All "small int" fields use uint8 directly (not a typed alias) so
// callers don't need ad-hoc conversions every time they read one.
// Score uses int32 because a 7-tile bingo on triple-triples can
// exceed 255 but is bounded well below int16's max.
type Anchor struct {
	HighestPossibleEquity float64
	HighestPossibleScore  int32
	Row                   uint8
	Col                   uint8
	LastAnchorCol         uint8
	Dir                   board.BoardDirection

	// WMP-only fields. Zero when not produced by the WMP path.
	TilesToPlay       uint8
	PlaythroughBlocks uint8
	WordLength        uint8
	LeftmostStartCol  uint8
	RightmostStartCol uint8
}

// LastAnchorCol uses the existing in-package sentinel value 100
// (greater than any valid 0..14 column on a 15-board), which fits
// in uint8 unchanged.

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
		HighestPossibleScore:  int32(score),
		Row:                   uint8(row),
		Col:                   uint8(col),
		LastAnchorCol:         uint8(lastAnchorCol),
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
	fullRackDescTileScores [game.RackTileLimit]int
	descTileScores         [game.RackTileLimit]int
	descTileScoresCopy     [game.RackTileLimit]int

	// Unrestricted multiplier tracking. When a tile placement is NOT
	// restricted to a single letter (multiple possibilities), we track
	// the multipliers and compute the max via inner product with
	// descending tile scores.
	numUnrestrictedMuls int
	lastWordMultiplier  int

	// Effective letter multipliers in descending order.
	// effective = wordMul*letterMul + crossWordMul
	descEffLetterMuls     [game.RackTileLimit]int
	descEffLetterMulsCopy [game.RackTileLimit]int

	// Cross-word multipliers with column tracking for recalculation.
	descCrossWordMuls     [game.RackTileLimit]crossWordMul
	descCrossWordMulsCopy [game.RackTileLimit]crossWordMul

	// Rack state for shadow
	shadowRack     [tilemapping.MaxAlphabetSize + 1]int
	shadowRackCopy [tilemapping.MaxAlphabetSize + 1]int
	rackCrossSet   uint64 // bitmask of tiles on rack
	numLettersOnRack int

	// Best leave values indexed by number of tiles remaining on rack.
	// bestLeaves[k] = max leave value over all possible leaves of size k.
	// Computed before shadow to enable equity-aware upper bounds.
	bestLeaves [game.RackTileLimit + 1]float64

	// Scratch buffer for enumerateLeaves; avoids heap allocation of a growing slice.
	leaveBuf [game.RackTileLimit]tilemapping.MachineLetter

	// Endgame equity state (bag empty)
	tilesInBag     int
	oppRackScore   int

	// Whether shadow is used for this generator. Automatically true for
	// TopPlayOnly and TopN recorders, false for AllPlays recorders.
	useShadow bool
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

// computeBestLeaves populates bestLeaves[k] with the maximum leave value
// over all possible leaves of size k from the current rack. Also sets
// endgame state (tilesInBag, oppRackScore) for endgame equity adjustments.
// Matches magpie's best_leaves + static_eval_get_shadow_equity.
func (gen *GordonGenerator) computeBestLeaves(rack *tilemapping.Rack) {
	s := &gen.shadow
	for i := range s.bestLeaves {
		s.bestLeaves[i] = math.Inf(-1)
	}

	// Set endgame state
	if gen.game != nil {
		s.tilesInBag = gen.game.Bag().TilesRemaining()
		oppRack := gen.game.RackFor(gen.game.NextPlayer())
		s.oppRackScore = 0
		for ml := tilemapping.MachineLetter(1); int(ml) < len(oppRack.LetArr); ml++ {
			s.oppRackScore += oppRack.LetArr[ml] * gen.letterDistribution.Score(ml)
		}
	} else {
		s.tilesInBag = 100 // assume mid-game if no game reference
		s.oppRackScore = 0
	}

	// Find the leave value calculator from equity calculators.
	var leaveCalc interface{ LeaveValue(tilemapping.MachineWord) float64 }
	for _, calc := range gen.equityCalculators {
		if lc, ok := calc.(interface{ LeaveValue(tilemapping.MachineWord) float64 }); ok {
			leaveCalc = lc
			break
		}
	}
	if leaveCalc == nil {
		// No leave calculator — use 0 for all leave sizes (score-only mode).
		for i := range s.bestLeaves {
			s.bestLeaves[i] = 0
		}
		return
	}

	if s.tilesInBag > 0 {
		gen.enumerateLeaves(rack, leaveCalc, 0, 0)
	} else {
		// Endgame: compute bestLeaves from tile scores directly.
		// bestLeaves[0] = outplay bonus.
		s.bestLeaves[0] = float64(2 * s.oppRackScore)
		// bestLeaves[k] for k>0 = best non-outplay adjustment =
		// -2 * (sum of k lowest-scoring tiles) - 10.
		// Build ascending tile scores from rack.
		var ascScores [game.RackTileLimit]int
		n := 0
		for ml := tilemapping.MachineLetter(0); int(ml) < len(rack.LetArr); ml++ {
			sc := gen.letterDistribution.Score(ml)
			for j := 0; j < rack.LetArr[ml]; j++ {
				ascScores[n] = sc
				n++
			}
		}
		slices.Sort(ascScores[:n])
		lowestSum := 0
		for k := 1; k <= n; k++ {
			lowestSum += ascScores[k-1]
			s.bestLeaves[k] = float64(-endgameNonOutplayLeavePenaltyMultiplier*lowestSum) - endgameNonOutplayConstantPenalty
		}
	}
}

// enumerateLeaves recursively enumerates all sub-multisets of the rack,
// recording the best leave value for each leave size in gen.shadow.bestLeaves.
// depth is the number of letters written into gen.shadow.leaveBuf so far.
func (gen *GordonGenerator) enumerateLeaves(
	rack *tilemapping.Rack,
	leaveCalc interface{ LeaveValue(tilemapping.MachineWord) float64 },
	ml tilemapping.MachineLetter,
	depth int,
) {
	if int(ml) >= len(rack.LetArr) {
		val := leaveCalc.LeaveValue(gen.shadow.leaveBuf[:depth])
		if val > gen.shadow.bestLeaves[depth] {
			gen.shadow.bestLeaves[depth] = val
		}
		return
	}
	count := rack.LetArr[ml]
	if count == 0 {
		gen.enumerateLeaves(rack, leaveCalc, ml+1, depth)
		return
	}
	// Try keeping 0, 1, ..., count copies of this letter in the leave.
	// After each recursive call we append one more copy of ml into leaveBuf
	// so the next iteration's recursive call sees it.
	for k := 0; k <= count; k++ {
		gen.enumerateLeaves(rack, leaveCalc, ml+1, depth+k)
		if k < count {
			gen.shadow.leaveBuf[depth+k] = ml
		}
	}
}

// initShadow sets up shadow state for a generation pass.
func (gen *GordonGenerator) initShadow(rack *tilemapping.Rack) {
	gen.shadow.anchorHeap.reset()

	// Build rack cross set and descending tile scores
	gen.shadow.rackCrossSet = 0
	gen.shadow.numLettersOnRack = int(rack.NumTiles())

	n := 0
	for ml := tilemapping.MachineLetter(0); int(ml) < len(rack.LetArr); ml++ {
		gen.shadow.shadowRack[ml] = rack.LetArr[ml]
		if rack.LetArr[ml] > 0 {
			gen.shadow.rackCrossSet |= uint64(1) << ml
			s := gen.letterDistribution.Score(ml)
			for j := 0; j < rack.LetArr[ml]; j++ {
				gen.shadow.fullRackDescTileScores[n] = s
				n++
			}
		}
	}
	for i := n; i < len(gen.shadow.fullRackDescTileScores); i++ {
		gen.shadow.fullRackDescTileScores[i] = 0
	}
	slices.SortFunc(gen.shadow.fullRackDescTileScores[:n], func(a, b int) int { return b - a })
}

// shadowPlayForAnchor computes the shadow (upper bound score) for a given anchor.
// The board is already in the correct orientation (transposed for vertical).
// csDir is the cross-set direction to check.
func (gen *GordonGenerator) shadowPlayForAnchor(row, col, lastAnchorCol int, dir board.BoardDirection) {
	s := &gen.shadow
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
	// leftExtSet comes from the row cache (loaded once per row in genShadow).
	// rightExtSet is anchor-specific and not cached, so we call the board.
	s.anchorLeftExtSet = uint64(gen.cache.squares[col].leftExtSet)
	sqIdx := gen.board.GetSqIdx(row, col)
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
	s.highestShadowEquity = math.Inf(-1)
	s.highestShadowScore = 0
	s.maxTilesToPlay = 0

	gen.tilesPlayed = 0
	gen.curAnchorCol = col

	// Reset WMP playthrough/anchor scratch state for this anchor.
	// Mirrors wmp_move_gen_reset_playthrough +
	// wmp_move_gen_reset_anchors at the top of shadow_play_for_anchor.
	if gen.wmpMoveGen.IsActive() {
		gen.wmpMoveGen.ResetPlaythrough()
		gen.wmpMoveGen.ResetAnchors()
	}

	// Copy rack for restoration
	copy(s.shadowRackCopy[:], s.shadowRack[:])
	origRackCrossSet := s.rackCrossSet

	curLetter := gen.cache.squares[col].letter
	if curLetter == 0 {
		gen.shadowStartNonplaythrough(dir)
	} else {
		gen.shadowStartPlaythrough(lastAnchorCol, curLetter, dir)
	}

	// Restore rack
	copy(s.shadowRack[:], s.shadowRackCopy[:])
	s.rackCrossSet = origRackCrossSet

	if s.maxTilesToPlay == 0 {
		return
	}

	if gen.wmpMoveGen.IsActive() && gen.wmpRecordSubAnchors {
		// Dump the per-(blocks, tiles) WMP anchors for this square
		// into the heap. Mirrors wmp_move_gen_add_anchors. Each
		// non-empty slot becomes its own Anchor with the
		// WMP-specific extension fields populated. Only useful
		// alongside a wordmap_gen-style move generator that can
		// process them efficiently; the default macondo recursive
		// generator processes each anchor exhaustively, so adding
		// N sub-anchors per square turns into N redundant
		// recursiveGen passes. The non-sub-anchor path below still
		// benefits from the WMP existence gating in shadow_record
		// (tighter highestShadowEquity bound).
		gen.addWMPAnchorsForSquare(row, col, lastAnchorCol, dir)
	} else {
		s.anchorHeap.addUnheaped(row, col, lastAnchorCol, dir,
			s.highestShadowEquity, s.highestShadowScore)
	}
}

// addWMPAnchorsForSquare walks the wmp move gen's dirty anchor list
// (only the per-(blocks, tiles) slots that were touched during the
// just-finished shadow_play_for_anchor pass) and pushes each into
// the shadow anchor heap. Mirrors MAGPIE's wmp_move_gen_add_anchors,
// but uses the dirty-index shortcut so we don't scan all 64 anchor
// slots per square AND don't copy any Anchor structs into a scratch
// buffer first.
func (gen *GordonGenerator) addWMPAnchorsForSquare(row, col, lastAnchorCol int, dir board.BoardDirection) {
	dirty := gen.wmpMoveGen.DirtyAnchorIndices()
	for _, idx := range dirty {
		a := gen.wmpMoveGen.AnchorAt(idx)
		gen.shadow.anchorHeap.anchors = append(gen.shadow.anchorHeap.anchors, Anchor{
			HighestPossibleEquity: a.HighestPossibleEquity,
			HighestPossibleScore:  int32(a.HighestPossibleScore),
			Row:                   uint8(row),
			Col:                   uint8(col),
			LastAnchorCol:         uint8(lastAnchorCol),
			Dir:                   dir,
			TilesToPlay:           uint8(a.TilesToPlay),
			PlaythroughBlocks:     uint8(a.PlaythroughBlocks),
			WordLength:            uint8(a.WordLength),
			LeftmostStartCol:      uint8(a.LeftmostStartCol),
			RightmostStartCol:     uint8(a.RightmostStartCol),
		})
	}
}

// shadowStartNonplaythrough handles shadow for an anchor on an empty square.
func (gen *GordonGenerator) shadowStartNonplaythrough(dir board.BoardDirection) {
	s := &gen.shadow
	col := s.currentLeftCol
	sq := &gen.cache.squares[col]

	crossSet := uint64(sq.crossSet)
	possibleLetters := s.rackPossible(crossSet)
	if possibleLetters == 0 {
		return
	}

	lm := int(sq.letterMul)
	wm := int(sq.wordMul)
	cs := int(sq.crossScore)
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
	gen.shadowNonplaythroughPlayLeft(dir == board.HorizontalDirection)
}

// shadowStartPlaythrough handles shadow for an anchor on a filled square.
func (gen *GordonGenerator) shadowStartPlaythrough(lastAnchorCol int,
	curLetter tilemapping.MachineLetter, dir board.BoardDirection) {
	s := &gen.shadow

	// Traverse the full length of existing tiles leftward
	for {
		s.shadowMainwordRestrictedScore += gen.letterDistribution.Score(curLetter)
		if gen.wmpMoveGen.IsActive() {
			gen.wmpMoveGen.AddPlaythroughLetter(byte(curLetter.Unblank()))
		}
		if s.currentLeftCol == 0 || s.currentLeftCol == lastAnchorCol+1 {
			break
		}
		s.currentLeftCol--
		curLetter = gen.cache.squares[s.currentLeftCol].letter
		if curLetter == 0 {
			s.currentLeftCol++
			break
		}
	}
	if gen.wmpMoveGen.IsActive() {
		// One block of leftward playthrough tiles consumed.
		// Mirrors wmp_move_gen_increment_playthrough_blocks call at
		// the bottom of MAGPIE's shadow_start_playthrough.
		gen.wmpMoveGen.IncrementPlaythroughBlocks()
	}

	gen.shadowPlaythroughPlayLeft(dir == board.HorizontalDirection)
}

// shadowNonplaythroughPlayLeft extends shadow leftward from a nonplaythrough anchor.
func (gen *GordonGenerator) shadowNonplaythroughPlayLeft(isUnique bool) {
	s := &gen.shadow

	for {
		// Try extending right first
		possibleRight := s.rackPossible(s.anchorRightExtSet)
		if possibleRight != 0 {
			gen.shadowPlayRight(isUnique)
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

		sq := &gen.cache.squares[s.currentLeftCol]
		crossSet := uint64(sq.crossSet)
		possibleHere := s.rackPossible(crossSet)
		if possibleHere == 0 {
			return
		}
		lm := int(sq.letterMul)
		wm := int(sq.wordMul)
		cs := int(sq.crossScore)
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
func (gen *GordonGenerator) shadowPlaythroughPlayLeft(isUnique bool) {
	s := &gen.shadow

	for {
		// Try extending right
		possibleRight := s.rackPossible(s.anchorRightExtSet)
		if possibleRight != 0 {
			gen.shadowPlayRight(isUnique)
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

		sq := &gen.cache.squares[s.currentLeftCol]
		crossSet := uint64(sq.crossSet)

		possibleHere := s.rackPossible(crossSet)
		if possibleHere == 0 {
			break
		}

		lm := int(sq.letterMul)
		wm := int(sq.wordMul)
		cs := int(sq.crossScore)
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
func (gen *GordonGenerator) shadowPlayRight(isUnique bool) {
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
	rackCopy := s.shadowRack // direct value copy; avoids zero-init + copy
	restrictedAny := false
	changedMuls := false

	// Snapshot WMP playthrough state so we can resume from the
	// post-shadow_play_left position when we unwind. Mirrors
	// wmp_move_gen_save_playthrough_state at the top of
	// MAGPIE's shadow_play_right.
	if gen.wmpMoveGen.IsActive() {
		gen.wmpMoveGen.SavePlaythroughState()
	}

	dim := gen.boardDim

	for s.currentRightCol < dim-1 && gen.tilesPlayed < s.numLettersOnRack {
		s.currentRightCol++
		gen.tilesPlayed++

		sq := &gen.cache.squares[s.currentRightCol]
		crossSet := uint64(sq.crossSet)

		// Possible letters: must be in cross-set, left extension set,
		// anchor right extension set, and on the rack (or blank).
		possibleHere := s.rackPossible(crossSet & uint64(sq.leftExtSet) & s.anchorRightExtSet)
		// After first use, right extension set becomes trivial
		s.anchorRightExtSet = board.TrivialCrossSet

		if possibleHere == 0 {
			break
		}

		lm := int(sq.letterMul)
		wm := int(sq.wordMul)
		cs := int(sq.crossScore)
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
		foundPlaythrough := false
		for s.currentRightCol+1 < dim {
			nextLetter := gen.cache.squares[s.currentRightCol+1].letter
			if nextLetter == 0 {
				break
			}
			foundPlaythrough = true
			if gen.wmpMoveGen.IsActive() {
				gen.wmpMoveGen.AddPlaythroughLetter(byte(nextLetter.Unblank()))
			}
			s.shadowMainwordRestrictedScore += gen.letterDistribution.Score(nextLetter)
			s.currentRightCol++
		}
		if gen.wmpMoveGen.IsActive() && foundPlaythrough {
			gen.wmpMoveGen.IncrementPlaythroughBlocks()
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
		s.shadowRack = rackCopy
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
	if gen.wmpMoveGen.IsActive() {
		gen.wmpMoveGen.RestorePlaythroughState()
	}
	gen.shadowMaybeRecalcEffMuls()
}

// shadowRecord computes the upper bound equity for the current shadow state
// and updates the highest shadow equity if it's better.
func (gen *GordonGenerator) shadowRecord() {
	s := &gen.shadow

	// WMP existence gating. When WMP is active we either:
	//   - For full-rack playthrough plays: confirm the rack + board
	//     tiles spell a real word at the right length, or
	//   - For nonplaythrough plays of length >= MIN_LENGTH: confirm
	//     at least one anagram of some subrack of that length is a
	//     real word.
	// On a miss the candidate is not recordable, so we return early.
	// Mirrors the WMP gating block at the top of MAGPIE's shadow_record.
	wmpActive := gen.wmpMoveGen.IsActive()
	useWMPLeaves := false
	if wmpActive {
		hasPlaythrough := gen.wmpMoveGen.HasPlaythrough()
		if hasPlaythrough && gen.tilesPlayed == s.numLettersOnRack {
			if !gen.wmpMoveGen.CheckPlaythroughFullRackExistence() {
				return
			}
		}
		if !hasPlaythrough && gen.tilesPlayed >= wmp.MinimumWordLength {
			if !gen.wmpMoveGen.NonplaythroughWordOfLengthExists(gen.tilesPlayed) {
				return
			}
			if s.tilesInBag > 0 {
				useWMPLeaves = true
			}
		}
	}

	// Compute unrestricted tiles score: inner product of descending tile scores
	// and descending effective letter multipliers
	tilesPlayedScore := 0
	for i := 0; i < game.RackTileLimit; i++ {
		tilesPlayedScore += s.descTileScores[i] * s.descEffLetterMuls[i]
	}

	bingoBonus := 0
	if gen.tilesPlayed == game.RackTileLimit {
		bingoBonus = 50
	}

	score := tilesPlayedScore +
		s.shadowMainwordRestrictedScore*s.shadowWordMultiplier +
		s.shadowPerpAdditionalScore +
		bingoBonus

	// Add equity adjustment matching magpie's static_eval_get_shadow_equity.
	equity := float64(score)
	if s.tilesInBag > 0 {
		// Bag not empty: add best leave value for remaining tiles.
		leaveCount := s.numLettersOnRack - gen.tilesPlayed
		if useWMPLeaves {
			// WMP-tightened best leaves only consider leaves whose
			// played-part actually has a valid word; they are <=
			// the unfiltered shadow.bestLeaves[leaveCount].
			equity += gen.wmpMoveGen.NonplaythroughBestLeaveValues()[leaveCount]
		} else {
			equity += s.bestLeaves[leaveCount]
		}
	} else {
		// Bag empty (endgame): adjustment depends on whether we play out.
		if gen.tilesPlayed < s.numLettersOnRack {
			// Not playing out: penalty based on lowest possible remaining rack score.
			// Use the lowest-valued tiles (end of descending scores) for tightest bound.
			lowestRackScore := 0
			for i := gen.tilesPlayed; i < s.numLettersOnRack; i++ {
				lowestRackScore += s.descTileScores[i]
			}
			equity += float64(-endgameNonOutplayLeavePenaltyMultiplier*lowestRackScore) - endgameNonOutplayConstantPenalty
		} else {
			// Playing out: bonus = 2 * opponent's rack score.
			equity += float64(2 * s.oppRackScore)
		}
	}

	if wmpActive {
		// Record this candidate into the per-(playthrough_blocks,
		// tiles_to_play) anchor table. The total word length is the
		// number of new tiles plus any board playthrough tiles
		// covered by the candidate. Mirrors the
		// wmp_move_gen_maybe_update_anchor call near the bottom of
		// MAGPIE's shadow_record.
		wordLength := gen.wmpMoveGen.NumTilesPlayedThrough() + gen.tilesPlayed
		if wordLength >= wmp.MinimumWordLength {
			gen.wmpMoveGen.MaybeUpdateAnchor(gen.tilesPlayed, wordLength,
				s.currentLeftCol, float64(score), equity)
		}
	}

	if equity > s.highestShadowEquity {
		s.highestShadowEquity = equity
	}
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

	// When both a real tile and blank are available for a restricted position,
	// greedily assigning the real tile may be sub-optimal if a later position
	// needs the same letter at a higher multiplier. Treat as unrestricted so
	// the inner-product rearrangement assigns tiles optimally.
	// Note: ml=0 means the blank itself is the only option (not a real+blank case).
	var tileScore int
	if s.shadowRack[ml] > 0 {
		if ml != 0 && s.shadowRack[0] > 0 {
			return false
		}
		s.shadowRack[ml]--
		if s.shadowRack[ml] == 0 {
			s.rackCrossSet &= ^possibleLetters
		}
		tileScore = gen.letterDistribution.Score(ml)
	} else {
		// Only blank can satisfy this restriction; blank scores 0
		s.shadowRack[0]--
		if s.shadowRack[0] == 0 {
			s.rackCrossSet &= ^uint64(1)
		}
		tileScore = 0
	}

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
	for i := 0; i < game.RackTileLimit; i++ {
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
		gen.lastAnchorCol = 100
		gen.cache.loadRow(gen.board, row, board.VerticalDirection, dim)
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
		gen.lastAnchorCol = 100
		gen.cache.loadRow(gen.board, row, board.HorizontalDirection, dim)
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

	// Invalidate the row cache: data from a previous GenAll call (different
	// board state) must not be reused.
	gen.cache.loadedRow = -1

	for gen.shadow.anchorHeap.len() > 0 {
		if gen.quitEarly {
			break
		}

		anchor := gen.shadow.anchorHeap.extractMax()

		// Check if we can stop: no remaining anchor can beat our best.
		// For TopPlayOnly, winner tracks the best move found.
		// For TopN, the last slot of topNPlays tracks the Nth-best.
		bestEquity := math.Inf(-1)
		if gen.winner != nil && !gen.winner.IsEmpty() {
			bestEquity = gen.winner.Equity()
			if len(gen.equityCalculators) == 0 {
				bestEquity = float64(gen.winner.Score())
			}
		}
		if n := len(gen.topNPlays); n > 0 && !gen.topNPlays[n-1].IsEmpty() {
			nth := gen.topNPlays[n-1].Equity()
			if len(gen.equityCalculators) == 0 {
				nth = float64(gen.topNPlays[n-1].Score())
			}
			if nth > bestEquity {
				bestEquity = nth
			}
		}
		if bestEquity > math.Inf(-1) && anchor.HighestPossibleEquity < bestEquity {
			if currentlyTransposed {
				gen.board.Transpose()
			}
			return
		}

		needTransposed := anchor.Dir == board.VerticalDirection
		if needTransposed != currentlyTransposed {
			gen.board.Transpose()
			currentlyTransposed = needTransposed
			// Board layout changed: cached row data is now stale.
			gen.cache.loadedRow = -1
		}
		gen.vertical = needTransposed

		// Anchor row/col are already in the correct coordinate system
		anchorRow := int(anchor.Row)
		anchorCol := int(anchor.Col)
		gen.curRowIdx = anchorRow
		gen.curAnchorCol = anchorCol
		gen.lastAnchorCol = int(anchor.LastAnchorCol)

		// Load row cache for this anchor's row, reusing the cached data
		// when consecutive anchors share the same row and direction.
		var csDir board.BoardDirection
		if anchor.Dir == board.HorizontalDirection {
			csDir = board.VerticalDirection
		} else {
			csDir = board.HorizontalDirection
		}
		if gen.cache.loadedRow != anchorRow || gen.cache.loadedDir != csDir {
			gen.cache.loadRow(gen.board, anchorRow, csDir, gen.boardDim)
		}

		// WMP-recorded anchors carry per-(blocks, tiles) extension
		// fields. Process them via wordmapGen, which retrieves the
		// word list directly from the WMP and avoids the GADDAG
		// traversal in recursiveGen. Non-WMP anchors fall back to
		// the regular Gordon recursive generator.
		if gen.wmpMoveGen.IsActive() && anchor.TilesToPlay > 0 {
			gen.wordmapGen(rack, &anchor)
		} else {
			gen.recursiveGen(anchorCol, rack, gd.GetRootNodeIndex(),
				anchorCol, anchorCol, !gen.vertical, 0, 0, 1)
		}
	}

	if currentlyTransposed {
		gen.board.Transpose()
	}
}

// RunShadowOnly runs the shadow pass and returns anchors sorted by
// descending highest possible score. For testing.
func (gen *GordonGenerator) RunShadowOnly(rack *tilemapping.Rack) []Anchor {
	gen.genShadow(rack)
	result := make([]Anchor, len(gen.shadow.anchorHeap.anchors))
	copy(result, gen.shadow.anchorHeap.anchors)
	sort.Slice(result, func(i, j int) bool {
		return result[i].HighestPossibleScore > result[j].HighestPossibleScore
	})
	return result
}

// RunShadowOnlyWMP runs the shadow pass with full WMP (and leave map)
// initialization, mirroring the slice of GenAllWithShadow that comes
// before recursiveGen. Returns the anchors in descending equity order
// (the same order MAGPIE's extract_sorted_anchors_for_test produces).
//
// tilesInBag and oppRackScore are stamped onto the generator so
// shadow_record's equity branch matches the MAGPIE shadow tests.
// For score-only assertions (sortingParameter == SortByScore) the
// exact tilesInBag value doesn't matter as long as it is > 0, since
// shadow.bestLeaves remains zero with no equity calculator set.
//
// Intended for tests that want to inspect anchor output without
// running the full recursive_gen pipeline.
func (gen *GordonGenerator) RunShadowOnlyWMP(rack *tilemapping.Rack, tilesInBag, oppRackScore int) []Anchor {
	gen.tilesInBag = tilesInBag
	gen.oppRackScore = oppRackScore
	gen.shadow.tilesInBag = tilesInBag
	gen.shadow.oppRackScore = oppRackScore

	gen.leavemap.Init(rack)
	for i := range gen.shadow.bestLeaves {
		gen.shadow.bestLeaves[i] = 0
	}

	gen.wmpMoveGen.Init(gen.letterDistribution, rack, gen.wmpData)
	if gen.wmpMoveGen.IsActive() {
		checkLeaves := tilesInBag > 0 && gen.sortingParameter != SortByScore
		gen.wmpMoveGen.CheckNonplaythroughExistence(checkLeaves, &gen.leavemap)
	}

	gen.genShadow(rack)

	// Extract anchors in descending equity order, matching MAGPIE's
	// extract_sorted_anchors_for_test behavior.
	n := gen.shadow.anchorHeap.len()
	sorted := make([]Anchor, 0, n)
	for gen.shadow.anchorHeap.len() > 0 {
		sorted = append(sorted, gen.shadow.anchorHeap.extractMax())
	}
	return sorted
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

	gen.smallPlays = gen.smallPlays[:0]

	// Cache equity state for the fast path in TopPlayOnlyRecorder.
	// The leave map fast path only works with CombinedStaticCalculator
	// (which bundles leave + peg + endgame into one). When separate
	// calculators are used (e.g., HASTY_BOT with 4 calculators),
	// the leave map can't replicate the full equity sum.
	gen.klv = nil
	gen.pegValues = nil
	if len(gen.equityCalculators) == 1 {
		if csc, ok := gen.equityCalculators[0].(*equity.CombinedStaticCalculator); ok {
			gen.klv = csc.KLV()
			gen.pegValues = csc.PEGValues()
		}
	}
	if gen.game != nil {
		gen.tilesInBag = gen.game.Bag().TilesRemaining()
		oppRack := gen.game.RackFor(gen.game.NextPlayer())
		gen.oppRackScore = 0
		for ml := tilemapping.MachineLetter(1); int(ml) < len(oppRack.LetArr); ml++ {
			gen.oppRackScore += oppRack.LetArr[ml] * gen.letterDistribution.Score(ml)
		}
		// Also set shadow state (used by shadowRecord for endgame equity).
		// When computeBestLeaves is skipped (leave map handles bestLeaves),
		// these must still be current for the shadow pass.
		gen.shadow.tilesInBag = gen.tilesInBag
		gen.shadow.oppRackScore = gen.oppRackScore
	}
	gen.leavemap.Init(rack)

	// Initialize bestLeaves before population.
	for i := range gen.shadow.bestLeaves {
		gen.shadow.bestLeaves[i] = math.Inf(-1)
	}

	// Populate leave map via incremental KWG traversal.
	// Also computes bestLeaves from rawLeave (only reachable indices).
	if gen.klv != nil && gen.game != nil {
		gen.populateLeaveMap(rack)
	} else {
		gen.leavemap.Initialized = false
	}

	// When leave map wasn't populated, fall back to enumeration.
	if !gen.leavemap.Initialized {
		gen.computeBestLeaves(rack)
	}

	// Initialize the WMP move gen for this rack and (if active) run
	// the nonplaythrough subrack enumeration so shadow_record can
	// gate on word-length existence and use WMP-tightened best leaves.
	// Mirrors generate_moves: wmp_move_gen_init in gen_load_position
	// followed by wmp_move_gen_check_nonplaythrough_existence between
	// gen_look_up_leaves_and_record_exchanges and gen_shadow.
	gen.wmpMoveGen.Init(gen.letterDistribution, rack, gen.wmpData)
	if gen.wmpMoveGen.IsActive() {
		// shadowRecord unconditionally consumes
		// wmpMoveGen.NonplaythroughBestLeaveValues() whenever
		// tilesInBag > 0 (see useWMPLeaves, shadow.go ~line 783),
		// so those values must be populated with real leaves
		// whenever that condition can fire — regardless of the
		// final sorting parameter, which only controls post-gen
		// sort and is left as SortByScore even when TopPlayOnly /
		// TopN recorders pick winners by equity inline. Tying
		// checkLeaves to the sort parameter (the previous
		// behavior) caused the WMP shadow bound to assume the best
		// leave was zero, which pruned low-score / high-leave
		// plays whose real equity would otherwise have won.
		// Caught by automatic.RunWMPAgreementTest.
		checkLeaves := gen.shadow.tilesInBag > 0
		gen.wmpMoveGen.CheckNonplaythroughExistence(checkLeaves, &gen.leavemap)
	}

	// Run shadow to rank anchors
	gen.genShadow(rack)

	// Generate moves from anchors in best-first order
	gen.genRecordScoringPlaysFromAnchors(rack)

	// Generate exchange moves after tile placements.
	if addExchange {
		gen.generateExchangeMoves(rack, 0, 0)
	}

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
