package wmp

import (
	"math"
	"math/bits"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/leavemap"
)

// Constants matching MAGPIE's wmp_move_gen.h.
const (
	// RackSize is the maximum number of tiles on a player's rack.
	// MAGPIE compiles with RACK_SIZE=7 and the offsets table below
	// is generated for that case.
	RackSize = 7
	// MinimumWordLength is the smallest playable word length.
	MinimumWordLength = 2
	// MaxPossiblePlaythroughBlocks bounds how many disjoint runs of
	// playthrough tiles a single anchor's word can cross. With a
	// 15-tile board the worst case is 8 (alternating tile/empty).
	MaxPossiblePlaythroughBlocks = (15 / 2) + 1
	// MaxWMPMoveGenAnchors is the size of the per-anchor scratch
	// table indexed by (playthrough_blocks, tiles_to_play).
	// playthrough_blocks ranges 0..MaxPossiblePlaythroughBlocks
	// inclusive, so the table needs one extra row.
	MaxWMPMoveGenAnchors = (RackSize + 1) * (MaxPossiblePlaythroughBlocks + 1)
)

// EquityMinValue is the sentinel returned for "no anchor seen yet"
// equity / score fields. Mirrors MAGPIE's EQUITY_MIN_VALUE.
var EquityMinValue = math.Inf(-1)

// combinationOffsets is the per-size starting offset into the flat
// 128-entry subrack array. For RACK_SIZE=7:
//
//	C(7,0)=1, C(7,1)=7, C(7,2)=21, C(7,3)=35,
//	C(7,4)=35, C(7,5)=21, C(7,6)=7,  C(7,7)=1.
//
// Cumulative: [0, 1, 8, 29, 64, 99, 120, 127]. Total = 128.
//
// Mirrors MAGPIE's BIT_RACK_COMBINATION_OFFSETS macro.
var combinationOffsets = [RackSize + 1]int{0, 1, 8, 29, 64, 99, 120, 127}

// totalSubracks is 2^RackSize, the size of the flat subrack arrays.
const totalSubracks = 1 << RackSize

// The MoveGen consumes the unified leavemap.LeaveMap (in package
// macondo/leavemap), using its complement-mode methods during
// nonplaythrough subrack enumeration. The same leave map type is used
// by macondo's regular shadow play through its regular AddLetter /
// TakeLetter methods.

// SubrackInfo holds one enumerated subrack: the multiset of tiles and the
// looked-up WMP entry (or nil). Mirrors MAGPIE's SubrackInfo. The subrack's
// leave value and tile score sum are kept apart, in arrays of their own
// (MoveGen.leaveValues, tileScoreSums), so the leave check reads only those.
type SubrackInfo struct {
	Subrack  BitRack
	WMPEntry *Entry
}

// blankWordCacheBytes is the size of a generator's cache of the word lists
// of the rack's subracks with a blank (see GetSubrackWords).
const blankWordCacheBytes = 4 * ResultBufferSize

// ScoreBaseCount is the number of score bases kept per anchor. See
// RecordPlacement.
const ScoreBaseCount = 2

// AnchorSlot is what the shadow has found for one (playthrough_blocks,
// tiles_to_play) pair of the square it is on: the placements' range of start
// columns, their highest score and equity bounds, and their score bases (see
// RecordPlacement). TilesToPlay is 0 for a slot with no placement. Kept
// small: the shadow writes one for every placement it bounds.
type AnchorSlot struct {
	HighestPossibleEquity float64
	HighestPossibleScore  int32
	ScoreBases            [ScoreBaseCount]int32
	TilesToPlay           uint8
	PlaythroughBlocks     uint8
	WordLength            uint8
	LeftmostStartCol      uint8
	RightmostStartCol     uint8
}

// Anchor is an anchor as PlaythroughSubracksInit takes it. Mirrors MAGPIE's
// Anchor (the WMP-relevant subset of fields).
type Anchor struct {
	HighestPossibleEquity float64
	HighestPossibleScore  float64
	LeftmostStartCol      int
	RightmostStartCol     int
	TilesToPlay           int
	PlaythroughBlocks     int
	WordLength            int
}

// MoveGen is the WMP-based move generator state. It mirrors MAGPIE's
// WMPMoveGen struct field-for-field (modulo Go conventions). The flow:
//
//  1. Init with the player's rack and a WMP. Becomes "active" iff a
//     non-nil WMP is supplied.
//  2. ResetPlaythrough then either:
//     a. CheckNonplaythroughExistence to enumerate all rack subsets
//     and look up which lengths have valid blankless plays, or
//     b. AddPlaythroughLetter / CheckPlaythroughFullRackExistence to
//     test specific board-anchored bingo candidates.
//  3. PlaythroughSubracksInit + GetSubrackWords to retrieve actual
//     words once an anchor has been picked.
//
// Anchor management (RecordPlacement) is done as the move generator
// shadow-plays across the board.
type MoveGen struct {
	wmp           *WMP
	playerBitRack BitRack
	fullRackSize  int

	// Tiles already played on the board for the current anchor.
	playthroughBitRack    BitRack
	numTilesPlayedThrough int
	playthroughBlocks     int

	// Snapshot of playthrough state, used by shadow_play_right to
	// reset back to the post-shadow_play_left position.
	playthroughBitRackCopy    BitRack
	numTilesPlayedThroughCopy int
	playthroughBlocksCopy     int

	// Pre-enumerated subracks. Each is a flat 128-entry array; the
	// per-size buckets start at combinationOffsets[size]. Indices
	// within a size run from 0..countBySize[size]-1.
	nonplaythroughInfos [totalSubracks]SubrackInfo
	playthroughInfos    [totalSubracks]SubrackInfo
	// leaveValues and tileScoreSums are the leave value and the sum of the
	// tile scores (a blank is 0) of each nonplaythrough subrack, by the
	// same index.
	leaveValues   [totalSubracks]float64
	tileScoreSums [totalSubracks]int32
	// playthroughHashes holds, for the subracks CandidateSubracks marked
	// that have playthrough tiles and no blank, the word map hash of the
	// subrack with those tiles, by index within the size.
	playthroughHashes [64]uint64

	// nonplaythroughBestLeaveValues[leaveSize] = best leave value
	// across all subracks of size (fullRackSize-leaveSize) that have
	// at least one valid word. Computed only when checkLeaves=true.
	nonplaythroughBestLeaveValues [RackSize + 1]float64
	// nonplaythroughHasWordOfLength[length] = true iff at least one
	// blankless subrack of that size matches a real word.
	nonplaythroughHasWordOfLength [RackSize + 1]bool
	countBySize                   [RackSize + 1]uint8

	anchors [MaxWMPMoveGenAnchors]AnchorSlot
	// tileScores[ml] is the score of tile ml, from the letter distribution
	// given to Init.
	tileScores   [maxAlphabetSize]int32
	tileScoresLD *tilemapping.LetterDistribution
	// dirtyAnchors holds the indices into anchors that were touched
	// since the last ResetAnchors call. This lets ResetAnchors avoid
	// re-zeroing the full 64-entry table on every shadow square,
	// which was the biggest hot spot in the profiler.
	dirtyAnchors []int

	// Scratch buffer used by GetSubrackWords for write_words_to_buffer.
	buffer [ResultBufferSize]byte
	// words is the word list GetSubrackWords found, numWords words of
	// wordLength letters end to end. It is the word map's own storage for
	// a subrack with no blank, and the buffer or the blank word cache
	// otherwise.
	words []byte
	// The blank word cache: the word lists of the rack's nonplaythrough
	// subracks with a blank, which take a lookup per letter the blank can
	// be to put together, kept for the next anchor that asks. blankOffsets
	// is the list's start in blankWords plus one, 0 for none, by subrack;
	// the lists are for the rack of the last CheckNonplaythroughExistence
	// or LoadRackState.
	blankWords   []byte
	blankUsed    int
	blankOffsets [totalSubracks]uint16
	blankCounts  [totalSubracks]uint16
	tilesToPlay  int
	wordLength  int
	numWords    int
	leaveValue  float64
}

// NewMoveGen returns a freshly zeroed MoveGen. Call Init before use.
func NewMoveGen() *MoveGen {
	return &MoveGen{}
}

// Init sets up the move generator for the given rack and WMP. If wmp,
// rack, or ld is nil, the MoveGen will report IsActive() == false and
// be a no-op (mirrors MAGPIE's wmp_move_gen_init nil-checks). The ld
// parameter is unused in the Go port — kept in the signature for
// API parity with MAGPIE.
func (mg *MoveGen) Init(ld *tilemapping.LetterDistribution, rack *tilemapping.Rack, wmp *WMP) {
	mg.wmp = wmp
	if wmp == nil || rack == nil || ld == nil {
		return
	}
	if mg.tileScoresLD != ld {
		mg.tileScoresLD = ld
		for ml := range mg.tileScores {
			mg.tileScores[ml] = 0
			if ml < int(ld.TileMapping().NumLetters()) {
				mg.tileScores[ml] = int32(ld.Score(tilemapping.MachineLetter(ml)))
			}
		}
	}
	mg.playerBitRack = BitRackFromRack(rack)
	mg.fullRackSize = int(rack.NumTiles())
	for i := range mg.nonplaythroughHasWordOfLength {
		mg.nonplaythroughHasWordOfLength[i] = false
	}
}

// RackState is what CheckNonplaythroughExistence works out for a rack: its
// subracks with their leave values and word map entries, and what follows
// from them. It depends on the rack, the leave values and the word map
// only, so a generator that meets the same rack again can put it back
// instead of working it out.
type RackState struct {
	infos           [totalSubracks]SubrackInfo
	leaveValues     [totalSubracks]float64
	tileScoreSums   [totalSubracks]int32
	bestLeaveValues [RackSize + 1]float64
	hasWordOfLength [RackSize + 1]bool
	countBySize     [RackSize + 1]uint8
}

// SaveRackState copies out the state CheckNonplaythroughExistence left.
func (mg *MoveGen) SaveRackState(st *RackState) {
	st.infos = mg.nonplaythroughInfos
	st.leaveValues = mg.leaveValues
	st.tileScoreSums = mg.tileScoreSums
	st.bestLeaveValues = mg.nonplaythroughBestLeaveValues
	st.hasWordOfLength = mg.nonplaythroughHasWordOfLength
	st.countBySize = mg.countBySize
}

// LoadRackState puts back a state saved for the rack Init was just given,
// with the same word map and leave values, in place of a call of
// CheckNonplaythroughExistence.
func (mg *MoveGen) LoadRackState(st *RackState) {
	mg.nonplaythroughInfos = st.infos
	mg.leaveValues = st.leaveValues
	mg.tileScoreSums = st.tileScoreSums
	mg.resetBlankWordCache()
	mg.nonplaythroughBestLeaveValues = st.bestLeaveValues
	mg.nonplaythroughHasWordOfLength = st.hasWordOfLength
	mg.countBySize = st.countBySize
}

// IsActive reports whether the MoveGen has a non-nil WMP and should be
// consulted by the move generator. Mirrors wmp_move_gen_is_active.
func (mg *MoveGen) IsActive() bool {
	return mg.wmp != nil
}

// ResetAnchors clears the per-(blocks, tiles) anchor table to "no
// anchor recorded" sentinels. Call before each anchor pass during
// shadow play. Mirrors wmp_move_gen_reset_anchors.
//
// Optimization: instead of re-zeroing the full 64-entry table on
// every shadow square, only reset slots that were touched since the
// last reset (tracked via dirtyAnchors). RecordPlacement records
// each newly-touched slot, and RecordPlacement's "first touch"
// branch is responsible for installing the per-slot sentinels (so
// ResetAnchors itself can leave them implicit).
func (mg *MoveGen) ResetAnchors() {
	for _, idx := range mg.dirtyAnchors {
		mg.anchors[idx] = AnchorSlot{}
	}
	mg.dirtyAnchors = mg.dirtyAnchors[:0]
}

// =====================================================================
// Playthrough state
// =====================================================================

// ResetPlaythrough clears the playthrough state. Mirrors
// wmp_move_gen_reset_playthrough.
func (mg *MoveGen) ResetPlaythrough() {
	mg.playthroughBitRack = BitRack{}
	mg.numTilesPlayedThrough = 0
	mg.playthroughBlocks = 0
}

// HasPlaythrough reports whether any playthrough tiles have been
// recorded. Mirrors wmp_move_gen_has_playthrough.
func (mg *MoveGen) HasPlaythrough() bool {
	return mg.numTilesPlayedThrough > 0
}

// NumTilesPlayedThrough returns the count of playthrough tiles
// currently recorded for the active anchor. Used by shadow_record to
// compute the full word length (tiles_played + numTilesPlayedThrough).
func (mg *MoveGen) NumTilesPlayedThrough() int {
	return mg.numTilesPlayedThrough
}

// PlaythroughBlocks returns the number of distinct playthrough blocks
// (contiguous runs of board tiles) the move gen has crossed.
func (mg *MoveGen) PlaythroughBlocks() int {
	return mg.playthroughBlocks
}

// DirtyAnchorIndices returns the indices into the per-(blocks, tiles)
// anchor table that were touched since the last ResetAnchors call.
// The caller uses AnchorAt to read each touched slot.
func (mg *MoveGen) DirtyAnchorIndices() []int {
	return mg.dirtyAnchors
}

// AnchorAt returns a pointer to the anchor slot at the given index
// into the per-(blocks, tiles) table. Pair with DirtyAnchorIndices
// to walk only the live entries.
func (mg *MoveGen) AnchorAt(idx int) *AnchorSlot {
	return &mg.anchors[idx]
}

// AddPlaythroughLetter records one playthrough letter at the current
// anchor. Mirrors wmp_move_gen_add_playthrough_letter.
func (mg *MoveGen) AddPlaythroughLetter(ml byte) {
	mg.playthroughBitRack.AddLetter(ml)
	mg.numTilesPlayedThrough++
}

// IncrementPlaythroughBlocks records that the current letter starts
// a new contiguous run of playthrough tiles. Mirrors
// wmp_move_gen_increment_playthrough_blocks.
func (mg *MoveGen) IncrementPlaythroughBlocks() {
	mg.playthroughBlocks++
}

// SavePlaythroughState snapshots the playthrough fields so a
// shadow_play_right can be reset back to the post-shadow_play_left
// position. Mirrors wmp_move_gen_save_playthrough_state.
func (mg *MoveGen) SavePlaythroughState() {
	mg.playthroughBitRackCopy = mg.playthroughBitRack
	mg.numTilesPlayedThroughCopy = mg.numTilesPlayedThrough
	mg.playthroughBlocksCopy = mg.playthroughBlocks
}

// RestorePlaythroughState reverts to the snapshot taken by
// SavePlaythroughState. Mirrors wmp_move_gen_restore_playthrough_state.
func (mg *MoveGen) RestorePlaythroughState() {
	mg.playthroughBitRack = mg.playthroughBitRackCopy
	mg.numTilesPlayedThrough = mg.numTilesPlayedThroughCopy
	mg.playthroughBlocks = mg.playthroughBlocksCopy
}

// CheckPlaythroughFullRackExistence asks "is there any word of length
// (fullRackSize + numTilesPlayedThrough) using all of the player's
// tiles plus all current playthrough tiles?" Used by the shadow play
// to detect anchor positions where a bingo is possible.
//
// The result is cached on playthroughInfos[combinationOffsets[fullRackSize]]
// for later word retrieval. Mirrors
// wmp_move_gen_check_playthrough_full_rack_existence.
func (mg *MoveGen) CheckPlaythroughFullRackExistence() bool {
	size := mg.fullRackSize
	idx := combinationOffsets[size]
	info := &mg.playthroughInfos[idx]
	info.Subrack = mg.playerBitRack
	info.Subrack.AddBitRack(&mg.playthroughBitRack)
	wordSize := size + mg.numTilesPlayedThrough
	info.WMPEntry = mg.wmp.GetWordEntry(&info.Subrack, wordSize)
	return info.WMPEntry != nil
}

// =====================================================================
// Nonplaythrough subrack enumeration and existence checks
// =====================================================================

// enumerateNonplaythroughSubracks recursively enumerates every distinct
// multiset subrack of the player's rack and stores it (with its leave
// value) in nonplaythroughInfos. Mirrors
// wmp_move_gen_enumerate_nonplaythrough_subracks.
func (mg *MoveGen) enumerateNonplaythroughSubracks(current *BitRack, nextML int, count int, scoreSum int32, lm *leavemap.LeaveMap) {
	maxNumThis := 0
	for ; nextML < maxAlphabetSize; nextML++ {
		maxNumThis = mg.playerBitRack.GetLetter(byte(nextML))
		if maxNumThis > 0 {
			break
		}
	}
	if nextML >= maxAlphabetSize {
		insertIndex := combinationOffsets[count] + int(mg.countBySize[count])
		mg.nonplaythroughInfos[insertIndex].Subrack = *current
		mg.leaveValues[insertIndex] = lm.CurrentValue()
		mg.tileScoreSums[insertIndex] = scoreSum
		mg.countBySize[count]++
		return
	}
	tileScore := mg.tileScores[nextML]
	for i := 0; i < maxNumThis; i++ {
		mg.enumerateNonplaythroughSubracks(current, nextML+1, count+i, scoreSum+int32(i)*tileScore, lm)
		current.AddLetter(byte(nextML))
		lm.ComplementAddLetter(tilemapping.MachineLetter(nextML), i)
	}
	mg.enumerateNonplaythroughSubracks(current, nextML+1, count+maxNumThis, scoreSum+int32(maxNumThis)*tileScore, lm)
	for i := maxNumThis - 1; i >= 0; i-- {
		current.TakeLetter(byte(nextML))
		lm.ComplementTakeLetter(tilemapping.MachineLetter(nextML), i)
	}
}

// checkNonplaythroughsOfSize looks up every previously enumerated
// subrack of the given size in the WMP and updates
// nonplaythroughHasWordOfLength / nonplaythroughBestLeaveValues
// accordingly. Mirrors wmp_move_gen_check_nonplaythroughs_of_size.
func (mg *MoveGen) checkNonplaythroughsOfSize(size int, checkLeaves bool) {
	leaveSize := mg.fullRackSize - size
	if checkLeaves {
		mg.nonplaythroughBestLeaveValues[leaveSize] = EquityMinValue
	} else {
		mg.nonplaythroughBestLeaveValues[leaveSize] = 0
	}
	offset := combinationOffsets[size]
	count := int(mg.countBySize[size])
	for idxForSize := 0; idxForSize < count; idxForSize++ {
		info := &mg.nonplaythroughInfos[offset+idxForSize]
		info.WMPEntry = mg.wmp.GetWordEntry(&info.Subrack, size)
		if info.WMPEntry == nil {
			continue
		}
		mg.nonplaythroughHasWordOfLength[size] = true
		if !checkLeaves {
			continue
		}
		if lv := mg.leaveValues[offset+idxForSize]; lv > mg.nonplaythroughBestLeaveValues[leaveSize] {
			mg.nonplaythroughBestLeaveValues[leaveSize] = lv
		}
	}
}

// CheckNonplaythroughExistence enumerates all rack subsets and queries
// the WMP to find out which word lengths have at least one valid
// blankless play. If checkLeaves is true, the best leave value for
// each "leave size" (= rack size minus word length) is also computed.
//
// The leave map is mutated during enumeration via its complement
// methods, then left in an undefined state on return.
//
// Mirrors wmp_move_gen_check_nonplaythrough_existence.
func (mg *MoveGen) CheckNonplaythroughExistence(checkLeaves bool, lm *leavemap.LeaveMap) {
	lm.SetCurrentIndex((1 << mg.fullRackSize) - 1)
	mg.resetBlankWordCache()
	for i := range mg.countBySize {
		mg.countBySize[i] = 0
	}
	for i := range mg.nonplaythroughHasWordOfLength {
		mg.nonplaythroughHasWordOfLength[i] = false
	}
	var empty BitRack
	mg.enumerateNonplaythroughSubracks(&empty, blankMachineLetter, 0, 0, lm)
	for size := MinimumWordLength; size <= mg.fullRackSize; size++ {
		mg.checkNonplaythroughsOfSize(size, checkLeaves)
	}
}

// NonplaythroughWordOfLengthExists reports whether at least one
// blankless subrack of the given size matches a valid word. Returns
// false for sizes outside [MinimumWordLength, RackSize].
func (mg *MoveGen) NonplaythroughWordOfLengthExists(wordLength int) bool {
	if wordLength < 0 || wordLength > RackSize {
		return false
	}
	return mg.nonplaythroughHasWordOfLength[wordLength]
}

// NonplaythroughBestLeaveValues returns the slice of best leave values
// indexed by leave size (= fullRackSize - wordLength). Valid indices
// run from 0 to RackSize. Caller must not mutate the returned slice.
func (mg *MoveGen) NonplaythroughBestLeaveValues() []float64 {
	return mg.nonplaythroughBestLeaveValues[:]
}

// =====================================================================
// Anchor recording
// =====================================================================

func wmpMoveGenAnchorIndex(playthroughBlocks, tilesPlayed int) int {
	return playthroughBlocks*(RackSize+1) + tilesPlayed
}

// RecordPlacement folds one shadow placement into the slot of the current
// (playthrough_blocks, tiles_played) pair: it widens the slot's range of
// start columns, raises its highest score and equity bounds, and updates its
// score bases. Mirrors MAGPIE's wmp_move_gen_maybe_update_anchor and
// wmp_move_gen_update_anchor_score_bases.
//
// A score base bounds the scores of an anchor's plays by the tiles they
// play: for every play of the anchor,
//
//	score <= base[b] + (b+1) * (sum of the play's tile scores)
//
// The shadow's bound for a placement puts the rack's best tiles on its
// squares, each tile's score times its square's effective multiplier, both
// in descending order. A play there with other tiles has, in descending
// order, tiles each no better than the shadow's, so it scores less by at
// least the difference of the two tile score sums times the smallest of the
// multipliers. So for a multiplier m no greater than that smallest one,
//
//	score <= bound - m*(shadow's tile score sum) + m*(play's tile score sum)
//
// and a greater m only adds to the last term. base[b] is the largest value
// of the first two terms over the slot's placements, with m = b+1 capped at
// each placement's smallest multiplier. Tiles the shadow restricted to one
// square are in every play of the placement and are counted in both sums.
//
// A slot's first placement appends its index to dirtyAnchors, so that the
// next ResetAnchors clears it.
func (mg *MoveGen) RecordPlacement(tilesPlayed, wordLength, startCol, score int, equity float64,
	shadowTileScoreSum, smallestMultiplier int) {
	idx := wmpMoveGenAnchorIndex(mg.playthroughBlocks, tilesPlayed)
	anchor := &mg.anchors[idx]
	base0 := int32(score - min(1, smallestMultiplier)*shadowTileScoreSum)
	base1 := int32(score - min(2, smallestMultiplier)*shadowTileScoreSum)
	if anchor.TilesToPlay == 0 {
		*anchor = AnchorSlot{
			HighestPossibleEquity: equity,
			HighestPossibleScore:  int32(score),
			ScoreBases:            [ScoreBaseCount]int32{base0, base1},
			TilesToPlay:           uint8(tilesPlayed),
			PlaythroughBlocks:     uint8(mg.playthroughBlocks),
			WordLength:            uint8(wordLength),
			LeftmostStartCol:      uint8(startCol),
			RightmostStartCol:     uint8(startCol),
		}
		mg.dirtyAnchors = append(mg.dirtyAnchors, idx)
		return
	}
	anchor.WordLength = uint8(wordLength)
	if c := uint8(startCol); c < anchor.LeftmostStartCol {
		anchor.LeftmostStartCol = c
	} else if c > anchor.RightmostStartCol {
		anchor.RightmostStartCol = c
	}
	if equity > anchor.HighestPossibleEquity {
		anchor.HighestPossibleEquity = equity
	}
	if int32(score) > anchor.HighestPossibleScore {
		anchor.HighestPossibleScore = int32(score)
	}
	if base0 > anchor.ScoreBases[0] {
		anchor.ScoreBases[0] = base0
	}
	if base1 > anchor.ScoreBases[1] {
		anchor.ScoreBases[1] = base1
	}
}

// GetTileScoreSum returns the sum of the tile scores of the idxForSize'th
// subrack of the current tilesToPlay size.
func (mg *MoveGen) GetTileScoreSum(idxForSize int) int32 {
	return mg.tileScoreSums[combinationOffsets[mg.tilesToPlay]+idxForSize]
}

// =====================================================================
// Word retrieval (after an anchor is selected)
// =====================================================================

// PlaythroughSubracksInit sets the move gen up for word retrieval against
// the given anchor: its word length and number of tiles to play.
// CandidateSubracks then says which subracks to ask GetSubrackWords for.
// Mirrors wmp_move_gen_playthrough_subracks_init.
func (mg *MoveGen) PlaythroughSubracksInit(anchor *Anchor) {
	subrackSize := anchor.TilesToPlay
	mg.wordLength = anchor.WordLength
	mg.numTilesPlayedThrough = anchor.WordLength - subrackSize
	mg.tilesToPlay = subrackSize
}

// SubrackScoreBound is the least of the anchor's score bound and what its
// score bases allow a subrack whose tile scores sum to tileScoreSum (see
// RecordPlacement).
func SubrackScoreBound(scoreBound int32, bases *[ScoreBaseCount]int32, tileScoreSum int32) int32 {
	bound := scoreBound
	for b, base := range bases {
		if v := base + int32(b+1)*tileScoreSum; v < bound {
			bound = v
		}
	}
	return bound
}

// CandidateSubracks returns the subracks of the current anchor's size that
// may have plays, as a bit set by index within the size; GetSubrackWords
// may be called for those only, and returns false for every other. A
// subrack is left out when:
//
//   - prune is set and its score bound (SubrackScoreBound, of scoreBound
//     and bases) plus its leave value is below cutoff: the caller is
//     saying that it would turn every such play away;
//   - the anchor has no playthrough tiles and the subrack's entry, looked
//     up when the rack was set up, is nil;
//   - the anchor has playthrough tiles, the subrack has no blank, and the
//     word map's filter says that the subrack with those tiles makes no
//     word.
//
// Mirrors wmp_move_gen_get_candidate_subracks. The leave check comes first
// and reads two small arrays; only what it keeps is hashed and tested
// against the filter, and nothing here reads the word map's buckets.
func (mg *MoveGen) CandidateSubracks(prune bool, scoreBound int32, bases *[ScoreBaseCount]int32, cutoff float64) uint64 {
	offset := combinationOffsets[mg.tilesToPlay]
	count := int(mg.countBySize[mg.tilesToPlay])
	leaveValues := mg.leaveValues[offset : offset+count]
	tileScoreSums := mg.tileScoreSums[offset : offset+count]
	kept := uint64(1)<<uint(count) - 1
	if prune {
		// The same test as SubrackScoreBound and a comparison, written
		// out so that a subrack costs no call and no branch.
		kept = 0
		base1, base2 := bases[0], bases[1]
		tileScoreSums = tileScoreSums[:len(leaveValues)]
		for i, lv := range leaveValues {
			ts := tileScoreSums[i]
			bound := min(scoreBound, base1+ts, base2+2*ts)
			var keep uint64
			if !(float64(bound)+lv < cutoff) {
				keep = 1
			}
			kept |= keep << (uint(i) & 63)
		}
	}
	candidates := uint64(0)
	if mg.wordLength <= mg.tilesToPlay {
		infos := mg.nonplaythroughInfos[offset : offset+count]
		for ; kept != 0; kept &= kept - 1 {
			i := bits.TrailingZeros64(kept)
			if infos[i].WMPEntry != nil {
				candidates |= 1 << uint(i)
			}
		}
		return candidates
	}
	wfl := &mg.wmp.WFLs[mg.wordLength]
	nonpt := mg.nonplaythroughInfos[offset : offset+count]
	pt := mg.playthroughInfos[offset : offset+count]
	for ; kept != 0; kept &= kept - 1 {
		i := bits.TrailingZeros64(kept)
		br := nonpt[i].Subrack
		br.AddBitRack(&mg.playthroughBitRack)
		pt[i].Subrack = br
		if br.GetLetter(blankMachineLetter) != 0 {
			candidates |= 1 << uint(i)
			continue
		}
		hash := br.MixTo64()
		if wfl.mayHaveWords(hash) {
			mg.playthroughHashes[i] = hash
			candidates |= 1 << uint(i)
		}
	}
	return candidates
}

// NumSubrackCombinations returns the number of distinct subracks of
// the current tilesToPlay size. Set by PlaythroughSubracksInit (or
// implicitly by enumeration during shadow play).
func (mg *MoveGen) NumSubrackCombinations() int {
	return int(mg.countBySize[mg.tilesToPlay])
}

// GetNonplaythroughSubrack returns the BitRack for the idxForSize'th
// subrack of the current tilesToPlay size, used by callers that need
// to inspect specific subracks during word retrieval.
func (mg *MoveGen) GetNonplaythroughSubrack(idxForSize int) *BitRack {
	offset := combinationOffsets[mg.tilesToPlay]
	return &mg.nonplaythroughInfos[offset+idxForSize].Subrack
}

// resetBlankWordCache forgets the cached word lists: the rack has changed.
func (mg *MoveGen) resetBlankWordCache() {
	if mg.blankUsed != 0 {
		mg.blankOffsets = [totalSubracks]uint16{}
		mg.blankUsed = 0
	}
}

// GetSubrackWords finds the words of the idxForSize'th subrack, one that
// CandidateSubracks marked for the current anchor. Returns false if it has
// none. After a true return, Words and GetWord give them. Mirrors
// wmp_move_gen_get_subrack_words.
//
// The words of a subrack with no blank are read where the word map keeps
// them. Those of one with a blank are put together from the words of each
// letter the blank can be; for a subrack with no playthrough tiles the
// result is kept for the rest of the rack's generation, since every anchor
// with that many tiles to play and no playthrough asks for it again.
func (mg *MoveGen) GetSubrackWords(idxForSize int) bool {
	subrackIdx := combinationOffsets[mg.tilesToPlay] + idxForSize
	isPlaythrough := mg.wordLength > mg.tilesToPlay
	wfl := &mg.wmp.WFLs[mg.wordLength]
	var info *SubrackInfo
	var blanks int
	if isPlaythrough {
		// CandidateSubracks left the subrack with its playthrough tiles
		// here, and its hash if it has no blank.
		info = &mg.playthroughInfos[subrackIdx]
		blanks = info.Subrack.GetLetter(blankMachineLetter)
		switch blanks {
		case 0:
			info.WMPEntry = wfl.getHashedWordEntry(&info.Subrack, mg.playthroughHashes[idxForSize&63])
		case 1:
			info.WMPEntry = wfl.getBlankEntry(&info.Subrack)
		case 2:
			info.WMPEntry = wfl.getDoubleBlankEntry(&info.Subrack)
		default:
			info.WMPEntry = nil
		}
	} else {
		// Looked up by CheckNonplaythroughExistence.
		info = &mg.nonplaythroughInfos[subrackIdx]
		blanks = info.Subrack.GetLetter(blankMachineLetter)
	}
	entry := info.WMPEntry
	if entry == nil {
		return false
	}
	if blanks == 0 {
		if entry.IsInlined() {
			mg.words = entry[:entry.numberOfInlinedBytes(mg.wordLength)]
		} else {
			start := int(entry.WordStart())
			mg.words = wfl.WordLetters[start : start+int(entry.NumWords())*mg.wordLength]
		}
		mg.numWords = len(mg.words) / mg.wordLength
		return mg.numWords > 0
	}
	if !isPlaythrough {
		if off := int(mg.blankOffsets[subrackIdx]); off != 0 {
			mg.numWords = int(mg.blankCounts[subrackIdx])
			mg.words = mg.blankWords[off-1 : off-1+mg.numWords*mg.wordLength]
			return true
		}
	}
	// A list is cached only while the cache has room for the longest.
	cache := !isPlaythrough && mg.blankUsed <= blankWordCacheBytes-ResultBufferSize
	buffer := mg.buffer[:]
	if cache {
		if mg.blankWords == nil {
			mg.blankWords = make([]byte, blankWordCacheBytes)
		}
		buffer = mg.blankWords[mg.blankUsed:]
	}
	resultBytes := mg.wmp.WriteWordsForEntry(entry, &info.Subrack, mg.wordLength, buffer)
	if resultBytes <= 0 || resultBytes%mg.wordLength != 0 {
		return false
	}
	mg.numWords = resultBytes / mg.wordLength
	mg.words = buffer[:resultBytes]
	if cache && mg.numWords <= math.MaxUint16 {
		mg.blankOffsets[subrackIdx] = uint16(mg.blankUsed + 1)
		mg.blankCounts[subrackIdx] = uint16(mg.numWords)
		mg.blankUsed += resultBytes
	}
	return true
}

// ReferenceSubrackWords writes the words of the idxForSize'th subrack of the
// current anchor into buffer by a plain lookup of its tiles and the
// anchor's playthrough tiles, using nothing CandidateSubracks or
// GetSubrackWords keep, and returns the number of bytes written. For tests:
// it is what those two must agree with.
func (mg *MoveGen) ReferenceSubrackWords(idxForSize int, buffer []byte) int {
	br := mg.nonplaythroughInfos[combinationOffsets[mg.tilesToPlay]+idxForSize].Subrack
	if mg.wordLength > mg.tilesToPlay {
		br.AddBitRack(&mg.playthroughBitRack)
	}
	return mg.wmp.WriteWordsToBuffer(&br, mg.wordLength, buffer)
}

// Words returns the words GetSubrackWords found, WordLength letters each,
// end to end. The slice is the word map's or the move gen's own storage:
// the caller must not write to it, and the next GetSubrackWords call may
// change it.
func (mg *MoveGen) Words() []byte { return mg.words }

// NumWords returns the number of words available after a successful
// GetSubrackWords call.
func (mg *MoveGen) NumWords() int { return mg.numWords }

// WordLength returns the word length the move gen is currently
// retrieving (set by PlaythroughSubracksInit).
func (mg *MoveGen) WordLength() int { return mg.wordLength }

// TilesToPlay returns the number of tiles from the player's rack that
// the move gen is currently retrieving plays for.
func (mg *MoveGen) TilesToPlay() int { return mg.tilesToPlay }

// GetWord returns the wordIdx'th word of Words.
func (mg *MoveGen) GetWord(wordIdx int) []byte {
	start := wordIdx * mg.wordLength
	return mg.words[start : start+mg.wordLength]
}

// GetLeaveValue returns the leave value of the idxForSize'th subrack
// of the current tilesToPlay size, recording it on the MoveGen for
// later retrieval. Mirrors wmp_move_gen_get_leave_value.
func (mg *MoveGen) GetLeaveValue(subrackIdx int) float64 {
	offset := combinationOffsets[mg.tilesToPlay]
	mg.leaveValue = mg.leaveValues[offset+subrackIdx]
	return mg.leaveValue
}

// LeaveValue returns the leave value most recently captured by
// GetLeaveValue.
func (mg *MoveGen) LeaveValue() float64 { return mg.leaveValue }

