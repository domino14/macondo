package wmp

import (
	"math"

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
	MaxWMPMoveGenAnchors = (RackSize + 1) * MaxPossiblePlaythroughBlocks
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

// SubrackInfo holds metadata about one enumerated subrack: the
// multiset of tiles, the looked-up WMP entry (or nil), and the leave
// value at the time of enumeration. Mirrors MAGPIE's SubrackInfo.
type SubrackInfo struct {
	Subrack    BitRack
	WMPEntry   *Entry
	LeaveValue float64
}

// Anchor is the per-(playthrough_blocks, tiles_to_play) anchor record
// the WMP move generator builds during shadow play. Mirrors MAGPIE's
// Anchor (the WMP-relevant subset of fields). Bit-packed in C; we use
// plain ints in Go.
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
// Anchor management (MaybeUpdateAnchor) is done as the move generator
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

	// nonplaythroughBestLeaveValues[leaveSize] = best leave value
	// across all subracks of size (fullRackSize-leaveSize) that have
	// at least one valid word. Computed only when checkLeaves=true.
	nonplaythroughBestLeaveValues [RackSize + 1]float64
	// nonplaythroughHasWordOfLength[length] = true iff at least one
	// blankless subrack of that size matches a real word.
	nonplaythroughHasWordOfLength [RackSize + 1]bool
	countBySize                   [RackSize + 1]uint8

	anchors [MaxWMPMoveGenAnchors]Anchor
	// dirtyAnchors holds the indices into anchors that were touched
	// since the last ResetAnchors call. This lets ResetAnchors avoid
	// re-zeroing the full 64-entry table on every shadow square,
	// which was the biggest hot spot in the profiler.
	dirtyAnchors []int

	// Scratch buffer used by GetSubrackWords for write_words_to_buffer.
	buffer      [ResultBufferSize]byte
	tilesToPlay int
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
	mg.playerBitRack = BitRackFromRack(rack)
	mg.fullRackSize = int(rack.NumTiles())
	for i := range mg.nonplaythroughHasWordOfLength {
		mg.nonplaythroughHasWordOfLength[i] = false
	}
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
// last reset (tracked via dirtyAnchors). MaybeUpdateAnchor records
// each newly-touched slot, and MaybeUpdateAnchor's "first touch"
// branch is responsible for installing the per-slot sentinels (so
// ResetAnchors itself can leave them implicit).
func (mg *MoveGen) ResetAnchors() {
	for _, idx := range mg.dirtyAnchors {
		mg.anchors[idx] = Anchor{}
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

// Anchors returns the per-(playthrough_blocks, tiles_to_play) anchor
// table. Slots with TilesToPlay == 0 are unused. The caller is
// expected to walk the slice and push the live anchors into its own
// anchor heap (mirroring wmp_move_gen_add_anchors).
//
// Most callers should prefer LiveAnchors, which iterates only the
// slots that were actually touched since the last ResetAnchors call.
func (mg *MoveGen) Anchors() []Anchor {
	return mg.anchors[:]
}

// DirtyAnchorIndices returns the indices into the per-(blocks, tiles)
// anchor table that were touched since the last ResetAnchors call.
// The caller can index into Anchors() (or use AnchorAt) to read each
// touched slot. Avoids copying anchor structs.
func (mg *MoveGen) DirtyAnchorIndices() []int {
	return mg.dirtyAnchors
}

// AnchorAt returns a pointer to the anchor slot at the given index
// into the per-(blocks, tiles) table. Pair with DirtyAnchorIndices
// to walk only the live entries.
func (mg *MoveGen) AnchorAt(idx int) *Anchor {
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
func (mg *MoveGen) enumerateNonplaythroughSubracks(current *BitRack, nextML int, count int, lm *leavemap.LeaveMap) {
	maxNumThis := 0
	for ; nextML < maxAlphabetSize; nextML++ {
		maxNumThis = mg.playerBitRack.GetLetter(byte(nextML))
		if maxNumThis > 0 {
			break
		}
	}
	if nextML >= maxAlphabetSize {
		insertIndex := combinationOffsets[count] + int(mg.countBySize[count])
		info := &mg.nonplaythroughInfos[insertIndex]
		info.Subrack = *current
		info.LeaveValue = lm.CurrentValue()
		mg.countBySize[count]++
		return
	}
	for i := 0; i < maxNumThis; i++ {
		mg.enumerateNonplaythroughSubracks(current, nextML+1, count+i, lm)
		current.AddLetter(byte(nextML))
		lm.ComplementAddLetter(tilemapping.MachineLetter(nextML), i)
	}
	mg.enumerateNonplaythroughSubracks(current, nextML+1, count+maxNumThis, lm)
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
		if info.LeaveValue > mg.nonplaythroughBestLeaveValues[leaveSize] {
			mg.nonplaythroughBestLeaveValues[leaveSize] = info.LeaveValue
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
	for i := range mg.countBySize {
		mg.countBySize[i] = 0
	}
	for i := range mg.nonplaythroughHasWordOfLength {
		mg.nonplaythroughHasWordOfLength[i] = false
	}
	var empty BitRack
	mg.enumerateNonplaythroughSubracks(&empty, blankMachineLetter, 0, lm)
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

// GetAnchor returns a pointer to the anchor slot for the given
// (playthrough_blocks, tiles_played) pair. Mirrors
// wmp_move_gen_get_anchor.
func (mg *MoveGen) GetAnchor(playthroughBlocks, tilesPlayed int) *Anchor {
	return &mg.anchors[wmpMoveGenAnchorIndex(playthroughBlocks, tilesPlayed)]
}

// MaybeUpdateAnchor updates the anchor slot for the current
// (playthrough_blocks, tiles_played) pair, expanding the recorded
// column range and pulling up the best score / equity if appropriate.
// Mirrors wmp_move_gen_maybe_update_anchor.
//
// On first touch (TilesToPlay == 0), the slot is freshly initialized
// from the incoming values and its index is appended to dirtyAnchors
// so the next ResetAnchors knows to clear it.
func (mg *MoveGen) MaybeUpdateAnchor(tilesPlayed, wordLength, startCol int, score, equity float64) {
	idx := wmpMoveGenAnchorIndex(mg.playthroughBlocks, tilesPlayed)
	anchor := &mg.anchors[idx]
	if anchor.TilesToPlay == 0 {
		anchor.TilesToPlay = tilesPlayed
		anchor.PlaythroughBlocks = mg.playthroughBlocks
		anchor.WordLength = wordLength
		anchor.LeftmostStartCol = startCol
		anchor.RightmostStartCol = startCol
		anchor.HighestPossibleEquity = equity
		anchor.HighestPossibleScore = score
		mg.dirtyAnchors = append(mg.dirtyAnchors, idx)
		return
	}
	anchor.WordLength = wordLength
	if startCol < anchor.LeftmostStartCol {
		anchor.LeftmostStartCol = startCol
	}
	if startCol > anchor.RightmostStartCol {
		anchor.RightmostStartCol = startCol
	}
	if equity > anchor.HighestPossibleEquity {
		anchor.HighestPossibleEquity = equity
	}
	if score > anchor.HighestPossibleScore {
		anchor.HighestPossibleScore = score
	}
}

// =====================================================================
// Word retrieval (after an anchor is selected)
// =====================================================================

// PlaythroughSubracksInit prepares the playthrough subrack array for
// word retrieval against the given anchor. If the anchor has no
// playthrough tiles the nonplaythrough subracks are reused. Mirrors
// wmp_move_gen_playthrough_subracks_init.
func (mg *MoveGen) PlaythroughSubracksInit(anchor *Anchor) {
	subrackSize := anchor.TilesToPlay
	mg.wordLength = anchor.WordLength
	mg.numTilesPlayedThrough = anchor.WordLength - subrackSize
	mg.tilesToPlay = subrackSize
	if mg.numTilesPlayedThrough == 0 {
		// Use the nonplaythrough subracks directly.
		return
	}
	offset := combinationOffsets[subrackSize]
	count := int(mg.countBySize[subrackSize])
	for idxForSize := 0; idxForSize < count; idxForSize++ {
		nonpt := &mg.nonplaythroughInfos[offset+idxForSize]
		pt := &mg.playthroughInfos[offset+idxForSize]
		pt.Subrack = nonpt.Subrack
		pt.Subrack.AddBitRack(&mg.playthroughBitRack)
	}
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

// GetSubrackWords looks up the idxForSize'th subrack in the WMP and
// expands its words into the internal scratch buffer. Returns false
// if no words exist for that subrack. After a true return, GetWord
// retrieves individual words by index. Mirrors
// wmp_move_gen_get_subrack_words.
func (mg *MoveGen) GetSubrackWords(idxForSize int) bool {
	offset := combinationOffsets[mg.tilesToPlay]
	subrackIdx := offset + idxForSize
	isPlaythrough := mg.wordLength > mg.tilesToPlay
	var info *SubrackInfo
	if isPlaythrough {
		info = &mg.playthroughInfos[subrackIdx]
	} else {
		info = &mg.nonplaythroughInfos[subrackIdx]
	}
	// Nonplaythrough subracks were already looked up by
	// CheckNonplaythroughExistence; playthrough subracks are looked
	// up lazily here.
	if isPlaythrough {
		info.WMPEntry = mg.wmp.GetWordEntry(&info.Subrack, mg.wordLength)
	}
	if info.WMPEntry == nil {
		return false
	}
	resultBytes := mg.wmp.WriteWordsForEntry(info.WMPEntry, &info.Subrack, mg.wordLength, mg.buffer[:])
	if resultBytes <= 0 || resultBytes%mg.wordLength != 0 {
		return false
	}
	mg.numWords = resultBytes / mg.wordLength
	return true
}

// NumWords returns the number of words available after a successful
// GetSubrackWords call.
func (mg *MoveGen) NumWords() int { return mg.numWords }

// WordLength returns the word length the move gen is currently
// retrieving (set by PlaythroughSubracksInit).
func (mg *MoveGen) WordLength() int { return mg.wordLength }

// TilesToPlay returns the number of tiles from the player's rack that
// the move gen is currently retrieving plays for.
func (mg *MoveGen) TilesToPlay() int { return mg.tilesToPlay }

// GetWord returns the wordIdx'th word in the current scratch buffer
// as a slice of MachineLetters. The slice aliases the internal buffer
// and is invalidated by the next GetSubrackWords call.
func (mg *MoveGen) GetWord(wordIdx int) []byte {
	start := wordIdx * mg.wordLength
	return mg.buffer[start : start+mg.wordLength]
}

// GetLeaveValue returns the leave value of the idxForSize'th subrack
// of the current tilesToPlay size, recording it on the MoveGen for
// later retrieval. Mirrors wmp_move_gen_get_leave_value.
func (mg *MoveGen) GetLeaveValue(subrackIdx int) float64 {
	offset := combinationOffsets[mg.tilesToPlay]
	mg.leaveValue = mg.nonplaythroughInfos[offset+subrackIdx].LeaveValue
	return mg.leaveValue
}

// LeaveValue returns the leave value most recently captured by
// GetLeaveValue.
func (mg *MoveGen) LeaveValue() float64 { return mg.leaveValue }

