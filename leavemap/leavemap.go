// Package leavemap provides an O(1) leave-value lookup table indexed
// by a per-tile bitmask. As tiles are played and unplayed during move
// generation, the bitmask is updated with a single AND/OR and the
// leave value is read directly from a fixed-size array.
//
// The structure mirrors MAGPIE's LeaveMap (src/ent/leave_map.h) and
// supports both update conventions in a single type:
//
//   - Regular methods (AddLetter / TakeLetter) flip bits in ASCENDING
//     order: the n'th copy of a letter sets / clears the bit at
//     position base+n. This matches the convention used by macondo's
//     normal recursiveGen / populateLeaveMap path.
//
//   - Complement methods (ComplementAddLetter / ComplementTakeLetter)
//     flip bits in DESCENDING order using a precomputed reversed-bit
//     table. They are the inverse of "moving a tile from the played
//     subset back into the leave," and are used by the WMP move
//     generator's nonplaythrough subrack enumeration where the
//     iteration counter naturally goes 0..numthis-1 but the bits need
//     to be touched highest-first for consistency with the leave-map
//     populator.
//
// Both modes share the same letterBaseIndex layout, so a single
// LeaveMap instance can be populated by one mode and consumed by the
// other as long as the convention pair is consistent.
package leavemap

import "github.com/domino14/word-golib/tilemapping"

// MaxRackSize is the largest rack the leave map will index. With 7
// tiles per rack the bitmask fits in 7 bits and the values array has
// 128 entries. Matches game.RackTileLimit; duplicated here so this
// package has no dependency on macondo/game.
const MaxRackSize = 7

// MaxValues is the size of the values array (2^MaxRackSize).
const MaxValues = 1 << MaxRackSize

// LeaveMap stores per-subset leave values keyed by a tile-bitmask.
//
// Field exposure: callers (in particular existing macondo movegen
// code and leave_map_test.go) read and write Values, CurrentIndex,
// TotalTiles, and Initialized directly. The methods are the preferred
// API; the fields are exported for backward compatibility with the
// pre-package macondo/movegen.leaveMap layout.
type LeaveMap struct {
	// Values holds the leave value for every reachable bitmask.
	Values [MaxValues]float64
	// LetterBaseIndex maps a MachineLetter to the base bit position
	// of its tiles in the bitmask. Each letter occupies count
	// consecutive bits starting at base.
	LetterBaseIndex [tilemapping.MaxAlphabetSize + 1]int
	// ReversedLetterBitMap is a precomputed table used by the
	// complement update mode. The j'th-from-bottom occupied bit of a
	// letter (j = 0..count-1) maps to the bit at the OPPOSITE end of
	// that letter's slot, so iterating j ascending touches bits
	// descending.
	ReversedLetterBitMap [MaxRackSize]int
	// CurrentIndex is the current bitmask. With Initialized=true and
	// no plays, all TotalTiles low bits are set.
	CurrentIndex int
	// TotalTiles is the number of tiles initially on the rack.
	TotalTiles int
	// Initialized is true after a successful Init call.
	Initialized bool
}

// Init assigns each tile in rack a unique bit position and sets
// CurrentIndex to all-bits-set (full rack = empty leave). Mirrors
// MAGPIE's leave_map_init.
//
// Init clears LetterBaseIndex and ReversedLetterBitMap before
// (re-)populating them so the same LeaveMap can be reused across
// multiple racks. If the rack has more than MaxRackSize tiles or is
// empty, Initialized is left false.
func (lm *LeaveMap) Init(rack *tilemapping.Rack) {
	lm.Initialized = false
	for i := range lm.LetterBaseIndex {
		lm.LetterBaseIndex[i] = 0
	}
	for i := range lm.ReversedLetterBitMap {
		lm.ReversedLetterBitMap[i] = 0
	}
	currentBase := 0
	for ml := 0; ml < len(rack.LetArr); ml++ {
		count := rack.LetArr[ml]
		if count <= 0 {
			continue
		}
		lm.LetterBaseIndex[ml] = currentBase
		// Build the reversed-bit table for the complement mode:
		// ReversedLetterBitMap[base + j] = bit at position
		// base + (count - 1 - j), so iterating j 0..count-1 in the
		// caller flips bits at base+(count-1), base+(count-2), ..., base.
		for j := 0; j < count; j++ {
			bitIdx := currentBase + count - j - 1
			lm.ReversedLetterBitMap[currentBase+j] = 1 << bitIdx
		}
		currentBase += count
	}
	lm.TotalTiles = currentBase
	if lm.TotalTiles == 0 || lm.TotalTiles > MaxRackSize {
		return
	}
	lm.CurrentIndex = (1 << currentBase) - 1
	lm.Initialized = true
}

// SetCurrentIndex overwrites the current bitmask. Mirrors
// leave_map_set_current_index.
func (lm *LeaveMap) SetCurrentIndex(idx int) { lm.CurrentIndex = idx }

// SetCurrentValue stores value at the current index. Mirrors
// leave_map_set_current_value.
func (lm *LeaveMap) SetCurrentValue(value float64) {
	lm.Values[lm.CurrentIndex] = value
}

// CurrentValue returns the value at the current index. Mirrors
// leave_map_get_current_value.
func (lm *LeaveMap) CurrentValue() float64 {
	return lm.Values[lm.CurrentIndex]
}

// AddLetter sets the bit for the (numberOnRack)'th copy of letter ml
// in the bitmask, in ASCENDING order: bit at LetterBaseIndex[ml] +
// numberOnRack. This is the regular update mode used by macondo's
// normal recursiveGen / populateLeaveMap path. numberOnRack is the
// count BEFORE the add (i.e., the new copy goes into slot
// numberOnRack). Mirrors leave_map_add_letter.
func (lm *LeaveMap) AddLetter(ml tilemapping.MachineLetter, numberOnRack int) {
	base := lm.LetterBaseIndex[ml]
	lm.CurrentIndex |= 1 << (base + numberOnRack)
}

// TakeLetter clears the bit for the (numberOnRack)'th copy of letter
// ml. numberOnRack is the count AFTER the take. Mirrors
// leave_map_take_letter.
func (lm *LeaveMap) TakeLetter(ml tilemapping.MachineLetter, numberOnRack int) {
	base := lm.LetterBaseIndex[ml]
	lm.CurrentIndex &^= 1 << (base + numberOnRack)
}

// ComplementAddLetter clears the bit for letter ml in the bitmask
// using the REVERSED-bit table. Used by the WMP move generator's
// nonplaythrough subrack enumeration where the caller passes a 0..n-1
// iteration counter but expects bits to be touched highest-first.
// Mirrors leave_map_complement_add_letter.
func (lm *LeaveMap) ComplementAddLetter(ml tilemapping.MachineLetter, numberOnRack int) {
	base := lm.LetterBaseIndex[ml]
	bitIdx := base + numberOnRack
	lm.CurrentIndex &^= lm.ReversedLetterBitMap[bitIdx]
}

// ComplementTakeLetter sets the same bit that ComplementAddLetter
// would clear, restoring the leave on the way out of an enumeration.
// Mirrors leave_map_complement_take_letter.
func (lm *LeaveMap) ComplementTakeLetter(ml tilemapping.MachineLetter, numberOnRack int) {
	base := lm.LetterBaseIndex[ml]
	bitIdx := base + numberOnRack
	lm.CurrentIndex |= lm.ReversedLetterBitMap[bitIdx]
}
