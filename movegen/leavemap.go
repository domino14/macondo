package movegen

import (
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/equity"
)

// LeaveMap Implementation
// =======================
// This implements Magpie-style O(1) leave value lookups during move generation.
//
// The key insight is that during move generation, we can track which tiles
// remain on the rack using a compact bit index. Each tile instance on the
// original rack gets one bit. As tiles are played, we clear bits; as they're
// returned during backtracking, we set bits.
//
// Example for rack AEINRST (7 unique tiles, 7 bits):
//   A = bit 0, E = bit 1, I = bit 2, N = bit 3, R = bit 4, S = bit 5, T = bit 6
//   Full rack = 0b1111111 = 127
//   After playing A: 0b1111110 = 126
//   Leave value = leaveMapValues[126]
//
// Example for rack AAEINST (2 A's, still 7 bits total):
//   A gets bits 0-1, E = bit 2, I = bit 3, N = bit 4, S = bit 5, T = bit 6
//   Full rack = 0b1111111 = 127
//   After playing one A: 0b1111110 = 126 (bit 0 cleared)
//   After playing both A's: 0b1111100 = 124 (bits 0,1 cleared)
//
// The leaveMapValues array is pre-populated at the start of GenAll by
// enumerating all 2^n possible subsets and looking up their leave values.

// initLeaveMap initializes the LeaveMap for the given rack.
// This should be called at the start of GenAll when equity calculators are set.
func (gen *GordonGenerator) initLeaveMap(rack *tilemapping.Rack, leaveCalc equity.Leaves) {
	if leaveCalc == nil {
		gen.leaveMapEnabled = false
		return
	}

	// Build base indices from rack contents
	// Each letter with count > 0 gets a contiguous range of bits
	currentBase := 0
	for ml := tilemapping.MachineLetter(0); ml < tilemapping.MachineLetter(len(rack.LetArr)); ml++ {
		count := rack.LetArr[ml]
		if count > 0 {
			gen.leaveMapBaseIndices[ml] = currentBase
			// Build reversed bit map for complement indexing
			for j := 0; j < count; j++ {
				bitIdx := currentBase + count - j - 1
				if currentBase+j < 7 {
					gen.leaveMapReversedBits[currentBase+j] = 1 << bitIdx
				}
			}
			currentBase += count
		}
	}

	if currentBase == 0 || currentBase > 7 {
		// Empty rack or too many tiles (shouldn't happen with 7-tile racks)
		gen.leaveMapEnabled = false
		return
	}

	// Set initial index (all bits set = full rack)
	fullRackIndex := (1 << currentBase) - 1
	gen.leaveMapIndex = fullRackIndex

	// Populate leave values by enumerating all subsets
	// We use a recursive approach similar to Magpie's generate_exchange_moves
	gen.populateLeaveValues(rack, leaveCalc)

	// CRITICAL: Reset index to full rack after enumeration!
	// populateLeaveValues modifies leaveMapIndex during recursion
	gen.leaveMapIndex = fullRackIndex

	gen.leaveMapEnabled = true
}

// populateLeaveValues fills the leaveMapValues array with leave values
// for all possible subsets of the rack.
func (gen *GordonGenerator) populateLeaveValues(rack *tilemapping.Rack, leaveCalc equity.Leaves) {
	// Copy rack into pre-allocated struct to avoid allocation
	gen.leaveMapRackCopy.CopyFrom(rack)

	// Start with complement index = 0 (keeping nothing = full leave = all tiles played)
	// We'll enumerate what we KEEP and store the value for that leave
	gen.leaveMapIndex = 0

	// Initialize all values to 0 (empty leave)
	for i := range gen.leaveMapValues {
		gen.leaveMapValues[i] = 0
	}

	// The leave value for keeping nothing (index 0) is the value of the empty leave
	gen.leaveMapValues[0] = 0

	// Enumerate all possible "kept" subsets recursively using pre-allocated leave array
	gen.enumerateLeavesForMap(&gen.leaveMapRackCopy, leaveCalc, 0, 0)
}

// enumerateLeavesForMap recursively enumerates all possible leaves and stores their values.
// Uses complement indexing: we iterate over what we KEEP.
// leaveLen is the current length of the leave stored in gen.leaveMapLeave.
func (gen *GordonGenerator) enumerateLeavesForMap(rack *tilemapping.Rack, leaveCalc equity.Leaves, startML tilemapping.MachineLetter, leaveLen int) {
	// Record the current leave value at the current complement index
	if leaveLen > 0 {
		value := leaveCalc.LeaveValue(gen.leaveMapLeave[:leaveLen])
		gen.leaveMapValues[gen.leaveMapIndex] = value
	}

	// Try adding more tiles to the leave (keeping more tiles)
	for ml := startML; ml < tilemapping.MachineLetter(len(rack.LetArr)); ml++ {
		count := rack.LetArr[ml]
		for i := 0; i < count; i++ {
			// Take from rack (conceptually "keep" this tile)
			rack.LetArr[ml]--

			// Update complement index - set the bit for this tile
			baseIdx := gen.leaveMapBaseIndices[ml]
			offset := int(rack.LetArr[ml]) // After decrement = position in sequence
			bitIdx := baseIdx + offset
			if bitIdx < 7 {
				reversedBit := gen.leaveMapReversedBits[bitIdx]
				gen.leaveMapIndex |= reversedBit
			}

			// Add to leave (using pre-allocated array)
			gen.leaveMapLeave[leaveLen] = ml

			// Recurse
			gen.enumerateLeavesForMap(rack, leaveCalc, ml, leaveLen+1)

			// Clear the bit (backtrack)
			if bitIdx < 7 {
				reversedBit := gen.leaveMapReversedBits[bitIdx]
				gen.leaveMapIndex &^= reversedBit
			}

			// Return to rack
			rack.LetArr[ml]++
		}
	}
}

// leaveMapTakeTile updates the LeaveMap index when a tile is taken from the rack.
// numberOnRackAfterTake is the count of this letter AFTER the take (rack.LetArr[ml] after rack.Take).
func (gen *GordonGenerator) leaveMapTakeTile(ml tilemapping.MachineLetter, numberOnRackAfterTake int) {
	baseIdx := gen.leaveMapBaseIndices[ml]
	bitIdx := baseIdx + numberOnRackAfterTake
	gen.leaveMapIndex &^= (1 << bitIdx) // Clear the bit
}

// leaveMapReturnTile updates the LeaveMap index when a tile is returned to the rack.
// numberOnRackBeforeReturn is the count of this letter BEFORE the return (rack.LetArr[ml] before rack.Add).
func (gen *GordonGenerator) leaveMapReturnTile(ml tilemapping.MachineLetter, numberOnRackBeforeReturn int) {
	baseIdx := gen.leaveMapBaseIndices[ml]
	bitIdx := baseIdx + numberOnRackBeforeReturn
	gen.leaveMapIndex |= (1 << bitIdx) // Set the bit
}

// leaveMapValue returns the cached leave value for the current rack state.
func (gen *GordonGenerator) leaveMapValue() float64 {
	return gen.leaveMapValues[gen.leaveMapIndex]
}

// LeaveMapEnabled returns whether the LeaveMap is initialized and usable.
func (gen *GordonGenerator) LeaveMapEnabled() bool {
	return gen.leaveMapEnabled
}
