package negamax

import (
	"sync/atomic"

	"github.com/domino14/macondo/tinymove"
)

const (
	ABDADA_SIZE = 32768 // 2^15
	ABDADA_WAYS = 4     // 4-way associative
	DEFER_DEPTH = 3     // Only defer moves at depth >= 3
)

// ABDADAEntry represents a move currently being searched
type ABDADAEntry struct {
	hash  atomic.Uint64 // Move hash
	depth atomic.Int32  // Depth at which it's being searched
}

// ABDADATable tracks moves currently being searched by threads
type ABDADATable struct {
	entries [ABDADA_SIZE][ABDADA_WAYS]ABDADAEntry
}

// NewABDADATable creates a new ABDADA table
func NewABDADATable() *ABDADATable {
	return &ABDADATable{}
}

// moveHash creates a hash for a move at a given position
// Using a simple hash function as suggested in the pseudocode
func (a *ABDADATable) moveHash(posKey uint64, move tinymove.SmallMove) uint64 {
	// Combine position key and move using multiplication and XOR
	// This spreads the bits evenly across the hash space
	tm := move.TinyMove()
	return posKey*1103515245 + uint64(tm)*12345
}

// deferMove checks if a move is already being searched at sufficient depth
// Returns true if the move should be deferred (someone else is searching it)
func (a *ABDADATable) deferMove(posKey uint64, move tinymove.SmallMove, depth int) bool {
	if depth < DEFER_DEPTH {
		return false // Don't defer shallow searches
	}

	hash := a.moveHash(posKey, move)
	index := hash % ABDADA_SIZE

	// Check all ways in this set
	for way := 0; way < ABDADA_WAYS; way++ {
		entry := &a.entries[index][way]
		entryHash := entry.hash.Load()
		entryDepth := entry.depth.Load()

		if entryHash == hash && entryDepth >= int32(depth) {
			// This move is being searched by another thread at >= depth
			return true
		}
	}

	return false
}

// startingSearch registers that we're about to search a move
func (a *ABDADATable) startingSearch(posKey uint64, move tinymove.SmallMove, depth int) {
	hash := a.moveHash(posKey, move)
	index := hash % ABDADA_SIZE

	// Find an empty slot or replace the oldest entry
	// We don't need perfect accuracy here - occasional collisions are OK
	for way := 0; way < ABDADA_WAYS; way++ {
		entry := &a.entries[index][way]
		oldHash := entry.hash.Load()

		// Try to claim an empty slot
		if oldHash == 0 {
			if entry.hash.CompareAndSwap(0, hash) {
				entry.depth.Store(int32(depth))
				return
			}
		}
	}

	// No empty slot, just overwrite the first one
	// This is fine - ABDADA is designed to be lockless and tolerate collisions
	a.entries[index][0].hash.Store(hash)
	a.entries[index][0].depth.Store(int32(depth))
}

// finishedSearch removes a move from the table after searching
func (a *ABDADATable) finishedSearch(posKey uint64, move tinymove.SmallMove) {
	hash := a.moveHash(posKey, move)
	index := hash % ABDADA_SIZE

	// Clear all matching entries
	for way := 0; way < ABDADA_WAYS; way++ {
		entry := &a.entries[index][way]
		if entry.hash.Load() == hash {
			entry.hash.Store(0)
			entry.depth.Store(0)
		}
	}
}
