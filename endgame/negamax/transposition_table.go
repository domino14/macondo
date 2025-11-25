package negamax

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"sync/atomic"

	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/zobrist"
	"github.com/pbnjay/memory"
	"github.com/rs/zerolog/log"
)

const (
	TTExact = 0x01
	TTLower = 0x02
	TTUpper = 0x03
)

const entrySize = 16

const bottom3ByteMask = (1 << 24) - 1
const depthMask = (1 << 6) - 1

// 16 bytes (entrySize) - lockless version using two atomic uint64s
type TableEntry struct {
	// atomic1: packed top4bytes(32) | score(16) | fifthbyte(8) | flagAndDepth(8)
	// atomic2: play (TinyMove is uint64)
	atomic1 uint64
	atomic2 uint64
}

// packTableEntry packs a table entry into two uint64s for atomic storage
func packTableEntry(top4bytes uint32, score int16, fifthbyte uint8, flagAndDepth uint8, play tinymove.TinyMove) (uint64, uint64) {
	atomic1 := uint64(top4bytes)<<32 | uint64(uint16(score))<<16 | uint64(fifthbyte)<<8 | uint64(flagAndDepth)
	atomic2 := uint64(play)
	return atomic1, atomic2
}

// unpackTableEntry unpacks two uint64s into a temporary struct for access
func unpackTableEntry(atomic1, atomic2 uint64) (top4bytes uint32, score int16, fifthbyte uint8, flagAndDepth uint8, play tinymove.TinyMove) {
	top4bytes = uint32(atomic1 >> 32)
	score = int16(uint16(atomic1 >> 16))
	fifthbyte = uint8(atomic1 >> 8)
	flagAndDepth = uint8(atomic1)
	play = tinymove.TinyMove(atomic2)
	return
}

// fullHash calculates the full 64-bit hash for this table entry
func fullHashFromPacked(top4bytes uint32, fifthbyte uint8, idx uint64) uint64 {
	return uint64(top4bytes)<<32 + uint64(fifthbyte)<<24 + (idx & bottom3ByteMask)
}

// Helper functions for unpacked data
func flagFromPacked(flagAndDepth uint8) uint8 {
	return flagAndDepth >> 6
}

func depthFromPacked(flagAndDepth uint8) uint8 {
	return flagAndDepth & depthMask
}

func validFromPacked(flagAndDepth uint8) bool {
	return flagFromPacked(flagAndDepth) != 0
}

// TableEntry helper methods for accessing packed data
func (e *TableEntry) unpack() (top4bytes uint32, score int16, fifthbyte uint8, flagAndDepth uint8, play tinymove.TinyMove) {
	atomic1 := atomic.LoadUint64(&e.atomic1)
	atomic2 := atomic.LoadUint64(&e.atomic2)
	return unpackTableEntry(atomic1, atomic2)
}

func (e *TableEntry) valid() bool {
	_, _, _, flagAndDepth, _ := e.unpack()
	return validFromPacked(flagAndDepth)
}

func (e *TableEntry) depth() uint8 {
	_, _, _, flagAndDepth, _ := e.unpack()
	return depthFromPacked(flagAndDepth)
}

func (e *TableEntry) flag() uint8 {
	_, _, _, flagAndDepth, _ := e.unpack()
	return flagFromPacked(flagAndDepth)
}

func (e *TableEntry) getScore() int16 {
	_, score, _, _, _ := e.unpack()
	return score
}

func (e *TableEntry) getTop4bytes() uint32 {
	top4bytes, _, _, _, _ := e.unpack()
	return top4bytes
}

func (e *TableEntry) getFifthbyte() uint8 {
	_, _, fifthbyte, _, _ := e.unpack()
	return fifthbyte
}

func (e *TableEntry) getPlay() tinymove.TinyMove {
	_, _, _, _, play := e.unpack()
	return play
}

type TranspositionTable struct {
	table        []TableEntry
	created      atomic.Uint64
	lookups      atomic.Uint64
	hits         atomic.Uint64
	sizePowerOf2 int
	sizeMask     uint64
	// "type 2" collisions. A type 2 collision happens when two positions share
	// the same lower bytes. A type 1 collision happens when two positions share the
	// same overall hash. We don't have a super easy way to detect the latter,
	// but it should be much less common.
	t2collisions atomic.Uint64
	// tornReads tracks how many lookups detected concurrent writes via validation check.
	// Very rare ABA risk: if the same position appears at the same slot between our two
	// atomic1 reads, we may accept corrupted data. Probability is negligible (same position
	// must hash to same slot within nanoseconds) and impact is minor (one bad cache entry).
	tornReads atomic.Uint64

	zobrist *zobrist.Zobrist
}

// GlobalTranspositionTable is a singleton instance. Since transposition tables
// take up a large enough amount of memory, and they're meant to be shared,
// we only really want to keep one in memory to avoid re-allocation costs.
var GlobalTranspositionTable = &TranspositionTable{}

// SetSingleThreadedMode is a no-op (lockless TT works for both single and multi-threaded)
func (t *TranspositionTable) SetSingleThreadedMode() {}

// SetMultiThreadedMode is a no-op (lockless TT works for both single and multi-threaded)
func (t *TranspositionTable) SetMultiThreadedMode() {}

func (t *TranspositionTable) lookup(zval uint64) TableEntry {
	t.lookups.Add(1)
	idx := zval & t.sizeMask

	// Validated atomic read: detect torn reads from concurrent writes
	atomic1Before := atomic.LoadUint64(&t.table[idx].atomic1)
	atomic2 := atomic.LoadUint64(&t.table[idx].atomic2)
	atomic1After := atomic.LoadUint64(&t.table[idx].atomic1)

	if atomic1Before != atomic1After {
		// Torn read detected - concurrent write happened between our reads
		t.tornReads.Add(1)
		return TableEntry{}
	}

	// Unpack and validate hash
	top4bytes, _, fifthbyte, flagAndDepth, _ := unpackTableEntry(atomic1Before, atomic2)
	fullHash := fullHashFromPacked(top4bytes, fifthbyte, idx)
	if fullHash != zval {
		if validFromPacked(flagAndDepth) {
			// There is another unrelated node at this position.
			t.t2collisions.Add(1)
		}
		return TableEntry{}
	}

	t.hits.Add(1)
	// Return as a "virtual" TableEntry for the caller to read
	// Pack it back into the atomic format
	result := TableEntry{}
	result.atomic1 = atomic1Before
	result.atomic2 = atomic2
	return result
}

func (t *TranspositionTable) store(zval uint64, score int16, flagAndDepth uint8, play tinymove.TinyMove) {
	idx := zval & t.sizeMask
	top4bytes := uint32(zval >> 32)
	fifthbyte := uint8(zval >> 24)
	t.created.Add(1)

	// Pack into two uint64s for atomic storage
	atomic1, atomic2 := packTableEntry(top4bytes, score, fifthbyte, flagAndDepth, play)

	// Atomic writes (write atomic1 last so validation check works)
	atomic.StoreUint64(&t.table[idx].atomic2, atomic2)
	atomic.StoreUint64(&t.table[idx].atomic1, atomic1)
}

func (t *TranspositionTable) Reset(fractionOfMemory float64, boardDim int) {
	// Get memory limit, if set.
	memLimit := debug.SetMemoryLimit(-1)
	totalMem := memory.TotalMemory()
	if memLimit != math.MaxInt64 {
		// Let's obey the set value as the memory limit.
		totalMem = uint64(memLimit)
	}
	desiredNElems := fractionOfMemory * (float64(totalMem) / float64(entrySize))
	// find biggest power of 2 lower than desired.
	t.sizePowerOf2 = int(math.Log2(desiredNElems))
	// Guarantee at least 2^24 elements in the table. Anything less and our
	// 5-byte full hash proxy won't work.
	if t.sizePowerOf2 < 24 {
		t.sizePowerOf2 = 24
	}

	numElems := 1 << t.sizePowerOf2
	t.sizeMask = uint64(numElems - 1)
	reset := false
	if t.table != nil && len(t.table) == numElems {
		reset = true
		clear(t.table)
	} else {
		t.table = make([]TableEntry, numElems)
	}

	if t.zobrist == nil || t.zobrist.BoardDim() != boardDim {
		log.Info().Msg("creating zobrist hash")
		t.zobrist = &zobrist.Zobrist{}
		t.zobrist.Initialize(boardDim)
		zd, err := os.Create("/tmp/macondo-zobrist-dump")
		if err != nil {
			log.Err(err).Msg("could not dump zobrist hashes to file")
		} else {
			t.zobrist.Dump(zd)
			zd.Close()
		}
	}

	log.Info().Int("num-elems", numElems).
		Float64("desired-num-elems", desiredNElems).
		Int("estimated-total-memory-bytes", numElems*entrySize).
		Uint64("mem-limit", totalMem).
		Bool("reset", reset).
		Msg("transposition-table-size")

	t.created.Store(0)
	t.lookups.Store(0)
	t.hits.Store(0)
	t.t2collisions.Store(0)
	t.tornReads.Store(0)
}

func (t *TranspositionTable) Zobrist() *zobrist.Zobrist {
	return t.zobrist
}

func (t *TranspositionTable) SetZobrist(z *zobrist.Zobrist) {
	t.zobrist = z
}

func (t *TranspositionTable) Stats() string {
	return fmt.Sprintf("created: %d lookups: %d hits: %d t2collisions: %d tornReads: %d",
		t.created.Load(), t.lookups.Load(), t.hits.Load(), t.t2collisions.Load(), t.tornReads.Load())
}

// a debug tt

type DebugTableEntry struct {
	score float32
	flag  uint8
	depth uint8
}

type DebugTranspositionTable struct {
	table   map[string]*DebugTableEntry
	created uint64
	lookups uint64
	hits    uint64
}

func (t *DebugTranspositionTable) lookup(cgp string) *DebugTableEntry {
	t.lookups++
	entry := t.table[cgp]
	if entry != nil {
		t.hits++
	}
	return entry
}

func (t *DebugTranspositionTable) store(cgp string, tentry DebugTableEntry) {
	// just overwrite whatever is there for now.
	t.table[cgp] = &tentry
	t.created++
}

func (t *DebugTranspositionTable) reset() {
	t.table = nil
	runtime.GC() // ?
	t.table = make(map[string]*DebugTableEntry)
	t.created = 0
	log.Info().Msg("allocated-debug-transposition-table")
}
