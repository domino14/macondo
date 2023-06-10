package negamax

import (
	"math"
	"runtime"

	"github.com/pbnjay/memory"
	"github.com/rs/zerolog/log"
)

const (
	TTExact = 0x01
	TTLower = 0x02
	TTUpper = 0x04
)

const entrySize = 14

var ttSizePowerOf2 int
var ttSizeMask int

// 14 bytes (entrySize)
type TableEntry struct {
	fullHash uint64
	score    float32
	flag     uint8
	depth    uint8
}

type TranspositionTable struct {
	table   []TableEntry
	created uint64
	lookups uint64
	hits    uint64
	// "type 2" collisions. A type 2 collision happens when two positions share
	// the same lower bytes. A type 1 collision happens when two positions share the
	// same overall hash. We don't have a super easy way to detect the latter,
	// but it should be much less common.
	t2collisions uint64
}

var globalTranspositionTable TranspositionTable

// var globalTranspositionTable DebugTranspositionTable

func init() {
	totalMem := memory.TotalMemory()
	desired := float64(totalMem) / float64(1.5) / float64(entrySize)
	// find biggest power of 2 lower than desired.
	ttSizePowerOf2 = int(math.Log2(desired))
	ttSizeMask = ((1 << ttSizePowerOf2) - 1)
}

func (t *TranspositionTable) lookup(zval uint64) *TableEntry {
	t.lookups++
	idx := zval & uint64(ttSizeMask)
	if t.table[idx].fullHash != zval {
		if t.table[idx].fullHash != 0 {
			// uninitialized. could also be 0 but there's a 1/(2^64) chance of that.
			// fmt.Println("type 2 collision", t.table[idx].fullHash, zval, ttSizeMask, zval&uint64(ttSizeMask))
			t.t2collisions++
		}
		return nil
	}
	t.hits++
	// otherwise, assume the same zobrist hash is the same position. this fails
	// very, very rarely. but it could happen.
	return &t.table[idx]
}

func (t *TranspositionTable) store(zval uint64, tentry TableEntry) {
	idx := zval & uint64(ttSizeMask)
	// just overwrite whatever is there for now.
	t.table[idx] = tentry
	t.created++
}

func (t *TranspositionTable) reset() {
	numElems := int(math.Pow(2, float64(ttSizePowerOf2)))

	log.Info().Int("num-elems", numElems).
		Int("estimated-total-memory-bytes", numElems*entrySize).
		Msg("transposition-table-size")
	t.table = nil
	runtime.GC() // ?

	t.table = make([]TableEntry, numElems)
	t.created = 0
	log.Info().Msg("allocated-transposition-table")
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
