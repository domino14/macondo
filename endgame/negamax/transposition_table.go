package negamax

import (
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"sync/atomic"
	"unsafe"

	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/zobrist"
	"github.com/pbnjay/memory"
	"github.com/rs/zerolog/log"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

const (
	TTExact = 0x01
	TTLower = 0x02
	TTUpper = 0x03
)

const entrySize = 16

const bottom3ByteMask = (1 << 24) - 1
const depthMask = (1 << 6) - 1

// pureBit marks an entry whose search reached only finished games within its
// depth. Such a value holds at that depth and every greater one, but not at a
// smaller one, where some of those lines would be cut off and estimated. It is
// the top bit of the fifth hash byte, so the hash check uses one bit fewer.
const pureBit = 0x80

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
	return uint64(top4bytes)<<32 + uint64(fifthbyte&^pureBit)<<24 + (idx & bottom3ByteMask)
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

// pure reports whether the entry's search reached only finished games.
func (e *TableEntry) pure() bool {
	_, _, fifthbyte, _, _ := e.unpack()
	return fifthbyte&pureBit != 0
}

func (e *TableEntry) getPlay() tinymove.TinyMove {
	_, _, _, _, play := e.unpack()
	return play
}

type TranspositionTable struct {
	// These fields are only read during a search. Nothing a search writes
	// may live beside them: every probe reads them, on every thread, and a
	// counter on the same cache line would have the threads take turns.
	table        []TableEntry
	sizePowerOf2 int
	sizeMask     uint64
	zobrist      *zobrist.Zobrist
	// stats holds the counters, a set per search thread (see statsFor).
	stats *[ttStatStripes]ttStats
	// hugeTouched has a bit for each 2 MB of the table that has been
	// written to, and hugeMisalign is how far into such a page the table
	// starts. Nil unless the kernel agreed to huge pages; see
	// touchHugePage.
	hugeTouched  []uint64
	hugeMisalign uint64
}

// hugePageSize is the size of the huge pages the table asks for.
const hugePageSize = 2 << 20

// touchHugePage sees to it that the first access to each 2 MB of a new
// table is a write.
//
// The table is gigabytes that lookups hit at random. On ordinary 4 KB pages
// nearly every lookup also misses the processor's page-translation cache and
// walks page tables too big to stay in cache themselves; on 2 MB pages the
// translation is almost always at hand. A solve of a few minutes ran 8%
// faster.
//
// Asking the kernel for huge pages (adviseHugePages) is not enough. A page
// of a new table is usually read, by a lookup, before it is written, by a
// store. A read of untouched memory maps the kernel's shared page of zeros,
// and the write that follows is then given ordinary small pages. So before
// a part of the table is first used, a slot in it gets a write that changes
// nothing (an atomic add of zero), which is when the kernel hands over a
// real huge page. It costs a test of one bit per lookup, and no time up
// front: a short solve pays only for the pages it uses.
func (t *TranspositionTable) touchHugePage(idx uint64) {
	if t.hugeTouched == nil {
		return
	}
	page := (idx*entrySize + t.hugeMisalign) / hugePageSize
	word, bit := &t.hugeTouched[page>>6], uint64(1)<<(page&63)
	if atomic.LoadUint64(word)&bit == 0 {
		atomic.AddUint64(&t.table[idx].atomic1, 0)
		atomic.OrUint64(word, bit)
	}
}

// ttStatStripes is the number of counter sets; a thread uses the one at its
// index modulo this.
const ttStatStripes = 256

// ttStats is one thread's counters, padded to cache lines of its own.
type ttStats struct {
	created atomic.Uint64
	lookups atomic.Uint64
	hits    atomic.Uint64
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
	_         [88]byte
}

// statsFor returns the counters a search thread adds to.
func (t *TranspositionTable) statsFor(thread int) *ttStats {
	return &t.stats[thread&(ttStatStripes-1)]
}

// ttTotals is the sum of every thread's counters.
type ttTotals struct {
	created, lookups, hits, t2collisions, tornReads uint64
}

func (t *TranspositionTable) totals() ttTotals {
	var sum ttTotals
	if t.stats == nil {
		return sum
	}
	for i := range t.stats {
		st := &t.stats[i]
		sum.created += st.created.Load()
		sum.lookups += st.lookups.Load()
		sum.hits += st.hits.Load()
		sum.t2collisions += st.t2collisions.Load()
		sum.tornReads += st.tornReads.Load()
	}
	return sum
}

// SetSingleThreadedMode is a no-op (lockless TT works for both single and multi-threaded)
func (t *TranspositionTable) SetSingleThreadedMode() {}

// SetMultiThreadedMode is a no-op (lockless TT works for both single and multi-threaded)
func (t *TranspositionTable) SetMultiThreadedMode() {}

// bucketSize is the number of entries that share a table index. A bucket is
// one cache line (4 x 16 bytes). A lookup searches the bucket, and a store
// replaces the entry for the same position, else an empty slot, else the
// shallowest entry -- so the entry of a leaf, which is cheap to recompute and
// by far the most common kind, never evicts the result of a deep search while
// a shallower entry sits beside it. The table holds as many entries as it
// would unbucketed; only where each may live changes.
const bucketSize = 4

// loadSlot reads one slot; ok is false if a concurrent store tore the read.
func (t *TranspositionTable) loadSlot(idx uint64, st *ttStats) (atomic1, atomic2 uint64, ok bool) {
	atomic1 = atomic.LoadUint64(&t.table[idx].atomic1)
	atomic2 = atomic.LoadUint64(&t.table[idx].atomic2)
	if atomic.LoadUint64(&t.table[idx].atomic1) != atomic1 {
		// Very rare ABA risk: the same value stored twice between the two
		// reads of atomic1 goes unnoticed. See tornReads.
		st.tornReads.Add(1)
		return 0, 0, false
	}
	return atomic1, atomic2, true
}

// bucketHash is what identifies zval within its bucket: the hash without the
// pure bit's position and without the bits that pick the slot in the bucket.
func bucketHash(zval uint64) uint64 {
	return zval &^ (pureBit << 24) &^ (bucketSize - 1)
}

// prefetch starts loading zval's bucket into cache, for a lookup that is
// coming soon. The table is far too large to be in cache, so a lookup
// otherwise waits on memory.
func (t *TranspositionTable) prefetch(zval uint64) {
	if len(t.table) == 0 {
		return
	}
	prefetch(unsafe.Pointer(&t.table[(zval&t.sizeMask)&^(bucketSize-1)]))
}

// lookup returns the entry for zval, or an invalid one. thread is the
// caller's search thread, which only picks the counters it adds to.
func (t *TranspositionTable) lookup(zval uint64, thread int) TableEntry {
	st := t.statsFor(thread)
	st.lookups.Add(1)
	base := (zval & t.sizeMask) &^ (bucketSize - 1)
	t.touchHugePage(base)
	want := bucketHash(zval)
	otherValid := false
	for i := uint64(0); i < bucketSize; i++ {
		atomic1, atomic2, ok := t.loadSlot(base+i, st)
		if !ok {
			continue
		}
		top4bytes, _, fifthbyte, flagAndDepth, _ := unpackTableEntry(atomic1, atomic2)
		if !validFromPacked(flagAndDepth) {
			continue
		}
		if fullHashFromPacked(top4bytes, fifthbyte, base) == want {
			st.hits.Add(1)
			return TableEntry{atomic1: atomic1, atomic2: atomic2}
		}
		otherValid = true
	}
	if otherValid {
		// Only unrelated nodes in this bucket.
		st.t2collisions.Add(1)
	}
	return TableEntry{}
}

func (t *TranspositionTable) store(zval uint64, score int16, flagAndDepth uint8, play tinymove.TinyMove, thread int) {
	t.storeEntry(zval, score, flagAndDepth, play, false, thread)
}

// storeEntry stores an entry; pure marks one whose search reached only
// finished games.
func (t *TranspositionTable) storeEntry(zval uint64, score int16, flagAndDepth uint8, play tinymove.TinyMove, pure bool, thread int) {
	top4bytes := uint32(zval >> 32)
	fifthbyte := uint8(zval>>24) &^ pureBit
	if pure {
		fifthbyte |= pureBit
	}
	st := t.statsFor(thread)
	st.created.Add(1)

	// The slot to replace, in order of preference: this position's, an empty
	// one, the one that cost least to compute.
	base := (zval & t.sizeMask) &^ (bucketSize - 1)
	t.touchHugePage(base)
	want := bucketHash(zval)
	victim := base
	victimRank := math.MaxInt
	for i := uint64(0); i < bucketSize; i++ {
		rank := 0
		if a1, a2, ok := t.loadSlot(base+i, st); ok {
			slotTop4, _, slotFifth, slotFlagAndDepth, _ := unpackTableEntry(a1, a2)
			if validFromPacked(slotFlagAndDepth) {
				if fullHashFromPacked(slotTop4, slotFifth, base) == want {
					victim = base + i
					break
				}
				rank = 1 + int(depthFromPacked(slotFlagAndDepth))
				if slotFifth&pureBit != 0 {
					// A pure entry's subtree reached only finished games,
					// which is cheap to search again, so it goes before any
					// entry that records real search effort.
					rank = 1
				}
			}
		}
		if rank < victimRank {
			victimRank = rank
			victim = base + i
		}
	}

	// Pack into two uint64s for atomic storage
	atomic1, atomic2 := packTableEntry(top4bytes, score, fifthbyte, flagAndDepth, play)

	// Atomic writes (write atomic1 last so validation check works)
	atomic.StoreUint64(&t.table[victim].atomic2, atomic2)
	atomic.StoreUint64(&t.table[victim].atomic1, atomic1)
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
		t.hugeTouched = nil
		if adviseHugePages(t.table) {
			t.hugeMisalign = uint64(uintptr(unsafe.Pointer(&t.table[0])) & (hugePageSize - 1))
			pages := (uint64(numElems)*entrySize+t.hugeMisalign)/hugePageSize + 1
			t.hugeTouched = make([]uint64, (pages+63)/64)
		}
	}

	if t.zobrist == nil || t.zobrist.BoardDim() != boardDim {
		log.Info().Msg("creating zobrist hash")
		t.zobrist = &zobrist.Zobrist{}
		t.zobrist.Initialize(boardDim)
	}

	p := message.NewPrinter(language.English)
	log.Info().
		Str("allocated-elems", p.Sprintf("%d", numElems)).
		Str("desired-elems", p.Sprintf("%.0f", desiredNElems)).
		Str("allocated-bytes", p.Sprintf("%d", numElems*entrySize)).
		Str("memory-budget", p.Sprintf("%d", totalMem)).
		Bool("reset", reset).
		Msg("transposition-table-size")

	t.ResetStats()
}

func (t *TranspositionTable) Zobrist() *zobrist.Zobrist {
	return t.zobrist
}

func (t *TranspositionTable) SetZobrist(z *zobrist.Zobrist) {
	t.zobrist = z
}

func (t *TranspositionTable) Stats() string {
	sum := t.totals()
	return fmt.Sprintf("created: %d lookups: %d hits: %d t2collisions: %d tornReads: %d",
		sum.created, sum.lookups, sum.hits, sum.t2collisions, sum.tornReads)
}

// Created returns the number of TT entries created
func (t *TranspositionTable) Created() uint64 {
	return t.totals().created
}

// Lookups returns the number of TT lookups performed
func (t *TranspositionTable) Lookups() uint64 {
	return t.totals().lookups
}

// Hits returns the number of TT hits
func (t *TranspositionTable) Hits() uint64 {
	return t.totals().hits
}

// ResetStats resets the statistics counters without reallocating the table
func (t *TranspositionTable) ResetStats() {
	t.stats = new([ttStatStripes]ttStats)
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
