package negamax

import (
	"testing"

	"github.com/matryer/is"
)

func TestTTableEntry(t *testing.T) {
	is := is.New(t)
	tt := &TranspositionTable{}
	tt.SetSingleThreadedMode()
	// Assure minimum size of 2<<24 elems
	tt.Reset(0, 15)
	tt.store(9409641586937047728, 12, 128+64+23, 0, 0)

	// arrayLength := len(tt.table)
	is.True(tt.sizePowerOf2 >= 24)

	te := tt.lookup(9409641586937047728, 0)
	is.True(te.valid())
	is.Equal(te.depth(), uint8(23))
	is.Equal(te.flag(), uint8(TTUpper))
	is.Equal(te.getScore(), int16(12))
	is.Equal(te.getTop4bytes(), uint32(2190852907))
	is.Equal(te.getFifthbyte(), uint8(61))

	is.Equal(tt.totals().t2collisions, uint64(0))
	// create a collision
	te = tt.lookup(9409641586953824944, 0)
	is.Equal(te, TableEntry{})
	is.Equal(tt.totals().t2collisions, uint64(1))

	// another lookup, in the next bucket, but this isn't a collision.
	// collision count should not go up.
	te = tt.lookup(9409641586937047728 + bucketSize, 0)
	is.Equal(te, TableEntry{})
	is.Equal(tt.totals().lookups, uint64(3))
	is.Equal(tt.totals().t2collisions, uint64(1))

}

// A bucket keeps its deepest entries: a store into a full bucket replaces the
// shallowest one, a pure entry before any other, and a position's own entry
// first of all.
func TestTTableBucketReplacement(t *testing.T) {
	is := is.New(t)
	tt := &TranspositionTable{}
	tt.Reset(0, 15)
	// Keys that share a bucket differ above the table's index bits.
	key := func(i uint64) uint64 { return 0x1234560 + i<<40 }
	has := func(i uint64) bool {
		te := tt.lookup(key(i), 0)
		return te.valid()
	}
	depths := []uint8{5, 0, 9, 3}
	for i, d := range depths {
		tt.store(key(uint64(i)), int16(i), TTExact<<6+d, 0, 0)
	}
	for i, d := range depths {
		te := tt.lookup(key(uint64(i)), 0)
		is.True(te.valid())
		is.Equal(te.depth(), d)
	}
	// A fifth entry evicts the depth-0 one, even though it is shallow itself.
	tt.store(key(4), 4, TTExact<<6+1, 0, 0)
	is.True(!has(1))
	is.True(has(4))
	for _, i := range []uint64{0, 2, 3} {
		is.True(has(i))
	}
	// Storing a position again replaces its own entry, whatever its depth.
	tt.store(key(2), 7, TTLower<<6+2, 0, 0)
	te := tt.lookup(key(2), 0)
	is.Equal(te.depth(), uint8(2))
	is.Equal(te.getScore(), int16(7))
	is.True(has(0))
	// A pure entry goes before a shallower impure one.
	tt.storeEntry(key(0), 0, TTExact<<6+5, 0, true, 0)
	tt.store(key(5), 5, TTExact<<6+4, 0, 0)
	is.True(!has(0))
	is.True(has(4))
	is.True(has(5))
}
