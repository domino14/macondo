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
	tt.Reset(0)
	tentry := TableEntry{
		score:        12,
		flagAndDepth: 128 + 64 + 23,
	}
	tt.store(9409641586937047728, tentry)

	// arrayLength := len(tt.table)
	is.True(tt.sizePowerOf2 >= 24)

	te := tt.lookup(9409641586937047728)
	is.True(te.valid())
	is.Equal(te.depth(), uint8(23))
	is.Equal(te.flag(), uint8(TTUpper))
	is.Equal(te.score, int16(12))
	is.Equal(te.top4bytes, uint32(2190852907))
	is.Equal(te.fifthbyte, uint8(61))

	is.Equal(tt.t2collisions.Load(), uint64(0))
	// create a collision
	te = tt.lookup(9409641586953824944)
	is.Equal(te, TableEntry{})
	is.Equal(tt.t2collisions.Load(), uint64(1))

	// another lookup, but this isn't a collision. collision count should not go up.
	te = tt.lookup(9409641586937047728 + 1)
	is.Equal(te, TableEntry{})
	is.Equal(tt.lookups.Load(), uint64(3))
	is.Equal(tt.t2collisions.Load(), uint64(1))

}
