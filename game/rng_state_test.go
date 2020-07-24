package game

import (
	"testing"

	"github.com/dgryski/go-pcgr"
	"github.com/matryer/is"
)

func TestReloadRNG(t *testing.T) {
	is := is.New(t)
	// Test that the RNG state can be saved and reloaded.
	seed := int64(0x7AFEBABEF00DBADA)
	seq := int64(17)

	rand := pcgr.New(seed, seq)
	initNums := []uint32{}
	for i := 0; i < 10; i++ {
		initNums = append(initNums, rand.Next())
	}
	// Now, create another RNG with the same seed/seq but stop halfway through.

	newNums := []uint32{}

	rand1 := pcgr.New(seed, seq)
	for i := 0; i < 5; i++ {
		newNums = append(newNums, rand1.Next())
	}

	// Simulate saving state into a new randomizer
	rand2 := &pcgr.Rand{State: rand1.State, Inc: rand1.Inc}
	for i := 0; i < 5; i++ {
		newNums = append(newNums, rand2.Next())
	}
	is.Equal(initNums, newNums)

}
