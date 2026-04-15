package rangefinder

import (
	"testing"

	"github.com/matryer/is"
)

// TestEnumerateLeavesCount verifies that enumerateLeaves and countMultisets
// agree, that every distinct multiset is generated exactly once, and that
// the hypergeometric priors sum to 1.
func TestEnumerateLeavesCount(t *testing.T) {
	is := is.New(t)

	// Bag: {A:2, B:1, C:3}, draw k=2.
	// Distinct multisets: {A,A}, {A,B}, {A,C}, {B,C}, {C,C} = 5.
	bagMap := make([]uint8, 27)
	bagMap[0] = 2 // A
	bagMap[1] = 1 // B
	bagMap[2] = 3 // C

	k := 2
	leaves := enumerateLeaves(bagMap, k)
	count := countMultisets(bagMap, k)

	is.Equal(len(leaves), count)
	is.Equal(len(leaves), 5)

	// Priors must be strictly positive and sum to 1.
	total := 0.0
	for _, l := range leaves {
		is.True(l.prior > 0)
		is.Equal(len(l.tiles), k)
		total += l.prior
	}
	is.True(total > 0.999 && total < 1.001)
}

// TestEnumerateSingleTile checks k=1: each distinct tile type becomes one leaf.
func TestEnumerateSingleTile(t *testing.T) {
	is := is.New(t)

	// Bag: {A:2, B:1} → two distinct leaves: {A}, {B}.
	bagMap := make([]uint8, 27)
	bagMap[0] = 2 // A: 2 copies
	bagMap[1] = 1 // B: 1 copy

	leaves := enumerateLeaves(bagMap, 1)
	count := countMultisets(bagMap, 1)

	is.Equal(len(leaves), count)
	is.Equal(len(leaves), 2)

	// P(A) = 2/3, P(B) = 1/3; sum = 1.
	total := 0.0
	for _, l := range leaves {
		is.True(l.prior > 0)
		total += l.prior
	}
	is.True(total > 0.999 && total < 1.001)
}

// TestEnumerateAllSame checks a bag with only one tile type: k copies → 1 leaf.
func TestEnumerateAllSame(t *testing.T) {
	is := is.New(t)

	// Bag: {E:5}, draw k=3 → only one leaf: {E,E,E}.
	bagMap := make([]uint8, 27)
	bagMap[4] = 5 // E: 5 copies (index 4 = E)

	leaves := enumerateLeaves(bagMap, 3)
	count := countMultisets(bagMap, 3)

	is.Equal(len(leaves), 1)
	is.Equal(count, 1)
	// Only one outcome → prior must be 1.
	is.True(leaves[0].prior > 0.999 && leaves[0].prior < 1.001)
}

// TestCountMultisetsScrabbleLike checks the count against known values for a
// late-game bag.
func TestCountMultisetsScrabbleLike(t *testing.T) {
	is := is.New(t)

	// Simulate a small late-game bag: 3 distinct tile types with small counts.
	bagMap := make([]uint8, 27)
	bagMap[0] = 1 // A:1
	bagMap[4] = 2 // E:2
	bagMap[8] = 1 // I:1  (total 4 tiles)

	// k=2: distinct multisets from {A:1, E:2, I:1}:
	// {A,E}, {A,I}, {E,E}, {E,I} = 4
	is.Equal(countMultisets(bagMap, 2), 4)
	leaves := enumerateLeaves(bagMap, 2)
	is.Equal(len(leaves), 4)

	total := 0.0
	for _, l := range leaves {
		is.True(l.prior > 0)
		total += l.prior
	}
	is.True(total > 0.999 && total < 1.001)
}
