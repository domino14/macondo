package rangefinder

import (
	"math"
	"strings"
	"testing"

	"github.com/matryer/is"
)

func closeTo(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("got %.4f, want %.4f (±%.4f)", got, want, tol)
	}
}

// The figures are from a real position whose whole finding was that the
// opponent kept consonants. No single tile moves much - the largest is N at
// +8.9 points, a tenth of a tile - so the read lives in the pattern across
// letters, and a graph has to make that pattern visible or it is useless here.
//
// The banded histogram this replaced actively inverted it: height was the
// probability itself, which is mostly a function of how many copies are left,
// so the nine unseen I's floated to the top while the read was pushing I down.
func TestMovementGraph(t *testing.T) {
	is := is.New(t)

	got := MovementGraph([]TileDeviation{
		{Tile: "A", HoldsPct: 23.66, ChanceHoldsPct: 24.87, Unseen: 8},
		{Tile: "B", HoldsPct: 12.25, ChanceHoldsPct: 6.66, Unseen: 2},
		{Tile: "N", HoldsPct: 25.00, ChanceHoldsPct: 16.10, Unseen: 5},
		{Tile: "U", HoldsPct: 3.54, ChanceHoldsPct: 13.03, Unseen: 4},
		{Tile: "Z", HoldsPct: 3.32, ChanceHoldsPct: 3.37, Unseen: 1},
		{Tile: "K", HoldsPct: 0, ChanceHoldsPct: 0, Unseen: 0},
	})

	// Sorted by movement, so the read is the shape of the list: what it
	// favours at the top, what it rules out at the bottom. Absolute
	// probability decides nothing - A is the second likeliest tile here and
	// sits near the bottom, because the read did not put it there.
	order := []string{"N", "B", "Z", "A", "U"}
	at := -1
	for _, tile := range order {
		i := strings.Index(got, "\n  "+tile+" ")
		is.True(i > at)
		at = i
	}

	// Bars run from a centre line, right for likelier and left for less, and
	// scale against the widest movement in this read - U at -9.5 here.
	is.True(strings.Contains(got, "|"+strings.Repeat("+", 21))) // N, +8.9
	is.True(strings.Contains(got, strings.Repeat("-", 22)+"|")) // U, -9.5
	is.True(strings.Contains(got, "|"+strings.Repeat("+", 13))) // B, +5.6
	is.True(strings.Contains(got, strings.Repeat("-", 3)+"|"))  // A, -1.2

	// Both figures on every row, because a bar is a picture and the exact
	// pair is what anyone would actually quote.
	is.True(strings.Contains(got, " 16.1 ->  25.0%   +8.9  (5 unseen)"))
	is.True(strings.Contains(got, " 13.0 ->   3.5%   -9.5  (4 unseen)"))

	// A tile the read left alone gets no bar at all, on either side.
	is.True(strings.Contains(got, "  Z  "+strings.Repeat(" ", graphHalf)+"|"))

	// Tiles nobody can hold are named apart from tiles the read ruled out.
	is.True(strings.Contains(got, "All played, none left to hold: K"))
	is.True(!strings.Contains(got, "\n  K "))
}

// A read is sometimes not about any tile. Here it is about the makeup of the
// rack, and the mean alone would not say how reliably.
func TestShapeSummary(t *testing.T) {
	is := is.New(t)

	is.Equal(ShapeSummary(nil), "")

	s := &RackShape{
		RackLength: 3,
		Vowels:     CountPair{Read: 0.9026, Chance: 1.2809},
		Consonants: CountPair{Read: 2.0084, Chance: 1.6517},
		Blanks:     CountPair{Read: 0.0890, Chance: 0.0674},
		VowelCount: []CountPair{
			{Read: 31.2, Chance: 18.3}, {Read: 45.1, Chance: 42.7},
			{Read: 20.4, Chance: 31.6}, {Read: 3.3, Chance: 7.4},
		},
	}
	is.True(s.Notable()) // 0.38 of a tile of vowels is a real difference

	got := ShapeSummary(s)
	is.True(strings.Contains(got, "2.01 consonants  where a random rack holds 1.65   +0.36"))
	is.True(strings.Contains(got, "0.90 vowels      where a random rack holds 1.28   -0.38"))
	// Both sides of the read are flagged; the blank barely moved and is not.
	is.Equal(strings.Count(got, "<-- the read"), 2)

	// The distribution follows, because "0.9 vowels on average" is equally
	// consistent with "usually one" and with "half the time two, half none".
	is.True(strings.Contains(got, "vowels held"))
	is.True(strings.Contains(got, "0               31.2%   18.3%"))
	is.True(strings.Contains(got, "3                3.3%    7.4%"))

	// A read that left the shape alone says so by not bringing it up.
	flat := &RackShape{
		RackLength: 3,
		Vowels:     CountPair{Read: 1.30, Chance: 1.28},
		Consonants: CountPair{Read: 1.63, Chance: 1.65},
		VowelCount: []CountPair{{Read: 18.1, Chance: 18.3}},
	}
	is.True(!flat.Notable())
	is.True(!strings.Contains(ShapeSummary(flat), "<-- the read"))
	is.True(!strings.Contains(ShapeSummary(flat), "vowels held"))
}

// The chance side of the shape is exact, not sampled: drawing k vowels from a
// pool is a hypergeometric, and these are the figures for the position above -
// 38 vowels among 89 unseen, three tiles drawn.
func TestHypergeometric(t *testing.T) {
	is := is.New(t)

	closeTo(t, hypergeometric(89, 38, 3, 0), 0.1833768, 1e-7)
	closeTo(t, hypergeometric(89, 38, 3, 1), 0.4266317, 1e-7)
	closeTo(t, hypergeometric(89, 38, 3, 2), 0.3157074, 1e-7)
	closeTo(t, hypergeometric(89, 38, 3, 3), 0.0742841, 1e-7)

	total, mean := 0.0, 0.0
	for k := range 4 {
		p := hypergeometric(89, 38, 3, k)
		total += p
		mean += float64(k) * p
	}
	closeTo(t, total, 1.0, 1e-9)
	// And the mean agrees with the simple expected count, 3 x 38/89.
	closeTo(t, mean, 3*38.0/89.0, 1e-9)

	// Impossible draws are zero rather than negative or NaN.
	is.Equal(hypergeometric(89, 38, 3, 4), 0.0) // more than we drew
	is.Equal(hypergeometric(10, 2, 3, 3), 0.0)  // more than exist
	is.Equal(hypergeometric(10, 9, 3, 0), 0.0)  // too few of the others
	closeTo(t, hypergeometric(10, 10, 3, 3), 1.0, 1e-12)
}

// The chance of drawing at least one copy is what makes a read legible:
// "they have the Z four times in five" rather than "+0.75 tiles".
func TestChanceOfHolding(t *testing.T) {
	is := is.New(t)

	// A single copy in the pool reduces to the familiar rackLen/unseen: three
	// chances out of eighty-three to pick up the only Z.
	closeTo(t, chanceOfHolding(83, 1, 3), 3.0/83.0, 1e-12)
	closeTo(t, chanceOfHolding(100, 1, 7), 7.0/100.0, 1e-12)

	// Six A's among 83 unseen, drawing three: 1 - C(77,3)/C(83,3).
	closeTo(t, chanceOfHolding(83, 6, 3), 1-(77.0*76.0*75.0)/(83.0*82.0*81.0), 1e-12)

	// More copies is likelier, and a longer rack is likelier still.
	is.True(chanceOfHolding(83, 6, 3) > chanceOfHolding(83, 1, 3))
	is.True(chanceOfHolding(83, 1, 7) > chanceOfHolding(83, 1, 3))

	// Degenerate cases don't produce nonsense.
	is.Equal(chanceOfHolding(83, 0, 3), 0.0) // none left to draw
	is.Equal(chanceOfHolding(83, 1, 0), 0.0) // nothing drawn
	is.Equal(chanceOfHolding(0, 1, 3), 0.0)  // nothing unseen
	// Only two tiles in the pool that aren't blanks, drawing three: certain.
	is.Equal(chanceOfHolding(5, 3, 3), 1.0)
	closeTo(t, chanceOfHolding(5, 5, 3), 1.0, 1e-12)
}
