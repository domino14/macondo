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

// The graph exists because the ratio bins it replaced could not tell a soul
// read from a rounding error: the game's last Z going from 4% to 71% and a
// tile nudged from 1% to 3% both came out as "way more than chance".
func TestHoldingGraph(t *testing.T) {
	is := is.New(t)

	got := HoldingGraph([]TileDeviation{
		// One Z left, and the read says they have it four times in five.
		{Tile: "Z", HoldsPct: 70.9, ChanceHoldsPct: 3.6, Unseen: 1, Tiles: 0.75},
		// Three times its expected share, and a twentieth of a tile.
		{Tile: "Q", HoldsPct: 3.6, ChanceHoldsPct: 1.2, Unseen: 1, Tiles: 0.07},
		// Common, and the read says nothing about it either way.
		{Tile: "E", HoldsPct: 31.5, ChanceHoldsPct: 30.9, Unseen: 11, Tiles: 0.02},
		// The read pushes it down, which is a finding too.
		{Tile: "A", HoldsPct: 21.6, ChanceHoldsPct: 45.8, Unseen: 9, Tiles: -0.62},
		// Already all played: not "they don't have it", but "nobody can".
		{Tile: "J", HoldsPct: 0, ChanceHoldsPct: 0, Unseen: 0, Tiles: 0},
		{Tile: "X", HoldsPct: 0, ChanceHoldsPct: 0, Unseen: 0, Tiles: 0},
	})

	// Position on the graph is the probability, so the Z sits near the top and
	// the tile with the bigger *ratio* sits at the bottom.
	is.True(strings.Contains(got, "   70- 80% | Z+\n"))
	is.True(strings.Contains(got, "    0- 10% | Q\n"))
	is.True(strings.Contains(got, "   30- 40% | E\n"))
	is.True(strings.Contains(got, "   20- 30% | A-\n"))

	// Empty bands are kept - the gap between the Z and everything else is the
	// shape of the read - but carry no trailing whitespace.
	is.True(strings.Contains(got, "   90-100% |\n"))
	is.True(strings.Contains(got, "   40- 50% |\n"))

	// Tiles nobody can hold are named apart from tiles the read ruled out.
	is.True(strings.Contains(got, "All played, none left to hold: J X"))
	is.True(!strings.Contains(got, "| J"))

	// And the exact figures survive, because a band is a ten-point range and
	// the difference between 71% and 79% is worth having.
	is.True(strings.Contains(got, "Standouts: Z 71% (chance 4%), A 22% (chance 46%)"))
	// A tile that only looks big as a ratio is not a standout.
	is.True(!strings.Contains(got, "Q 4%"))
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
