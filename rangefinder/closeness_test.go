package rangefinder

import (
	"math"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
)

func TestCloseness(t *testing.T) {
	is := is.New(t)
	ml := func(xs ...int) []tilemapping.MachineLetter {
		out := make([]tilemapping.MachineLetter, len(xs))
		for i, x := range xs {
			out[i] = tilemapping.MachineLetter(x)
		}
		return out
	}
	// A toy valuer: each tile is worth its own number, so swapping 4 for 5 is
	// cheap and 4 for 20 is not.
	value := func(w tilemapping.MachineWord) float64 {
		v := 0.0
		for _, x := range w {
			v += float64(x)
		}
		return v
	}
	truth := ml(1, 2, 4)
	leaves := [][]tilemapping.MachineLetter{ml(1, 2, 4), ml(1, 2, 5), ml(1, 2, 20), ml(7, 8, 9)}
	prior := []float64{1, 1, 1, 1}

	// All weight on the truth: full overlap, zero distance, and the soft lifts
	// are -log2 of the prior's expected kernel.
	c := closeness(truth, leaves, []float64{1, 0, 0, 0}, prior, 4, value)
	is.Equal(c.Overlap, 1.0)
	is.Equal(c.ValueDist, 0.0)
	is.True(c.SoftLiftOverlap > 0 && c.SoftLiftValue > 0)

	// A near miss (5 for 4) earns more credit than a far one (20 for 4), and
	// both more than a leave sharing nothing.
	near := closeness(truth, leaves, []float64{0, 1, 0, 0}, prior, 4, value)
	far := closeness(truth, leaves, []float64{0, 0, 1, 0}, prior, 4, value)
	none := closeness(truth, leaves, []float64{0, 0, 0, 1}, prior, 4, value)
	is.Equal(near.Overlap, far.Overlap) // the overlap kernel cannot tell them apart
	is.True(near.SoftLiftValue > far.SoftLiftValue)
	is.True(far.SoftLiftOverlap > none.SoftLiftOverlap)
	is.True(math.Abs(near.ValueDist-2) < 1e-9) // |8-7| + one swapped tile
}
