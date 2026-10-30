package rangefinder

import (
	"math"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
)

func TestCarryLift(t *testing.T) {
	ml := func(s ...int) []tilemapping.MachineLetter {
		out := make([]tilemapping.MachineLetter, len(s))
		for i, x := range s {
			out[i] = tilemapping.MachineLetter(x)
		}
		return out
	}
	// Uninformative read: every leave keeps its weight.
	if got := carryLift(ml(1, 2, 2, 3), 2, map[string]float64{}); got != 1 {
		t.Fatalf("empty lik: got %v, want 1", got)
	}
	// Rack ABC, one tile carried, the old read likes A three times as much.
	lik := map[string]float64{sortedKey(ml(1)): 3}
	if got := carryLift(ml(3, 1, 2), 1, lik); math.Abs(got-5.0/3) > 1e-12 {
		t.Fatalf("got %v, want 5/3", got)
	}
	// Rack AAB, two carried: AA has weight 1 of C(3,2)=3, AB 2 of 3.
	lik = map[string]float64{sortedKey(ml(1, 1)): 4, sortedKey(ml(2, 1)): 1}
	if got := carryLift(ml(1, 2, 1), 2, lik); math.Abs(got-(4.0+2)/3) > 1e-12 {
		t.Fatalf("got %v, want 2", got)
	}
}
