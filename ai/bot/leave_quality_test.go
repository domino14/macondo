package bot

import (
	"math"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
)

func TestEnumerateLeavesPrior(t *testing.T) {
	pool := []uint8{2, 9, 0, 3, 1} // blank x2, A x9, C x3, D x1
	n := 15
	for k := 1; k <= 6; k++ {
		total := 0.0
		for _, l := range enumerateLeaves(pool, k) {
			total += multisetWays(l, pool)
		}
		want := 1.0
		for x := 0; x < k; x++ {
			want *= float64(n-x) / float64(x+1)
		}
		if math.Abs(total-want) > 1e-6*want {
			t.Errorf("k=%d: ways sum to %v, want C(%d,%d)=%v", k, total, n, k, want)
		}
	}
	if w := multisetWays([]tilemapping.MachineLetter{1, 1}, pool); w != 36 {
		t.Errorf("AA: %v ways, want 36", w)
	}
}
