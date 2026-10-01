package bot

import "testing"

func TestSimPliesFor(t *testing.T) {
	for _, c := range []struct {
		unseen, minPlies, fixed, want int
	}{
		// Default: unseen plies with 2..7 tiles in the bag, else max(2, min).
		{9, 0, 0, 9}, {14, 5, 0, 14}, {15, 0, 0, 2}, {15, 2, 0, 2}, {40, 5, 0, 5},
		// Fixed: that depth everywhere, including one tile in the bag (unseen 8).
		{8, 0, 2, 2}, {9, 0, 2, 2}, {14, 5, 2, 2}, {15, 5, 3, 3}, {60, 0, 6, 6},
	} {
		if got := simPliesFor(c.unseen, c.minPlies, c.fixed); got != c.want {
			t.Errorf("simPliesFor(unseen %d, min %d, fixed %d) = %d, want %d", c.unseen, c.minPlies, c.fixed, got, c.want)
		}
	}
}
