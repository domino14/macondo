package negamax

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/domino14/macondo/board"
	"github.com/matryer/is"
)

// TestLazySMPScaling tests how LazySMP performance scales with thread count
func TestLazySMPScaling(t *testing.T) {
	t.Skip()
	is := is.New(t)

	// Test with increasing thread counts to see where scaling breaks down
	threadCounts := []int{1, 2, 4, 8, 10, 12, 16, 20, 24}

	plies := 14

	fmt.Printf("\n=== Testing VsJoey %d plies (scaling from 1 to 24 threads) ===\n", plies)

	baselineTime := 0.0

	for _, threads := range threadCounts {
		// Setup solver using existing test infrastructure
		s, err := setUpSolver("NWL18", "english", board.VsJoey, plies, "DIV", "AEFILMR", 412, 371, 1)
		is.NoErr(err)

		s.threads = threads
		if threads > 1 {
			s.lazySMPOptim = true
		}

		start := time.Now()
		v, _, err := s.Solve(context.Background(), plies)
		elapsed := time.Since(start).Seconds()

		is.NoErr(err)
		is.Equal(v, int16(55))

		if threads == 1 {
			baselineTime = elapsed
			fmt.Printf("%2d threads: %8.2fs (baseline)\n", threads, elapsed)
		} else {
			speedup := baselineTime / elapsed
			efficiency := speedup / float64(threads) * 100
			fmt.Printf("%2d threads: %8.2fs (%.2fx speedup, %.1f%% efficiency)\n",
				threads, elapsed, speedup, efficiency)
		}
	}
}
