package rangefinder

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
)

// leaveStr converts a sorted slice of MachineLetter to a human-readable string.
// 0 = blank (?), 1 = A, 2 = B, ...
func leaveStr(tiles []tilemapping.MachineLetter) string {
	var sb strings.Builder
	for _, ml := range tiles {
		if ml == 0 {
			sb.WriteByte('?')
		} else {
			sb.WriteByte(byte('A' + ml - 1))
		}
	}
	return sb.String()
}

// TestLeavesPrintout enumerates all 4-tile leaves from the given unseen pool
// and writes them to /tmp/leaves.txt, sorted by prior descending.
//
// Unseen tiles: ? AA C EE F G III LL N O Q R SS TT U  (22 tiles total)
// Opponent made a 3-tile play → kept 4 tiles (k=4).
func TestLeavesPrintout(t *testing.T) {
	// Build bagMap: index 0=blank, 1=A, 2=B, ..., 26=Z
	bagMap := make([]uint8, 27)
	bagMap[0] = 1  // ?  (blank)
	bagMap[1] = 2  // A
	bagMap[3] = 1  // C
	bagMap[5] = 2  // E
	bagMap[6] = 1  // F
	bagMap[7] = 1  // G
	bagMap[9] = 3  // I
	bagMap[12] = 2 // L
	bagMap[14] = 1 // N
	bagMap[15] = 1 // O
	bagMap[17] = 1 // Q
	bagMap[18] = 1 // R
	bagMap[19] = 2 // S
	bagMap[20] = 2 // T
	bagMap[21] = 1 // U

	k := 4
	N := 0
	for _, c := range bagMap {
		N += int(c)
	}

	totalDistinct := countMultisets(bagMap, k)
	leaves := enumerateLeaves(bagMap, k)

	// Sanity: enumerateLeaves and countMultisets must agree.
	if len(leaves) != totalDistinct {
		t.Errorf("count mismatch: countMultisets=%d enumerateLeaves=%d", totalDistinct, len(leaves))
	}

	// Sort by prior descending.
	sort.Slice(leaves, func(i, j int) bool { return leaves[i].prior > leaves[j].prior })

	// Verify priors sum to 1.
	total := 0.0
	for _, l := range leaves {
		total += l.prior
	}

	out, err := os.Create("/tmp/leaves.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	fmt.Fprintf(out, "Unseen pool: ? AA C EE F G III LL N O Q R SS TT U\n")
	fmt.Fprintf(out, "Total unseen tiles (N): %d\n", N)
	fmt.Fprintf(out, "Opponent played 3 tiles, kept k=%d\n", k)
	fmt.Fprintf(out, "C(%d,%d) = %d  (total ways to draw %d tiles)\n\n", N, k, comb(N, k), k)
	fmt.Fprintf(out, "Distinct leaves: %d\n", totalDistinct)
	fmt.Fprintf(out, "Prior sum: %.8f\n\n", total)

	fmt.Fprintf(out, "%-10s  %12s  %10s  %s\n", "Leave", "Prior", "Numerator", "= Π C(avail,count) / C(N,k)")
	fmt.Fprintf(out, "%s\n", strings.Repeat("-", 72))

	denom := float64(comb(N, k))
	for _, l := range leaves {
		// Compute numerator = Π C(bagMap[t], count_t)
		counts := map[tilemapping.MachineLetter]int{}
		for _, ml := range l.tiles {
			counts[ml]++
		}
		num := 1
		parts := []string{}
		for ml, cnt := range counts {
			c := comb(int(bagMap[ml]), cnt)
			num *= c
			if int(bagMap[ml]) == cnt {
				parts = append(parts, fmt.Sprintf("C(%d,%d)=%d", bagMap[ml], cnt, c))
			} else {
				parts = append(parts, fmt.Sprintf("C(%d,%d)=%d", bagMap[ml], cnt, c))
			}
		}
		sort.Strings(parts)
		fmt.Fprintf(out, "%-10s  %12.8f  %10d  %s / %.0f\n",
			leaveStr(l.tiles), l.prior, num, strings.Join(parts, " * "), denom)
	}

	t.Logf("Wrote %d leaves to /tmp/leaves.txt (N=%d, k=%d, total prior=%.8f)",
		len(leaves), N, k, total)
}

// comb computes the binomial coefficient C(n, k) as an integer.
func comb(n, k int) int {
	if k < 0 || k > n {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	if k > n-k {
		k = n - k
	}
	result := 1
	for i := 0; i < k; i++ {
		result = result * (n - i) / (i + 1)
	}
	return result
}
