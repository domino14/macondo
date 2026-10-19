package rangefinder

import (
	"sync"

	"github.com/domino14/word-golib/tilemapping"
)

// Fourth-order terms for the imputation.
//
// Orders one to three pack a sorted sub-multiset into a dense index over
// ordered tuples, alphaSize^order wide, of which only the sorted tuples are
// used. At order four that would be 27^4 = 531,441 slots for 27 tile types,
// three float64 arrays each, in every accumulator including the cross-fitting
// folds -- for the C(30,4) = 27,405 sorted multisets that can actually occur.
// So order four ranks a sorted multiset directly, by the combinatorial number
// system, and carries the inverse as a table.

// order4Index ranks sorted 4-multisets over an alphabet and unranks them.
type order4Index struct {
	alphaSize int
	size      int
	binom     [][5]int // binom[n][k] for k ≤ 4
	tiles     [][4]tilemapping.MachineLetter
}

var (
	order4Cache   = map[int]*order4Index{}
	order4CacheMu sync.Mutex
)

// order4IndexFor builds, or reuses, the index for an alphabet size. The table
// is the same for every inference over the same tile set, so it is built once.
func order4IndexFor(alphaSize int) *order4Index {
	order4CacheMu.Lock()
	defer order4CacheMu.Unlock()
	if ix, ok := order4Cache[alphaSize]; ok {
		return ix
	}
	n := alphaSize + 3
	ix := &order4Index{alphaSize: alphaSize, binom: make([][5]int, n+1)}
	for i := 0; i <= n; i++ {
		ix.binom[i][0] = 1
		for k := 1; k <= 4 && k <= i; k++ {
			ix.binom[i][k] = ix.binom[i-1][k-1] + ix.binom[i-1][k]
		}
	}
	ix.size = ix.binom[n][4]
	ix.tiles = make([][4]tilemapping.MachineLetter, ix.size)
	for a := 0; a < alphaSize; a++ {
		for b := a; b < alphaSize; b++ {
			for c := b; c < alphaSize; c++ {
				for d := c; d < alphaSize; d++ {
					ix.tiles[ix.rankInts(a, b, c, d)] = [4]tilemapping.MachineLetter{
						tilemapping.MachineLetter(a), tilemapping.MachineLetter(b),
						tilemapping.MachineLetter(c), tilemapping.MachineLetter(d)}
				}
			}
		}
	}
	order4Cache[alphaSize] = ix
	return ix
}

// rankInts maps a sorted multiset a ≤ b ≤ c ≤ d to its index. The multiset
// becomes the strict combination {a, b+1, c+2, d+3} from alphaSize+3 symbols,
// ranked in colex order: C(a,1) + C(b+1,2) + C(c+2,3) + C(d+3,4).
func (ix *order4Index) rankInts(a, b, c, d int) int {
	return ix.binom[a][1] + ix.binom[b+1][2] + ix.binom[c+2][3] + ix.binom[d+3][4]
}

// rank is rankInts over a sorted slice of exactly four tiles.
func (ix *order4Index) rank(s []tilemapping.MachineLetter) int {
	return ix.rankInts(int(s[0]), int(s[1]), int(s[2]), int(s[3]))
}

// forSubMultisets calls fn once for every distinct sub-multiset of exactly
// order m drawn from runs, as a sorted slice that fn must not retain. The runs
// come sorted by tile, so the slice is too. Recursion is over the runs, at
// most one frame per distinct tile, and nothing is allocated per call.
func forSubMultisets(runs []tileRun, m int, fn func(sub []tilemapping.MachineLetter)) {
	var buf [8]tilemapping.MachineLetter
	if m > len(buf) {
		return
	}
	var rec func(i, filled int)
	rec = func(i, filled int) {
		if filled == m {
			fn(buf[:m])
			return
		}
		if i == len(runs) {
			return
		}
		// Room left across the remaining runs; prune when it cannot reach m.
		left := 0
		for _, r := range runs[i:] {
			left += r.c
		}
		if filled+left < m {
			return
		}
		r := runs[i]
		take := min(r.c, m-filled)
		for k := take; k >= 0; k-- {
			for j := 0; j < k; j++ {
				buf[filled+j] = r.t
			}
			rec(i+1, filled+k)
		}
	}
	rec(0, 0)
}

// packIdx is the dense index the lower orders use for a sorted sub-multiset
// of order one to three.
func packIdx(alphaSize int, s []tilemapping.MachineLetter) int {
	switch len(s) {
	case 1:
		return int(s[0])
	case 2:
		return int(s[0])*alphaSize + int(s[1])
	default:
		return (int(s[0])*alphaSize+int(s[1]))*alphaSize + int(s[2])
	}
}
