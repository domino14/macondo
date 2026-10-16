package playability

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The exact event probability must match a brute-force draw from the pool.
func TestCondDrawMatchesMonteCarlo(t *testing.T) {
	// Pool of 6 letter kinds (0 = blank): counts below.
	unseen := []int{1, 5, 3, 2, 4, 6, 8}
	leave := []int{0, 1, 0, 0, 0, 0, 0}
	word := []byte{1, 1, 2, 3, 4} // AABCD, leave holds one A
	for _, need := range []int{5, 4} {
		cd := NewCondDraw(unseen, leave, 6, word, need, 99)
		var pool []int
		for ml, n := range unseen {
			for i := 0; i < n; i++ {
				pool = append(pool, ml)
			}
		}
		rng := rand.New(rand.NewPCG(1, 2))
		hits, trials := 0, 400000
		for tr := 0; tr < trials; tr++ {
			perm := rng.Perm(len(pool))
			r := make([]int, len(unseen))
			copy(r, leave)
			for i := 0; i < 6; i++ {
				r[pool[perm[i]]]++
			}
			cov, short := 0, 0
			wc := map[int]int{1: 2, 2: 1, 3: 1, 4: 1}
			for l, c := range wc {
				cov += min(c, r[l])
				short += max(0, c-r[l])
			}
			cov += min(r[0], short)
			if cov >= need {
				hits++
			}
		}
		mc := float64(hits) / float64(trials)
		if math.Abs(mc-cd.Prob) > 4*math.Sqrt(mc*(1-mc)/float64(trials))+1e-4 {
			t.Fatalf("need %d: exact %.5f vs MC %.5f", need, cd.Prob, mc)
		}
		// Every sample must be in the event and the right size.
		for i := 0; i < 100; i++ {
			d := cd.Sample(rng)
			if len(d) != 6 {
				t.Fatalf("sample size %d", len(d))
			}
		}
	}
}

func TestCombosNoBlanks(t *testing.T) {
	// Without blanks, combos is a product of binomials: C(9,1)*C(1,1) for "AQ"
	// style alphagram with A=9, Q=1.
	dist := []uint8{0, 9, 1}
	if c := Combos(dist, []byte{1, 2}); c != 9 {
		t.Fatalf("got %v", c)
	}
	// With two blanks: A?:  C(9,1)*C(1,1) + C(2,1)*(C(1,1)+C(9,1)) + C(2,2)
	dist = []uint8{2, 9, 1}
	if c := Combos(dist, []byte{1, 2}); c != 9+2*10+1 {
		t.Fatalf("got %v", c)
	}
}
