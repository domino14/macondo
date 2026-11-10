package rangefinder

import (
	"math"
	"slices"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/montecarlo"
)

// CarryFrom reweights this read's complete posterior over the opponent's
// leave by prev, the read of their play before (made when we were last to
// move). Their rack this turn was the leave they kept then plus fresh draws,
// so the prior of this turn's rack R = L + played is no longer flat: with
// prev's likelihood ratio lik(L') = post(L')/prior(L'), normalised to mean 1
// under the prior, it becomes, up to a constant, the mean of lik over the ways
// to pick |L'| of R's tiles as the carried ones:
//
//	carry(L) = Σ_{L' ⊆ R} [Π_x C(R_x, L'_x) / C(|R|, |L'|)] · lik(L')^power
//
// With an uninformative prev (lik ≡ 1) every carry(L) is 1 and the posterior
// is unchanged. Leaves prev never weighed count as uninformative. ok is false
// when either read is not a complete tile-placement posterior.
func (r *RangeFinder) CarryFrom(prev *RangeFinder, power float64) (racks []montecarlo.InferredRack, ok bool) {
	cur, old := r.inference, prev.inference
	if cur == nil || old == nil || !cur.Complete || !old.Complete ||
		len(cur.InferredRacks) == 0 || len(old.InferredRacks) == 0 {
		return nil, false
	}
	a := old.RackLength
	lik := make(map[string]float64, len(old.InferredRacks))
	sumPost, sumPrior := 0.0, 0.0
	for _, ir := range old.InferredRacks {
		pr := combinatorialPrior(ir.Leave, prev.inferenceBagMap)
		if pr <= 0 || len(ir.Leave) != a {
			continue
		}
		lik[sortedKey(ir.Leave)] = ir.Weight / pr
		sumPost += ir.Weight
		sumPrior += pr
	}
	if sumPost <= 0 {
		return nil, false
	}
	norm := sumPrior / sumPost
	for k, v := range lik {
		lik[k] = math.Pow(v*norm, power)
	}

	racks = make([]montecarlo.InferredRack, len(cur.InferredRacks))
	rack := make([]tilemapping.MachineLetter, 0, 7)
	for i, ir := range cur.InferredRacks {
		rack = append(rack[:0], ir.Leave...)
		rack = append(rack, r.lastOppMoveRackTiles...)
		racks[i] = montecarlo.InferredRack{Leave: ir.Leave, Weight: ir.Weight * carryLift(rack, a, lik)}
	}
	return racks, true
}

// Snapshot keeps this read's posterior for a later CarryFrom. The next
// PrepareFinder starts a fresh posterior and bag map, so the snapshot stays
// intact.
func (r *RangeFinder) Snapshot() *RangeFinder {
	return &RangeFinder{inference: r.inference, inferenceBagMap: r.inferenceBagMap}
}

// carryLift is the mean of lik over the size-a sub-multisets of rack, each
// weighted by its hypergeometric share Π C(R_x, L'_x) / C(|R|, a).
func carryLift(rack []tilemapping.MachineLetter, a int, lik map[string]float64) float64 {
	if a <= 0 || a > len(rack) {
		return 1
	}
	slices.Sort(rack)
	var letters []tilemapping.MachineLetter
	var counts []int
	for i, t := range rack {
		if i == 0 || t != rack[i-1] {
			letters = append(letters, t)
			counts = append(counts, 0)
		}
		counts[len(counts)-1]++
	}
	sub := make([]tilemapping.MachineLetter, 0, a)
	total, wsum := 0.0, 0.0
	var walk func(j, left int, w float64)
	walk = func(j, left int, w float64) {
		if left == 0 {
			l, ok := lik[string(machineBytes(sub))]
			if !ok {
				l = 1
			}
			total += w * l
			wsum += w
			return
		}
		if j == len(letters) {
			return
		}
		for c := min(counts[j], left); c >= 0; c-- {
			for range c {
				sub = append(sub, letters[j])
			}
			walk(j+1, left-c, w*choose(counts[j], c))
			sub = sub[:len(sub)-c]
		}
	}
	walk(0, a, 1)
	if wsum == 0 {
		return 1
	}
	return total / wsum
}

func sortedKey(l []tilemapping.MachineLetter) string {
	s := slices.Clone(l)
	slices.Sort(s)
	return string(machineBytes(s))
}

func machineBytes(l []tilemapping.MachineLetter) []byte {
	b := make([]byte, len(l))
	for i, t := range l {
		b[i] = byte(t)
	}
	return b
}

func choose(n, k int) float64 {
	c := 1.0
	for i := range k {
		c = c * float64(n-i) / float64(i+1)
	}
	return c
}
