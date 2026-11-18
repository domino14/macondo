package main

import (
	"sync/atomic"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
)

// -readmix eps blends the bot's complete posterior with the prior over the
// same leaves, (1-eps)·post + eps·prior, both normalised: with probability eps
// a sim iteration draws the opponent's leave as a plain sim would. A guard for
// reads that are confidently wrong, where a leave near the truth held with
// certainty does worse than no read at all.
var readMixEps float64

var readMixed atomic.Int64 // reads blended, for the log

func mixRead(g *game.Game, racks []montecarlo.InferredRack) []montecarlo.InferredRack {
	pool := unseenPool(g)
	prior := make([]float64, len(racks))
	sumPost, sumPrior := 0.0, 0.0
	for i, r := range racks {
		prior[i] = leavePrior(r.Leave, pool)
		sumPost += r.Weight
		sumPrior += prior[i]
	}
	if sumPost <= 0 || sumPrior <= 0 {
		return racks
	}
	readMixed.Add(1)
	out := make([]montecarlo.InferredRack, len(racks))
	for i, r := range racks {
		out[i] = montecarlo.InferredRack{Leave: r.Leave,
			Weight: (1-readMixEps)*r.Weight/sumPost + readMixEps*prior[i]/sumPrior}
	}
	return out
}

// -readsmooth alpha smooths the bot's read over one-tile swaps: each leave's
// likelihood (posterior over prior) becomes (1-alpha) of its own plus alpha of
// the mean over the leaves one tile away. The mixture screens found a read
// spread over the truth's one-tile swaps worth more than one spread over
// leaves alike in value or features, so a leave earns weight from its
// neighbours' evidence on the tiles they share.
var readSmoothAlpha float64

func smoothRead(g *game.Game, racks []montecarlo.InferredRack) []montecarlo.InferredRack {
	pool := unseenPool(g)
	lik := make(map[string]float64, len(racks))
	prior := make([]float64, len(racks))
	for i, r := range racks {
		prior[i] = leavePrior(r.Leave, pool)
		if prior[i] > 0 {
			lik[leaveKey(r.Leave)] = r.Weight / prior[i]
		}
	}
	out := make([]montecarlo.InferredRack, len(racks))
	nb := make([]tilemapping.MachineLetter, 0, 7)
	for i, r := range racks {
		own, ok := lik[leaveKey(r.Leave)]
		if !ok {
			out[i] = r
			continue
		}
		sum, n := 0.0, 0
		l := sorted(r.Leave)
		for j := range l {
			if j > 0 && l[j] == l[j-1] {
				continue
			}
			for t := range pool {
				if tilemapping.MachineLetter(t) == l[j] {
					continue
				}
				nb = append(nb[:0], l...)
				nb[j] = tilemapping.MachineLetter(t)
				if l, ok := lik[leaveKey(nb)]; ok {
					sum += l
					n++
				}
			}
		}
		smoothed := own
		if n > 0 {
			smoothed = (1-readSmoothAlpha)*own + readSmoothAlpha*sum/float64(n)
		}
		out[i] = montecarlo.InferredRack{Leave: r.Leave, Weight: prior[i] * smoothed}
	}
	readSmoothed.Add(1)
	return out
}

var readSmoothed atomic.Int64

func unseenPool(g *game.Game) []uint8 {
	pool := g.Bag().PeekMap()
	for _, t := range g.RackFor(g.NextPlayer()).TilesOn() {
		pool[t]++
	}
	return pool
}

// leavePrior is Π C(pool_x, L_x), the number of ways to hold the leave.
func leavePrior(leave []tilemapping.MachineLetter, pool []uint8) float64 {
	var counts [64]int
	p := 1.0
	for _, t := range leave {
		counts[t]++
		p *= float64(int(pool[t])-counts[t]+1) / float64(counts[t])
	}
	return p
}
