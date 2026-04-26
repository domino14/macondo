package rangefinder

import (
	"context"
	"sort"
	"sync/atomic"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"

	"github.com/domino14/macondo/ai/simplesimmer"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
)

const (
	// DefaultMaxEnumeratedLeaves is the default threshold for exhaustive enumeration.
	// When the number of distinct leaves drawable from the bag is at or below this
	// value, inferEnumerated is used instead of Monte Carlo sampling.
	// At ~100 ms per mini-sim and 16 threads this corresponds to roughly 30 s of wall
	// time, which is the natural knee of the tractability curve:
	//   k=1 → ≤27 leaves    (always enumerate)
	//   k=2 → ≤378 leaves   (always enumerate)
	//   k=3 → ≤3650 leaves  (enumerate when bag is modest, ≤~20 distinct tiles)
	//   k=4 → ≤27000 leaves (enumerate only when bag ≤~15 tiles total)
	// Anything above 750 typically exceeds the default 1m sampling budget and won't
	// beat MC in practice.
	DefaultMaxEnumeratedLeaves = 750

	// enumerateEarlyExitEps controls how much prior mass we allow to go unprocessed
	// when the leaf count exceeds enumerateAlwaysFullThreshold.  Leaves are processed
	// in descending prior order; the lowest-prior tail whose cumulative mass is below
	// this fraction of the total is dropped before any mini-sims run.  A value of 0.01
	// means at most 1% of the posterior weight is skipped (since likelihood ≤ 1).
	// Has no effect when len(leaves) ≤ enumerateAlwaysFullThreshold.
	enumerateEarlyExitEps = 0.01

	// enumerateAlwaysFullThreshold is the leaf count below which the early-exit
	// truncation is never applied — every leaf is processed regardless of its prior.
	// Set equal to DefaultMaxEnumeratedLeaves: if we already decided the space is
	// small enough to enumerate exhaustively, we should actually enumerate it fully.
	// Truncation is only applied when a position somehow exceeds the MC fallback
	// threshold but is still routed to inferEnumerated (e.g. via a manually raised
	// maxEnumeratedLeaves config).  In practice this means:
	//   k=1 (≤27), k=2 (≤378), k=4 late-game (≤~1940) — always full.
	enumerateAlwaysFullThreshold = DefaultMaxEnumeratedLeaves
)

// enumLeaf represents one possible opponent leave together with its
// exact multivariate-hypergeometric prior probability.
type enumLeaf struct {
	tiles []tilemapping.MachineLetter
	prior float64
}

// countMultisets returns the number of distinct multisets of size k that can be
// drawn from bagMap (without allocating).  Used to decide whether exhaustive
// enumeration is cheaper than sampling.
func countMultisets(bagMap []uint8, k int) int {
	count := 0
	var recurse func(minIdx, remaining int)
	recurse = func(minIdx, remaining int) {
		if remaining == 0 {
			count++
			return
		}
		for i := minIdx; i < len(bagMap); i++ {
			if bagMap[i] == 0 {
				continue
			}
			maxCopies := int(bagMap[i])
			if remaining < maxCopies {
				maxCopies = remaining
			}
			for cnt := 1; cnt <= maxCopies; cnt++ {
				recurse(i+1, remaining-cnt)
			}
		}
	}
	recurse(0, k)
	return count
}

// enumerateLeaves generates every distinct multiset of size k that can be drawn
// from bagMap.  Each returned enumLeaf carries its exact hypergeometric prior.
// Leaves are not in any particular order; call sort before using.
func enumerateLeaves(bagMap []uint8, k int) []enumLeaf {
	var result []enumLeaf
	buf := make([]tilemapping.MachineLetter, 0, k)

	var recurse func(minIdx, remaining int)
	recurse = func(minIdx, remaining int) {
		if remaining == 0 {
			tiles := make([]tilemapping.MachineLetter, len(buf))
			copy(tiles, buf)
			p := combinatorialPrior(tiles, bagMap)
			result = append(result, enumLeaf{tiles: tiles, prior: p})
			return
		}
		for i := minIdx; i < len(bagMap); i++ {
			if bagMap[i] == 0 {
				continue
			}
			maxCopies := int(bagMap[i])
			if remaining < maxCopies {
				maxCopies = remaining
			}
			prevLen := len(buf)
			// Take 1..maxCopies copies of tile i, then recurse for tiles i+1..
			for cnt := 1; cnt <= maxCopies; cnt++ {
				buf = append(buf, tilemapping.MachineLetter(i))
				recurse(i+1, remaining-cnt)
			}
			buf = buf[:prevLen] // restore buf before moving to next tile
		}
	}
	recurse(0, k)
	return result
}

// inferEnumerated exhaustively evaluates every possible opponent leave and
// applies full Bayesian weighting: weight = prior(L) * likelihood(L|play).
//
// This is mathematically exact (no sampling noise) and faster than MC for
// small leave spaces (≤ DefaultMaxEnumeratedLeaves distinct leaves).
//
// Unlike the sampling path — where SetRandomRack implicitly samples from the
// prior and the IS weight is likelihood only — here we iterate uniformly over
// distinct leaves and must multiply by combinatorialPrior explicitly.
//
// Leaves are processed in descending prior order so the most probable
// candidates are evaluated first, enabling early exit when remaining prior
// mass is negligible.
func (r *RangeFinder) inferEnumerated(ctx context.Context) error {
	leaves := enumerateLeaves(r.inferenceBagMap, r.inference.RackLength)
	if len(leaves) == 0 {
		return nil
	}
	r.exhaustiveTotal = len(leaves) // record before any truncation or timeout

	// Sort descending by prior so early-exit (below) covers high-probability leaves.
	sort.Slice(leaves, func(i, j int) bool { return leaves[i].prior > leaves[j].prior })

	// Pre-truncate: for large leaf spaces, skip the lowest-prior tail that
	// contributes < enumerateEarlyExitEps of total prior mass.  Since likelihood ≤ 1,
	// the discarded posterior mass is bounded by the same fraction.
	//
	// For small spaces (≤ enumerateAlwaysFullThreshold), we always enumerate every
	// leaf — no truncation — so rare tiles (Q, J, Z, blank) are never silently dropped.
	// k=1 (≤27) and k=2 (≤378) always fall in this category.
	if len(leaves) > enumerateAlwaysFullThreshold {
		totalPrior := 0.0
		for _, l := range leaves {
			totalPrior += l.prior
		}
		keepThreshold := totalPrior * (1 - enumerateEarlyExitEps)
		cumulative := 0.0
		keepN := len(leaves)
		for i, l := range leaves {
			cumulative += l.prior
			if cumulative >= keepThreshold {
				keepN = i + 1
				break
			}
		}
		leaves = leaves[:keepN]
	}

	log.Info().
		Int("total-leaves", len(leaves)).
		Int("rack-length", r.inference.RackLength).
		Msg("exhaustive-inference-start")

	// Fan out over worker threads using a shared atomic index.
	var nextIdx atomic.Int64
	type threadResult struct {
		racks []montecarlo.InferredRack
	}
	results := make([]threadResult, r.threads)

	eg := errgroup.Group{}
	for t := 0; t < r.threads; t++ {
		t := t
		eg.Go(func() error {
			gc := r.gameCopies[t]
			opp := gc.PlayerOnTurn()
			simmer := r.aiplayers[t].(*simplesimmer.SimpleSimmer)

			// fullRack is reused across iterations to avoid per-leaf allocation.
			fullRack := make([]tilemapping.MachineLetter, len(r.lastOppMoveRackTiles)+r.inference.RackLength)
			copy(fullRack, r.lastOppMoveRackTiles)

			for {
				i := int(nextIdx.Add(1)) - 1
				if i >= len(leaves) {
					return nil
				}
				if ctx.Err() != nil {
					return nil
				}

				leaf := leaves[i]

				// Build the full deterministic rack: played tiles + this leave.
				// Passing a full RackTileLimit-length knownRack to SetRandomRack causes it
				// to: (1) put back the old rack, (2) remove fullRack from the bag, (3) draw
				// 0 additional tiles — giving us an exact, deterministic rack assignment.
				copy(fullRack[len(r.lastOppMoveRackTiles):], leaf.tiles)
				if _, err := gc.SetRandomRack(opp, fullRack); err != nil {
					return err
				}

				// Copy lastOppMove and set the correct leave for this iteration.
				lastOppMove := &move.Move{}
				lastOppMove.CopyFrom(r.lastOppMove)
				lastOppMove.SetLeave(leaf.tiles)

				if _, err := simmer.GenAndSim(context.Background(), 10, lastOppMove); err != nil {
					return err
				}
				r.simCount.Add(1)

				bestPlays := simmer.BestPlays().PlaysNoLock()
				// weight = prior * likelihood  (prior is explicit here; the sampling path omits it
				// because SetRandomRack already draws from the prior distribution).
				likelihoodP, _ := softmaxLikelihood(bestPlays, lastOppMove, gc.Board(), r.Tau())
				if likelihoodP <= 0 {
					continue
				}

				tiles := make([]tilemapping.MachineLetter, len(leaf.tiles))
				copy(tiles, leaf.tiles)
				results[t].racks = append(results[t].racks, montecarlo.InferredRack{
					Leave:  tiles,
					Weight: leaf.prior * likelihoodP,
				})
			}
		})
	}

	if err := eg.Wait(); err != nil {
		return err
	}

	for _, res := range results {
		r.inference.InferredRacks = append(r.inference.InferredRacks, res.racks...)
	}

	log.Info().
		Int("inferred-count", len(r.inference.InferredRacks)).
		Uint64("sim-count", r.simCount.Load()).
		Msg("exhaustive-inference-done")

	return nil
}
