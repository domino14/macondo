package main

import (
	"sync/atomic"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
)

// -readmarg theta tells the sim only the tiles the bot's read is confident
// of: every tile (and every repeat of it) that the read's leaves hold with
// total weight at least theta. By default the rest of the opponent's rack is
// drawn as in a plain sim; with -readmargcond the read keeps those of its
// leaves that hold the confident tiles. A read with no confident tile is left
// as it is. The partial-oracle screens ask how much of the oracle's gain a
// few right tiles carry; this is the version our own read can give.
var (
	readMargTheta     float64
	readMargCondition bool
)

var readMargSeen, readMargForced atomic.Int64 // reads seen and changed, for the log

func margRead(g *game.Game, racks []montecarlo.InferredRack) []montecarlo.InferredRack {
	readMargSeen.Add(1)
	// atLeast[t][c-1] is the weight of the leaves holding at least c of t.
	var atLeast [64][7]float64
	total := 0.0
	for _, r := range racks {
		var counts [64]int
		for _, t := range r.Leave {
			if counts[t] < 7 {
				atLeast[t][counts[t]] += r.Weight
			}
			counts[t]++
		}
		total += r.Weight
	}
	if total <= 0 {
		return racks
	}
	var forced []tilemapping.MachineLetter
	for t := range atLeast {
		for c := range atLeast[t] {
			if atLeast[t][c] < readMargTheta*total {
				break
			}
			forced = append(forced, tilemapping.MachineLetter(t))
		}
	}
	if len(forced) == 0 {
		return racks
	}
	readMargForced.Add(1)
	substFor.Store(g, &substitution{Sub: tilemapping.MachineWord(forced).UserVisible(g.Alphabet()), Dist: float64(len(forced))})
	if !readMargCondition {
		return []montecarlo.InferredRack{{Leave: forced, Weight: 1}}
	}
	var need [64]int
	for _, t := range forced {
		need[t]++
	}
	out := make([]montecarlo.InferredRack, 0, len(racks))
	for _, r := range racks {
		var counts [64]int
		for _, t := range r.Leave {
			counts[t]++
		}
		ok := true
		for t, n := range need {
			if counts[t] < n {
				ok = false
				break
			}
		}
		if ok {
			out = append(out, r)
		}
	}
	if len(out) == 0 {
		return racks
	}
	return out
}
