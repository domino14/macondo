package montecarlo

import (
	"sort"

	"github.com/domino14/macondo/stats"
	"github.com/rs/zerolog/log"
)

const IterationsCutoff = 5000

// use stats to figure out when to stop simming.

func shouldStop(plays []*SimmedPlay, sc StoppingCondition, iterationCount int) bool {
	// This function runs as the sim is ongoing. So we should be careful
	// what we do with memory here.
	if len(plays) < 2 {
		return true
	}
	if iterationCount > IterationsCutoff {
		return true
	}
	// Otherwise, do some statistics.
	// shallow copy the array so we can sort it/play with it.
	c := make([]*SimmedPlay, len(plays))
	// count ignored plays
	ignoredPlays := 0
	for i := range c {
		c[i] = plays[i]
		c[i].RLock()
		if c[i].ignore {
			ignoredPlays++
		}
		c[i].RUnlock()
	}
	if ignoredPlays >= len(c)-1 {
		// if there is only 1 unignored play, exit.
		return true
	}

	// sort copy by win pct.
	sort.Slice(c, func(i, j int) bool {
		c[i].RLock()
		c[j].RLock()
		defer c[j].RUnlock()
		defer c[i].RUnlock()
		if c[i].winPctStats.Mean() == c[j].winPctStats.Mean() {
			return c[i].equityStats.Mean() > c[j].equityStats.Mean()
		}
		return c[i].winPctStats.Mean() > c[j].winPctStats.Mean()
	})

	// we want to cut off plays that have no chance of winning.
	// assume the very top play is the winner, and then cut off plays that have
	// no chance of catching up.
	// "no chance" is of course defined by the stopping condition :)

	var ci float64
	switch sc {
	case Stop95:
		ci = stats.Z95
	case Stop98:
		ci = stats.Z98
	case Stop99:
		ci = stats.Z99
	}

	tentativeWinner := c[0]
	tentativeWinner.RLock()
	μ := tentativeWinner.winPctStats.Mean()
	e := tentativeWinner.winPctStats.StandardError(ci)
	tentativeWinner.RUnlock()
	newIgnored := 0
	// assume standard normal distribution (?)
	for _, p := range c[1:] {
		p.RLock()
		if p.ignore {
			p.RUnlock()
			continue
		}
		μi := p.winPctStats.Mean()
		ei := p.winPctStats.StandardError(ci)
		p.RUnlock()
		if passTest(μ, e, μi, ei) {
			p.Ignore()
			newIgnored++
		}
	}
	if newIgnored > 0 {
		log.Debug().Int("newIgnored", newIgnored).Msg("sim-cut-off")
	}
	if ignoredPlays+newIgnored >= len(c)-1 {
		// if there is only 1 unignored play, exit.
		return true
	}
	return false
}

// passTest: determine if a random variable X > Y with the given
// confidence level; return true if X > Y.
func passTest(μ, e, μi, ei float64) bool {
	// Z := zVal(μ, v, μi, vi)
	// X > Y if (μ - e) > (μi + ei)
	return (μ - e) > (μi + ei)
}

// func zVal(μ, v, μi, vi float64) float64 {
// 	// mean of X - Y = E(X-Y) = E(X) - E(Y)
// 	mean := μ - μi
// 	// variance of (X-Y) = V(X) + V(Y)
// 	variance := v + vi
// 	stdev := math.Sqrt(variance)
// 	// P(X > Y) = P(X - Y > 0)
// 	// let D = X - Y
// 	// then P(D > 0)
// 	// convert to standard normal variable (mean 0 stdev 1)
// 	// = P ((D - mean) / (stdev) > (0 - mean) / stdev)
// 	// then P(Z>(0 - mean)/stdev)
// 	// 95 percentile is Z 1.96
// 	// 99 percentile is Z 2.58
// 	return -mean / stdev
// }

// func zValStdev(μ, s, μi, si float64) float64 {
// 	return zVal(μ, s*s, μi, si*si)
// }
