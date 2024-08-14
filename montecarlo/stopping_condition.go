package montecarlo

import (
	"math"
	"sort"
	"sync"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/stats"
)

type StoppingCondition int

const (
	StopNone StoppingCondition = iota
	Stop90
	Stop95
	Stop98
	Stop99
	Stop999
)

const StopConditionCheckInterval = 128

const (
	defaultIterationsCutoff             = 2000
	defaultPerPlyStopScaling            = 625
	defaultSimilarPlaysIterationsCutoff = 750
	defaultMinReasonableWProb           = 0.005 // or 0.5%

)

type AutoStopperSimmedPlay struct {
	equityStats stats.Statistic
	winPctStats stats.Statistic
	ignore      bool
	origPlay    *SimmedPlay
}

// use stats to figure out when to stop simming.

type AutoStopper struct {
	sync.Mutex
	iterationsCutoff             int
	perPlyStopScaling            int
	similarPlaysIterationsCutoff int
	minReasonableWProb           float64

	stoppingCondition   StoppingCondition
	playSimilarityCache map[string]bool

	simmedPlays []*AutoStopperSimmedPlay
}

// copyForStatCutoff copies parts of the simmed plays array. We only deep-copy
// what we actually need (which are a subset of the stats and the ignore attribute)
// to save time.
func (a *AutoStopper) copyForStatCutoff(plays *SimmedPlays) {
	plays.RLock()
	defer plays.RUnlock()

	if len(a.simmedPlays) == 0 {
		a.simmedPlays = make([]*AutoStopperSimmedPlay, len(plays.plays))
		for i := range a.simmedPlays {
			a.simmedPlays[i] = &AutoStopperSimmedPlay{}
		}
	}

	for i := range a.simmedPlays {
		a.simmedPlays[i].equityStats = plays.plays[i].equityStats
		a.simmedPlays[i].winPctStats = plays.plays[i].winPctStats
		a.simmedPlays[i].ignore = plays.plays[i].ignore
		a.simmedPlays[i].origPlay = plays.plays[i]
	}
}

func newAutostopper() *AutoStopper {
	return &AutoStopper{
		iterationsCutoff:             defaultIterationsCutoff,
		perPlyStopScaling:            defaultPerPlyStopScaling,
		similarPlaysIterationsCutoff: defaultSimilarPlaysIterationsCutoff,
		minReasonableWProb:           defaultMinReasonableWProb,
		stoppingCondition:            StopNone,
		playSimilarityCache:          map[string]bool{},
	}
}

func (a *AutoStopper) reset() {
	a.playSimilarityCache = map[string]bool{}
}

func (a *AutoStopper) shouldStop(iterationCount uint64, simmedPlays *SimmedPlays, maxPlies int) bool {
	// This function runs as the sim is ongoing. So we should be careful
	// what we do with memory here.
	t := time.Now()
	a.Lock()
	defer a.Unlock()

	sc := a.stoppingCondition

	if len(simmedPlays.plays) < 2 {
		return true
	}
	if int(iterationCount) > a.iterationsCutoff+(maxPlies*a.perPlyStopScaling) {
		return true
	}
	// Otherwise, do some statistics.

	a.copyForStatCutoff(simmedPlays)

	// count ignored plays
	ignoredPlays := 0
	bottomUnignoredWinPct := 0.0
	bottomUnignoredSerr := 0.0
	for i := range a.simmedPlays {
		if a.simmedPlays[i].ignore {
			ignoredPlays++
		}
	}
	if ignoredPlays >= len(a.simmedPlays)-1 {
		// if there is only 1 unignored play, exit.
		return true
	}

	// sort copy by win pct.
	sort.Slice(a.simmedPlays, func(i, j int) bool {
		// move ignored plays to the bottom.
		if a.simmedPlays[i].ignore {
			return false
		}
		if a.simmedPlays[j].ignore {
			return true
		}
		if a.simmedPlays[i].winPctStats.Mean() == a.simmedPlays[j].winPctStats.Mean() {
			return a.simmedPlays[i].equityStats.Mean() > a.simmedPlays[j].equityStats.Mean()
		}
		return a.simmedPlays[i].winPctStats.Mean() > a.simmedPlays[j].winPctStats.Mean()
	})

	// find the bottom unignored win pct play
	for i := len(a.simmedPlays) - 1; i >= 0; i-- {
		if !a.simmedPlays[i].ignore {
			bottomUnignoredWinPct = a.simmedPlays[i].winPctStats.Mean()
			bottomUnignoredSerr = a.simmedPlays[i].winPctStats.StandardError()
			break
		}
	}

	// we want to cut off plays that have no chance of winning.
	// assume the very top play is the winner, and then cut off plays that have
	// no chance of catching up.
	// "no chance" is of course defined by the stopping condition :)

	var ci float64
	switch sc {
	case Stop90:
		ci = stats.Z90
	case Stop95:
		ci = stats.Z95
	case Stop98:
		ci = stats.Z98
	case Stop99:
		ci = stats.Z99
	case Stop999:
		ci = stats.Z999
	}
	tiebreakByEquity := false
	tentativeWinner := a.simmedPlays[0]
	μ := tentativeWinner.winPctStats.Mean()
	e := tentativeWinner.winPctStats.StandardError()

	if zTest(float64(a.minReasonableWProb), μ, e, -ci, true) {
		// If the top play by win % has basically no win chance, tiebreak the whole
		// thing by equity.
		tiebreakByEquity = true
	} else if zTest(1-a.minReasonableWProb, μ, e, ci, false) &&
		zTest(1-a.minReasonableWProb, bottomUnignoredWinPct, bottomUnignoredSerr, ci, false) {
		// If the top play by win % has basically no losing chance, check if the bottom
		// play also has no losing chance
		tiebreakByEquity = true
	}

	if tiebreakByEquity {
		// We may need to re-determine the tentative winner.
		highestEquity := -1000000.0
		highestEquityIdx := -1
		for idx, p := range a.simmedPlays {
			eq := p.equityStats.Mean()
			if eq > highestEquity {
				highestEquityIdx = idx
				highestEquity = eq
			}
		}
		if highestEquityIdx != 0 {
			a.simmedPlays[0], a.simmedPlays[highestEquityIdx] = a.simmedPlays[highestEquityIdx], a.simmedPlays[0]
			tentativeWinner = a.simmedPlays[0]
			log.Info().
				Str("old-tentative-winner", a.simmedPlays[highestEquityIdx].origPlay.play.ShortDescription()).
				Str("tentative-winner", tentativeWinner.origPlay.play.ShortDescription()).
				Msg("tiebreaking by equity, re-determining tentative winner")
		}
		μ = tentativeWinner.equityStats.Mean()
		e = tentativeWinner.equityStats.StandardError()
		log.Debug().Msg("stopping-condition-tiebreak-by-equity")
	}

	newIgnored := 0
	// assume standard normal distribution (?)
	for _, p := range a.simmedPlays[1:] {
		if p.ignore {
			continue
		}
		μi := p.winPctStats.Mean()
		ei := p.winPctStats.StandardError()
		if tiebreakByEquity {
			μi = p.equityStats.Mean()
			ei = p.equityStats.StandardError()
		}
		if welchTest(μ, e, μi, ei, ci) {
			p.origPlay.Ignore()
			newIgnored++
		} else if iterationCount > uint64(a.similarPlaysIterationsCutoff) {
			if materiallySimilar(tentativeWinner, p, a.playSimilarityCache) {
				p.origPlay.Ignore()
				newIgnored++
			}
		}
	}
	if newIgnored > 0 {
		log.Debug().Int("newIgnored", newIgnored).Msg("sim-cut-off")
	}
	log.Debug().Dur("time-elapsed-ms", time.Since(t)).Msg("time-for-cutoff-alg")

	// if there is only 1 unignored play, exit.
	return ignoredPlays+newIgnored >= len(a.simmedPlays)-1
}

// welchTest: determine if a random variable X > Y with the given z-score; return true if X > Y.
// μ and e are the mean and standard error of variable X
// μi, ei are the mean and standard error of variable Y
func welchTest(μ, e, μi, ei, z float64) bool {
	sediff := math.Sqrt(e*e + ei*ei)
	if sediff == 0 {
		return true
	}
	zcalc := (μ - μi) / sediff
	return zcalc > z
}

// zTest does a Z-test. M, e are the mean/stderror for the variable we're testing. (sample mean)
// μ is the population mean.
func zTest(μ, M, e, z float64, sgnflip bool) bool {
	zcalc := (M - μ) / e
	if !sgnflip {
		return zcalc > z
	} else {
		return z > zcalc
	}
}

func materiallySimilar(p1, p2 *AutoStopperSimmedPlay, pcache map[string]bool) bool {

	p1ps := p1.origPlay.play.ShortDescription()
	p2ps := p2.origPlay.play.ShortDescription()
	if p1ps > p2ps {
		p1ps, p2ps = p2ps, p1ps
	}
	lookupstr := p1ps + "|" + p2ps
	if similar, ok := pcache[lookupstr]; ok {
		log.Trace().Str("lookupstr", lookupstr).
			Bool("similar", similar).
			Msg("in-similarity-cache")
		return similar
	}

	// two plays are "materially similar" if they use the same tiles and
	// start at the same square.
	p1r, p1c, p1v := p1.origPlay.play.CoordsAndVertical()
	p2r, p2c, p2v := p2.origPlay.play.CoordsAndVertical()

	if !(p1r == p2r && p1c == p2c && p1v == p2v) {
		pcache[lookupstr] = false
		return false
	}
	if p1.origPlay.play.TilesPlayed() != p2.origPlay.play.TilesPlayed() {
		pcache[lookupstr] = false
		return false
	}
	if len(p1.origPlay.play.Tiles()) != len(p2.origPlay.play.Tiles()) {
		pcache[lookupstr] = false
		return false
	}
	// these plays start at the same square and are the same length.
	// do they use the same tiles?
	a1 := make([]tilemapping.MachineLetter, len(p1.origPlay.play.Tiles()))
	a2 := make([]tilemapping.MachineLetter, len(p2.origPlay.play.Tiles()))
	copy(a1, p1.origPlay.play.Tiles())
	copy(a2, p2.origPlay.play.Tiles())
	sort.Slice(a1, func(i, j int) bool { return a1[i] < a1[j] })
	sort.Slice(a2, func(i, j int) bool { return a2[i] < a2[j] })
	for i := range a1 {
		if a1[i] != a2[i] {
			pcache[lookupstr] = false
			return false
		}
	}
	log.Debug().Str("lookupstr", lookupstr).Msg("materially-similar")
	pcache[lookupstr] = true
	return true
}
