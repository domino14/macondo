package montecarlo

import (
	"testing"

	"github.com/matryer/is"
)

// namedPlay describes a simmed play for the sorting tests. Pushing a single
// value into a Statistic makes Mean() return it exactly, which matters here:
// the win probabilities being ordered differ in their eighth decimal place.
type namedPlay struct {
	name    string
	winProb float64
	equity  float64
	ignored bool
}

func sortNamedPlays(plays []namedPlay, ignoredAtBottom bool) []string {
	names := map[*SimmedPlay]string{}
	simmed := make([]*SimmedPlay, len(plays))
	for i, p := range plays {
		sp := &SimmedPlay{}
		sp.winPctStats.Push(p.winProb)
		sp.equityStats.Push(p.equity)
		if p.ignored {
			sp.Ignore()
		}
		names[sp] = p.name
		simmed[i] = sp
	}

	s := &Simmer{simmedPlays: &SimmedPlays{plays: simmed}}
	s.sortPlaysByWinRate(ignoredAtBottom)

	sorted := make([]string, len(simmed))
	for i, sp := range s.simmedPlays.plays {
		sorted[i] = names[sp]
	}
	return sorted
}

// turn19Plays is the simmed field from woogles game PewhwF3WGB turn 19, the
// position reported in #502. Every play is a certain win -- they differ only
// from the eighth decimal place on -- while their equities span 8.6 points.
var turn19Plays = []namedPlay{
	{name: "L11 RETAX", winProb: 0.9999999996191397, equity: 8.704273079829118},
	{name: "N1 AY", winProb: 0.9999999894255265, equity: 17.31290228184371},
	{name: "3G A.YE", winProb: 0.9999999881373213, equity: 11.93},
	{name: "1M .AX", winProb: 0.9999999858521573, equity: 12.37},
	{name: "1M .YX", winProb: 0.99999993771813, equity: 14.12},
}

// TestSortPlaysByWinRate_DecidedRanksByEquity covers #502: the analyzer took
// the head of this list as the optimal play and so called the played RETAX
// optimal, docking nothing for the 8.6 points of equity it gave up.
func TestSortPlaysByWinRate_DecidedRanksByEquity(t *testing.T) {
	is := is.New(t)

	is.Equal(sortNamedPlays(turn19Plays, true), []string{
		"N1 AY",     // 17.31
		"1M .YX",    // 14.12
		"1M .AX",    // 12.37
		"3G A.YE",   // 11.93
		"L11 RETAX", // 8.70
	})
}

// TestSortPlaysByWinRate_ContestedRanksByWinProb guards the other direction:
// while the game is still in doubt, win probability outranks equity no matter
// how wide the equity gap is.
func TestSortPlaysByWinRate_ContestedRanksByWinProb(t *testing.T) {
	is := is.New(t)

	is.Equal(sortNamedPlays([]namedPlay{
		{name: "low win, high equity", winProb: 0.51, equity: 40},
		{name: "high win, low equity", winProb: 0.62, equity: -10},
		{name: "mid win, mid equity", winProb: 0.55, equity: 12},
	}, true), []string{
		"high win, low equity",
		"mid win, mid equity",
		"low win, high equity",
	})
}

// TestSortPlaysByWinRate_SaturatedGroupsStayOrdered checks that switching to
// equity inside the saturated groups doesn't let a certain loss climb over a
// contested play, or a contested play over a certain win.
func TestSortPlaysByWinRate_SaturatedGroupsStayOrdered(t *testing.T) {
	is := is.New(t)

	is.Equal(sortNamedPlays([]namedPlay{
		{name: "certain loss, huge equity", winProb: 0.0001, equity: 90},
		{name: "contested, no equity", winProb: 0.5, equity: -30},
		{name: "certain win, low equity", winProb: 0.9999, equity: 1},
		{name: "certain win, high equity", winProb: 0.99989, equity: 20},
		{name: "certain loss, less equity", winProb: 0.0002, equity: 10},
	}, true), []string{
		"certain win, high equity",
		"certain win, low equity",
		"contested, no equity",
		"certain loss, huge equity",
		"certain loss, less equity",
	})
}

// TestSortPlaysByWinRate_IgnoredStayAtBottom checks that pruned plays are still
// parked below the live ones, sorted among themselves.
func TestSortPlaysByWinRate_IgnoredStayAtBottom(t *testing.T) {
	is := is.New(t)

	is.Equal(sortNamedPlays([]namedPlay{
		{name: "pruned, best equity", winProb: 0.9999999999, equity: 30, ignored: true},
		{name: "live, worst equity", winProb: 0.9999999998, equity: 2},
		{name: "pruned, worse equity", winProb: 0.9999999997, equity: 20, ignored: true},
		{name: "live, best equity", winProb: 0.9999999996, equity: 5},
	}, true), []string{
		"live, best equity",
		"live, worst equity",
		"pruned, best equity",
		"pruned, worse equity",
	})
}

func TestWinProbsIndistinguishable(t *testing.T) {
	tests := []struct {
		name     string
		wi, wj   float64
		expected bool
	}{
		{"contested", 0.62, 0.55, false},
		{"float noise", 0.62, 0.62 + 1e-10, true},
		// #502: ten times winPctSortEpsilon apart, both certain wins.
		{"both certain wins", 0.9999999996191397, 0.9999999894255265, true},
		{"both certain losses", 0.0001, 0.004, true},
		{"certain win vs contested", 0.9999, 0.9, false},
		{"certain loss vs contested", 0.001, 0.4, false},
		{"straddling the win threshold", 0.9951, 0.9949, false},
	}

	for _, tt := range tests {
		got := winProbsIndistinguishable(tt.wi, tt.wj)
		if got != tt.expected {
			t.Errorf("%s: winProbsIndistinguishable(%v, %v) = %v, expected %v",
				tt.name, tt.wi, tt.wj, got, tt.expected)
		}
	}
}
