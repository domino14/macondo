package explainer

import (
	"math"
	"strings"
	"testing"

	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/matryer/is"
)

func near(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 0.01 {
		t.Errorf("got %.4f, want %.4f", got, want)
	}
}

// followup is one sampled follow-up play at a single score.
func followup(play string, pct float64, score int) *FollowupFact {
	return &FollowupFact{FollowupFamily: &stats.FollowupFamily{
		Play: play, Pct: pct, MinScore: score, MaxScore: score, TilePlay: true,
		Ways: []*stats.FollowupWay{{Play: play, Score: score, Pct: pct}},
	}}
}

// The figures are from a real position: C10 TIFO, whose ply-2 mean score - what
// an ordinary next turn is worth - was 34.05. Four ways to extend QUAD at 2F,
// none of which is worth building a turn around on its own, and which together
// are the biggest thing on the board.
func TestOneHookTakenFourWaysIsOneChance(t *testing.T) {
	is := is.New(t)
	const typical = 34.05

	fs := []*FollowupFact{
		followup("2F (QUAD)RUPLE", 4.91, 46),
		followup("2F (QUAD)RUPLY", 4.81, 52),
		followup("2F (QUAD)RUPED", 3.49, 48),
		followup("2F (QUAD)RUPOLE", 2.58, 48),
	}

	// Every one of them is big enough; not one of them is frequent enough.
	// This is the code being right about each play and wrong about the board.
	for _, f := range fs {
		upside, big := bigChance(f, typical)
		is.True(!big)
		is.True(upside < bigChanceMinUpside)
	}

	cs := clusterFollowups(fs, "C10 TIFO", typical)
	is.Equal(len(cs), 1)
	c := cs[0]
	is.Equal(len(c.Plays), 4)
	near(t, c.Pct, 15.79)
	near(t, c.AvgScore, 48.60)
	// The hook is 15.79% of the 34.05-point mean, so the turns without it are
	// worth 31.32, and the hook is what the other 2.73 points of the mean are:
	// baseline plus upside is the mean, always.
	near(t, c.Baseline, 31.32)
	near(t, c.Upside, 2.73)
	near(t, c.Baseline+c.Upside, typical)
	is.True(c.IsBigChance)

	// The likeliest play leads, so a coach naming one names the one a reader
	// is most likely to have.
	is.Equal(c.Plays[0].Play, "2F (QUAD)RUPLE")
	is.Equal(c.MinScore, 46)
	is.Equal(c.MaxScore, 52)
	is.Equal(c.Label(), "(QUAD) in row 2")
}

// A chance can be so reliable that it becomes its own baseline, and judging it
// against the play's mean next turn then asks it to be a fifth bigger than
// itself. These are the sampled follow-ups after N10 ME(E)T, which keeps QU
// for a hook at B2 worth 56 points 63% of the time: the mean next turn is
// 48.58 *because of* that hook, and the old rule scored the biggest thing on
// the board at zero for it.
func TestADominantChanceIsStillAChance(t *testing.T) {
	is := is.New(t)
	const mean = 48.58

	cs := clusterFollowups([]*FollowupFact{
		followup("B2 Q(U)AFF", 32.34, 56),
		followup("B2 Q(U)IFF", 30.97, 56),
		followup("B2 N(U)FF", 2.07, 20),
		followup("B2 B(U)FF", 1.24, 24),
		followup("B2 Q(U)OD", 1.16, 28),
		followup("B2 C(U)FF", 1.16, 24),
		followup("B2 B(U)FFO", 0.63, 30),
		followup("B2 B(U)FFY", 0.55, 48),
		followup("B2 G(U)FF", 0.55, 22),
		followup("B2 Q(U)ERN", 0.51, 32),
	}, "N10 ME(E)T", mean)

	is.Equal(len(cs), 1)
	c := cs[0]
	is.Equal(c.Label(), "(U) in column B")
	near(t, c.Pct, 71.18)
	near(t, c.AvgScore, 52.69)

	// The hook is most of the mean, so it cannot clear a bar set at 1.25 times
	// that mean. This is the whole bug, kept here as an assertion.
	is.True(c.AvgScore < mean*bigChanceScoreRatio)

	// Taken back out, the turns without it are worth 38, and the hook is what
	// the other ten points of the mean are.
	near(t, c.Baseline, 38.43)
	near(t, c.Upside, 10.15)
	near(t, c.Baseline+c.Upside, mean)
	is.True(c.IsBigChance)
}

// Two hooks of the same size hide behind each other when each is taken out of
// the mean on its own: the baseline the first one is measured against still
// contains the second. Taking both out at once is what makes the pair add up
// to what it is worth.
func TestTwoBigChancesDoNotHideBehindEachOther(t *testing.T) {
	is := is.New(t)
	// 40% at 60, 40% at 60, 20% at 20 - so the mean next turn is 52, and a
	// turn with neither hook in it is worth 20.
	const mean = 52.0

	fs := []*FollowupFact{
		followup("2F (QUAD)RUPLE", 40, 60),
		followup("1H (Z)WIEBACK", 40, 60),
		followup("3L PURL", 20, 20),
	}

	// One at a time: each hook is weighed against a turn that still has the
	// other hook in it, which is nearly as good, so each looks like a third of
	// what it is.
	marginal, big := bigChance(fs[0], mean)
	is.True(big)
	near(t, marginal, 5.33)

	cs := clusterFollowups(fs, "best", mean)
	is.Equal(len(cs), 3)
	byPlay := map[string]*FollowupCluster{}
	for _, c := range cs {
		byPlay[c.Label()] = c
	}
	quad, zwie, junk := byPlay["2F (QUAD)RUPLE"], byPlay["1H (Z)WIEBACK"], byPlay["3L PURL"]

	// Both out at once, and the baseline is the 20-point turn that is what's
	// actually left when neither hook comes up.
	near(t, quad.Baseline, 20)
	near(t, quad.Upside, 16)
	near(t, zwie.Upside, 16)
	is.True(quad.IsBigChance)
	is.True(zwie.IsBigChance)
	is.True(marginal*2 < quad.Upside+zwie.Upside)

	// Which is the property worth having: the upsides account for the whole
	// gap between the turn this play gets and the turn it would get with
	// neither hook available.
	near(t, quad.Upside+zwie.Upside, mean-quad.Baseline)

	// And the junk that makes up the baseline is not itself promoted by the
	// baseline dropping to it.
	is.True(!junk.IsBigChance)
}

// The same frequency and a third of the size is not the same finding. These
// are the other five follow-ups from that position, and clustering them must
// not turn an ordinary turn into an opportunity.
func TestClusteringDoesNotInventChances(t *testing.T) {
	is := is.New(t)
	const typical = 34.05

	cs := clusterFollowups([]*FollowupFact{
		followup("3L PURL", 6.04, 23),
		followup("3L YUP", 2.71, 28),
		followup("3L PULP", 2.52, 27),
		followup("3L PUP", 2.32, 25),
		followup("3L PUR", 2.13, 21),
	}, "C10 TIFO", typical)

	is.Equal(len(cs), 1)
	near(t, cs[0].Pct, 15.72) // as often as the QUAD hook
	near(t, cs[0].AvgScore, 24.53)
	is.True(!cs[0].IsBigChance) // and worth less than an ordinary turn
	// With no hook to name the spot by, a bare "3L" would say nothing, so the
	// words are named instead - and the tail is counted rather than listed.
	is.Equal(cs[0].Label(), "3L PURL / YUP / PULP / +2 more")
}

// Two unrelated words in one empty lane share a square and nothing else, so
// the square is all they have in common - and "J9" on its own tells a reader
// nothing about what the chance is. These are real plays from a real position.
func TestAnOpenSquareIsNamedByItsWords(t *testing.T) {
	is := is.New(t)

	cs := clusterFollowups([]*FollowupFact{
		followup("J9 CAPERED", 1.59, 82),
		followup("J9 PEASCOD", 1.45, 81),
	}, "12K QU(ID)", 45)

	is.Equal(len(cs), 1)
	is.Equal(cs[0].Through, "") // nothing on the board to hang it on
	is.Equal(cs[0].Label(), "J9 CAPERED / PEASCOD")

	// A hook keeps its own name: the tiles are the point, and every way of
	// taking them is listed underneath anyway.
	cs = clusterFollowups([]*FollowupFact{
		followup("B8 (R)EPLACED", 2.79, 88),
		followup("B8 (R)ESPACED", 2.54, 80),
	}, "12K QU(ID)", 45)
	is.Equal(cs[0].Through, "R")
	is.Equal(cs[0].Label(), "(R) in column B")
}

// A front extension starts on a different square for every length of prefix,
// so keying on where the play begins would split one hook into several. What
// stays put is the tiles already on the board.
func TestExtensionsGroupOnThePlaythrough(t *testing.T) {
	is := is.New(t)

	// ZOIC sits at row 8, columns F-I. CENOZOIC starts a column later than
	// SAPROZOIC does, and both run through the same four tiles.
	cs := clusterFollowups([]*FollowupFact{
		followup("8B CENO(ZOIC)", 3.0, 48),
		followup("8A SAPRO(ZOIC)", 2.0, 52),
	}, "best", 30)
	is.Equal(len(cs), 1)
	is.Equal(len(cs[0].Plays), 2)
	is.Equal(cs[0].Label(), "(ZOIC) in row 8")

	// Extending in both directions at once, from a shared starting square.
	cs = clusterFollowups([]*FollowupFact{
		followup("5A HI(THERMOS)T", 3.0, 45),
		followup("5A NE(THERMOS)T", 2.0, 45),
	}, "best", 30)
	is.Equal(len(cs), 1)
	is.Equal(cs[0].Label(), "(THERMOS) in row 5")

	// Columns work the same way, and are named the way a player names them.
	cs = clusterFollowups([]*FollowupFact{
		followup("C6 CENO(ZOIC)", 3.0, 48),
		followup("C5 SAPRO(ZOIC)", 2.0, 52),
	}, "best", 30)
	is.Equal(len(cs), 1)
	is.Equal(cs[0].Label(), "(ZOIC) in column C")
}

// Grouping has to stop somewhere: two hooks are two chances even when they
// share a row, a square, or a set of letters.
func TestUnrelatedPlaysStayApart(t *testing.T) {
	is := is.New(t)

	cs := clusterFollowups([]*FollowupFact{
		followup("8B CENO(ZOIC)", 3.0, 48),  // row 8, through ZOIC
		followup("9B CENO(ZOIC)", 3.0, 48),  // a different row
		followup("8B (QUAD)RUPLE", 3.0, 46), // same square, different tiles
		followup("B8 CENO(ZOIC)", 3.0, 48),  // same letters, going down
		followup("3L PURL", 3.0, 23),        // no playthrough at all
		followup("3M PURL", 3.0, 23),        // nor here, one square over
	}, "best", 30)
	is.Equal(len(cs), 6)

	// Exchanges and passes are not places on a board and never join anything.
	notPlacements := []*FollowupFact{
		followup("(exch AEI)", 4.0, 0),
		followup("(exch OU)", 3.0, 0),
		followup("(Pass)", 2.0, 0),
	}
	for _, f := range notPlacements {
		f.TilePlay = false
	}
	cs = clusterFollowups(notPlacements, "best", 30)
	is.Equal(len(cs), 3)
}

// The cluster is the finding, but the plays are what a reader writes on a
// board, so both have to reach the prompt.
func TestAClusterIsReportedWithItsPlays(t *testing.T) {
	is := is.New(t)

	cs := clusterFollowups([]*FollowupFact{
		followup("2F (QUAD)RUPLE", 4.91, 46),
		followup("2F (QUAD)RUPLY", 4.81, 52),
		followup("2F (QUAD)RUPED", 3.49, 48),
		followup("2F (QUAD)RUPOLE", 2.58, 48),
	}, "C10 TIFO", 34.05)

	out := renderChances("### Chances", &montecarlo.CandidateStats{
		Play: "C10 TIFO", Leave: "PRU"}, chancesBaseline(cs, 34.05), cs)

	// The opportunity, sized whole, against the turns it doesn't come up in.
	is.True(strings.Contains(out, "(QUAD) in row 2"))
	is.True(strings.Contains(out, "46-52 pts"))
	is.True(strings.Contains(out, "15.79% of the time"))
	is.True(strings.Contains(out, "upside +2.7"))
	is.True(strings.Contains(out, "none of these chances comes up - is worth about 31"))

	// And every way of taking it, named and priced.
	for _, play := range []string{"RUPLE", "RUPLY", "RUPED", "RUPOLE"} {
		is.True(strings.Contains(out, "2F (QUAD)"+play))
	}
	is.True(strings.Contains(out, "52 pts   4.81% of the time"))

	// A lone play is still reported as itself, with no breakdown to expand.
	one := clusterFollowups([]*FollowupFact{
		followup("15G PREAD(JUST)", 11.5, 57),
	}, "12K QU(ID)", 34.05)
	out = renderChances("### Chances", &montecarlo.CandidateStats{
		Play: "12K QU(ID)"}, chancesBaseline(one, 34.05), one)
	is.True(strings.Contains(out, "15G PREAD(JUST)"))
	is.True(!strings.Contains(out, "in row 15"))
	is.Equal(strings.Count(out, "15G PREAD(JUST)"), 1)
}

// A hook no single spelling reaches often enough is still a hook our play
// created. Setups are judged per opportunity for the same reason chances are.
func TestASetupSplitAcrossSpellingsIsStillASetup(t *testing.T) {
	is := is.New(t)

	// Neither play alone clears setupMinPct; together they are well past it.
	fs := []*FollowupFact{
		followup("8B CENO(ZOIC)", 3.5, 48),
		followup("8A SAPRO(ZOIC)", 3.0, 52),
	}
	for _, f := range fs {
		f.WayRequirements = []string{"requires us to play 12K QU(ID) first"}
		is.True(f.Pct < setupMinPct)
	}

	cs := clusterFollowups(fs, "12K QU(ID)", 30)
	is.Equal(len(cs), 1)
	is.True(cs[0].IsSetup)
	is.True(cs[0].Worthwhile())

	// The requirement still has to name our play. A hook that was there all
	// along is not something we set up.
	for _, f := range fs {
		f.WayRequirements = []string{"none"}
	}
	cs = clusterFollowups(fs, "12K QU(ID)", 30)
	is.True(!cs[0].IsSetup)
}
