package explainer

import (
	"slices"
	"strings"
	"testing"

	"github.com/domino14/macondo/montecarlo"
	"github.com/matryer/is"
)

func TestDottedPlay(t *testing.T) {
	is := is.New(t)

	// Playthrough parentheses become the dots the move parser wants, one per
	// tile already on the board.
	is.Equal(DottedPlay("5D (S)PIC(A)"), "5D .PIC.")
	is.Equal(DottedPlay("12A (D)ISCOMBOB(U)LATE"), "12A .ISCOMBOB.LATE")
	is.Equal(DottedPlay("1H (Z)WIEBAcK"), "1H .WIEBAcK") // blank case survives
	is.Equal(DottedPlay("15G PREADJUST"), "15G PREADJUST")
	// Already-dotted input is left alone, so either notation works on -vs.
	is.Equal(DottedPlay("5D .PIC."), "5D .PIC.")
	// Exchanges and passes only lose their wrapping.
	is.Equal(DottedPlay("(exch QU)"), "exch QU")
	is.Equal(DottedPlay("exchange QU"), "exchange QU")
	is.Equal(DottedPlay("pass"), "pass")
	is.Equal(DottedPlay("(Pass)"), "pass")
	is.Equal(DottedPlay("  8H QUIXOTIC  "), "8H QUIXOTIC")
}

// The play we were asked about has to survive the cut however badly it did -
// the whole point is to explain a move that lost.
func TestTrimCandidatesKeepsTheComparison(t *testing.T) {
	is := is.New(t)

	candidates := make([]montecarlo.CandidateStats, 20)
	for i := range candidates {
		candidates[i] = montecarlo.CandidateStats{Play: "play" + string(rune('A'+i))}
	}

	kept, idx := trimCandidates(candidates, 17)
	is.Equal(len(kept), candidatesShown+1)
	is.Equal(idx, candidatesShown)
	is.Equal(kept[idx].Play, "playR")
	is.Equal(kept[0].Play, "playA") // the best play is still first

	// A play already inside the cut doesn't get duplicated.
	kept, idx = trimCandidates(candidates, 3)
	is.Equal(len(kept), candidatesShown)
	is.Equal(idx, 3)

	// No comparison at all, and short lists, are left alone.
	kept, idx = trimCandidates(candidates, -1)
	is.Equal(len(kept), candidatesShown)
	is.Equal(idx, -1)
	kept, idx = trimCandidates(candidates[:3], 2)
	is.Equal(len(kept), 3)
	is.Equal(idx, 2)

	// Trimming must not write through into the original slice: the caller
	// takes pointers into what comes back.
	kept, _ = trimCandidates(candidates, 17)
	kept[candidatesShown].Play = "clobbered"
	is.Equal(candidates[candidatesShown].Play, "playI")
}

func TestFollowupDiff(t *testing.T) {
	is := is.New(t)

	setup := func(play string, pct float64, score int) *FollowupFact {
		return &FollowupFact{
			FollowupFamily: fam(play, pct, score, score),
			IsSetup:        true,
		}
	}
	dull := func(play string) *FollowupFact {
		return &FollowupFact{FollowupFamily: fam(play, 1.0, 12, 12)}
	}

	best := []*FollowupFact{setup("15G PREAD(JUST)", 11.5, 57), dull("2B PI(T)")}
	rival := []*FollowupFact{setup("5D (S)CAP(A)", 8.0, 44), dull("2B PI(T)")}

	onlyBest, onlyRival := followupDiff(best, rival)
	is.Equal(len(onlyBest), 1)
	is.Equal(onlyBest[0].Play, "15G PREAD(JUST)")
	is.Equal(len(onlyRival), 1)
	is.Equal(onlyRival[0].Play, "5D (S)CAP(A)")

	// A play both sides can make isn't something either one gave up, even
	// when the model would spell it differently.
	shared := []*FollowupFact{setup("i8 skiwear", 11.0, 60)}
	both := []*FollowupFact{setup("I8 sKIWEAR", 11.0, 60)}
	onlyBest, onlyRival = followupDiff(shared, both)
	is.Equal(len(onlyBest), 0)
	is.Equal(len(onlyRival), 0)
}

// comparedFacts is fakeFacts with a head-to-head attached.
func comparedFacts(wasBest bool) *PositionFacts {
	f := fakeFacts()
	c := &Comparison{
		Play:        f.Candidates[1].Play,
		FromHistory: true,
		Rival:       &f.Candidates[1],
		Deltas: Deltas{
			WinPct: 7.7, Equity: -1.5, Score: -2, LeaveValue: 3.2,
			OppMeanScore: -3.7, OppBingoPct: -1.0,
			OurMeanScore: 10.4, OurBingoPct: 3.0,
			Established: true,
		},
		OnlyBest: []*FollowupFact{f.Followups[0]},
	}
	if wasBest {
		c.Play = f.Candidates[0].Play
		c.WasBest = true
	}
	f.Comparison = c
	f.Flags = computeFlags(f)
	return f
}

func TestComparisonPrompt(t *testing.T) {
	is := is.New(t)
	f := comparedFacts(false)

	is.True(f.Flags["has_comparison"])
	is.True(!f.Flags["comparison_was_best"])

	p, err := BuildPrompt(f, false)
	is.NoErr(err)
	is.True(slices.Contains(p.Concepts, "comparison"))
	is.True(!slices.Contains(p.Concepts, "comparison-was-best"))

	// The head-to-head arithmetic is done for the model, signed so that
	// positive always means the recommended play is ahead.
	is.True(strings.Contains(p.User, "### Head to head: 12K QU(ID) versus 5D (S)CAP(A)"))
	is.True(strings.Contains(p.User, "the play the reader actually made"))
	is.True(strings.Contains(p.User, "+7.70"))
	is.True(strings.Contains(p.User, "-3.70"))
	is.True(strings.Contains(p.User, "The win% gap is established"))
	is.True(strings.Contains(p.User, "Chances only in the sampled follow-ups after 12K QU(ID)"))
	// The upside contrast is the figure that carries "you gave something up".
	is.True(strings.Contains(p.User, "Big follow-up chances"))

	// And the question being asked changes accordingly.
	is.True(strings.Contains(p.User, "why 12K QU(ID) beats 5D (S)CAP(A), the play they played"))
}

// An overlapping interval means the simulation has not shown a difference, and
// the prompt has to say so rather than let the model invent a reason.
func TestUnestablishedGapIsFlagged(t *testing.T) {
	is := is.New(t)
	f := comparedFacts(false)
	f.Comparison.Deltas.Established = false

	p, err := BuildPrompt(f, false)
	is.NoErr(err)
	is.True(strings.Contains(p.User, "The win% gap is NOT established"))
	is.True(strings.Contains(p.User, "has not shown one of these plays to be better"))
}

func TestComparisonWhenTheyFoundTheBestPlay(t *testing.T) {
	is := is.New(t)
	f := comparedFacts(true)

	is.True(f.Flags["has_comparison"])
	is.True(f.Flags["comparison_was_best"])

	p, err := BuildPrompt(f, false)
	is.NoErr(err)
	is.True(slices.Contains(p.Concepts, "comparison-was-best"))
	is.True(strings.Contains(p.User, "### Head to head: 12K QU(ID) is the top play"))
	is.True(strings.Contains(p.User, "the runner-up, 5D (S)CAP(A)"))
	is.True(strings.Contains(p.User, "is the right one here, and what it beat"))
}

// End to end on a real simulation: a play that ranks badly still gets the full
// treatment - it is forced into the sim, kept through the trim, and given its
// own follow-up and lane analysis.
func TestComparisonEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("runs a simulation")
	}
	is := is.New(t)
	an, simmer, simStats := simulate(t, examplePosition, 14, 3, 120)

	all := simmer.CandidateStats()
	is.True(len(all) > candidatesShown) // otherwise the trim isn't exercised
	worst := all[len(all)-1]

	f, err := an.BuildFacts(simmer, simStats, &ComparisonRequest{Move: worst.Move, FromHistory: true}, nil)
	is.NoErr(err)
	is.True(f.Comparison != nil)

	c := f.Comparison
	is.Equal(c.Play, worst.Play)
	is.True(c.FromHistory)
	is.True(!c.WasBest)
	is.Equal(c.Rival.Play, worst.Play)

	// It survived the cut even though it finished last.
	is.True(slices.ContainsFunc(f.Candidates, func(cs montecarlo.CandidateStats) bool {
		return cs.Play == worst.Play
	}))

	// It got the same analysis the best play gets.
	is.True(c.RivalPlayStats != nil)
	is.Equal(len(c.RivalFollowups), len(c.RivalPlayStats.OurFollowups))

	// The worst play by win% cannot be ahead of the best one.
	is.True(c.Deltas.WinPct >= 0)

	// And it is in the board-dynamics set, however low it ranked.
	lanePlays := []string{}
	for _, lc := range f.Lanes {
		lanePlays = append(lanePlays, lc.Play)
	}
	is.True(slices.Contains(lanePlays, worst.Play))
	is.Equal(f.Lanes[0].Play, f.Best.Play) // the best play still leads

	p, err := BuildPrompt(f, false)
	is.NoErr(err)
	is.True(strings.Contains(p.User, "### Head to head"))
	is.True(strings.Contains(p.User, "### What the simulation saw after "+worst.Play))
}

// Naming the top play gets a comparison against the runner-up instead, so
// there is still something to explain.
func TestComparingAgainstTheBestPlayEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("runs a simulation")
	}
	is := is.New(t)
	an, simmer, simStats := simulate(t, examplePosition, 8, 3, 120)

	all := simmer.CandidateStats()
	f, err := an.BuildFacts(simmer, simStats, &ComparisonRequest{Move: all[0].Move}, nil)
	is.NoErr(err)

	is.True(f.Comparison != nil)
	is.True(f.Comparison.WasBest)
	is.Equal(f.Comparison.Play, f.Best.Play)
	is.Equal(f.Comparison.Rival.Play, f.Candidates[1].Play)
	is.True(f.Flags["comparison_was_best"])
}

// A play the simulation never evaluated can't be compared against, and that
// has to be a skipped comparison rather than a failed explanation.
func TestUnsimmedComparisonIsSkipped(t *testing.T) {
	if testing.Short() {
		t.Skip("runs a simulation")
	}
	is := is.New(t)
	an, simmer, simStats := simulate(t, examplePosition, 8, 3, 120)

	// A legal play on this board that the sim was never given.
	other, err := an.game.ParseMove(an.game.PlayerOnTurn(), false,
		strings.Fields(DottedPlay("12K QU(ID)")), false)
	is.NoErr(err)

	f, err := an.BuildFacts(simmer, simStats, &ComparisonRequest{Move: other}, nil)
	is.NoErr(err)
	if f.Comparison != nil {
		// It was in the sim after all, which is fine - just not what this
		// test is about.
		t.Skip("the play happened to be simulated")
	}
	is.True(!f.Flags["has_comparison"])
}
