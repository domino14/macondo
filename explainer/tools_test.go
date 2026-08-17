package explainer

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/matryer/is"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// way builds one route to a follow-up play, as montecarlo/stats would export
// it from the simulation log.
func way(play, draw string, score int, pct float64, bingo bool) *stats.FollowupWay {
	return &stats.FollowupWay{Play: play, NeededDraw: draw, Score: score, Pct: pct, Bingo: bingo}
}

// fam builds a follow-up family. With no ways given it's a play with a single
// route, the way the exporter builds one.
func fam(play string, pct float64, minScore, maxScore int, ways ...*stats.FollowupWay) *stats.FollowupFamily {
	f := &stats.FollowupFamily{
		Play: play, Pct: pct, MinScore: minScore, MaxScore: maxScore, TilePlay: true,
	}
	if len(ways) == 0 {
		ways = []*stats.FollowupWay{way(play, "", maxScore, pct, false)}
	}
	f.Ways = ways
	f.Bingo = ways[0].Bingo
	seen := map[string]bool{}
	for _, w := range ways {
		if !seen[w.NeededDraw] {
			seen[w.NeededDraw] = true
			f.NeededDraws = append(f.NeededDraws, w.NeededDraw)
		}
	}
	return f
}

// blankyFollowups is a real follow-up table, from a position with ?AEIKUW on
// our rack. Note the lowercase letters: those plays use the blank. ZWIEBACK
// and SAWLIKE can each be made in more than one way, so they are grouped.
func blankyFollowups() []*FollowupFact {
	fams := []*stats.FollowupFamily{
		fam("7F W(E)AK", 11.11, 23, 23),
		fam("1H (Z)WIEBACK", 6.72, 116, 134,
			way("1H (Z)WIEBAcK", "B", 134, 2.85, false),
			way("1H (Z)WIEbACK", "C", 125, 2.77, false),
			way("1H (Z)WIEbAcK", "?", 116, 1.10, false)),
		fam("I8 SAWLIKE", 5.94, 80, 81,
			way("I8 SAWlIKE", "S", 81, 3.34, true),
			way("I8 sAWLIKE", "L", 80, 2.60, true)),
		fam("F12 WAKE", 3.46, 30, 30),
		fam("I8 sKIWEAR", 3.17, 86, 86),
		fam("2D KIT(TI)WAkE", 2.24, 69, 69),
		fam("12A (D)ISCOMBOB(U)LATE", 1.02, 42, 42),
	}
	out := make([]*FollowupFact, 0, len(fams))
	for _, f := range fams {
		fact := &FollowupFact{FollowupFamily: f}
		for range f.Ways {
			fact.WayRequirements = append(fact.WayRequirements, "none")
		}
		out = append(out, fact)
	}
	return out
}

// factsWith wraps follow-ups in the minimum fact pack the tools need.
func factsWith(bestPlay string, followups []*FollowupFact) *PositionFacts {
	return &PositionFacts{
		Best:      &montecarlo.CandidateStats{Play: bestPlay},
		Followups: followups,
		Flags:     Flags{},
	}
}

func TestGetPlayMetadata(t *testing.T) {
	is := is.New(t)
	cfg := config.DefaultConfig()

	// Test with opening position
	bpos := "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 ADOPQRR/ 0/0 0 lex NWL23;"
	g, err := cgp.ParseCGP(cfg, bpos)
	is.NoErr(err)
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)

	leavesFile := ""
	conf := &bot.BotConfig{Config: *cfg, LeavesFile: leavesFile}

	tp, err := bot.NewBotTurnPlayerFromGame(g.Game, conf, pb.BotRequest_HASTY_BOT)
	is.NoErr(err)
	an := NewAnalyzer()
	an.SetGame(tp)
	an.SetConfig(cfg)

	// Test exchange move with parentheses format (as reported in issue #425)
	md, err := an.GetPlayMetadata("(exch Q)")
	is.NoErr(err)
	is.Equal(md.Play, "(exch Q)")
	is.Equal(md.Score, 0)
	is.Equal(md.TilesUsed, 1) // Exchanging 1 tile
	is.Equal(md.IsBingo, false)

	// Test exchange move without parentheses
	md, err = an.GetPlayMetadata("exch Q")
	is.NoErr(err)
	is.Equal(md.Play, "exch Q")
	is.Equal(md.Score, 0)
	is.Equal(md.TilesUsed, 1)

	// Test pass move
	md, err = an.GetPlayMetadata("pass")
	is.NoErr(err)
	is.Equal(md.Play, "pass")
	is.Equal(md.Score, 0)
	is.Equal(md.TilesUsed, 0)
}

// newFollowupAnalyzer sets up an analyzer on a position whose best play is
// G12 OO, leaving AFLRY.
func newFollowupAnalyzer(t *testing.T) *Analyzer {
	is := is.New(t)
	cfg := config.DefaultConfig()
	// This is tested indirectly via TestExplainGame
	bpos := "PEC4D3QUAY/1EUOI2UG4V1/2MINX1ER3TI1/5UNTANGLeD1/8I3N2/6RONZ2T2/5HOPE1T1H2/3COMBES1R4/2BOW5A4/3V1ASEItIES2/3EEW4T4/3DAD9/5L9/15/15 AFLOORY/AAFGNRT 337/281 0 lex CSW24;"
	g, err := cgp.ParseCGP(config.DefaultConfig(), bpos)
	is.NoErr(err)
	g.RecalculateBoard() // to calculate cross-scores etc.
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)

	leavesFile := ""
	conf := &bot.BotConfig{Config: *cfg, LeavesFile: leavesFile}

	tp, err := bot.NewBotTurnPlayerFromGame(g.Game, conf, pb.BotRequest_HASTY_BOT)
	is.NoErr(err)
	an := NewAnalyzer()
	an.SetGame(tp)
	an.SetConfig(cfg)
	return an
}

// Whether a follow-up is a setup is decided by putting the best play on the
// board and seeing whether the follow-up becomes possible - not by reading a
// percentage off a table.
func TestFollowupRequirements(t *testing.T) {
	is := is.New(t)
	an := newFollowupAnalyzer(t)

	ps := &stats.PlayStats{OurFollowups: []*stats.FollowupFamily{
		fam("L9 L(E)AFERY", 13.24, 38, 38, way("L9 L(E)AFERY", "E", 38, 13.24, false)),
		fam("L9 L(E)AFY", 10.94, 26, 26),
		fam("I12 LYRA", 3.54, 33, 33),
		fam("14G FLAYERS", 0.78, 77, 77, way("14G FLAYERS", "ES", 77, 0.78, true)),
	}}
	facts, err := an.analyzeFollowups("G12 OO", 40, ps)
	is.NoErr(err)
	an.facts = factsWith("G12 OO", facts)

	f, err := an.GetFuturePlayMetadata("L9 L(E)AFERY")
	is.NoErr(err)
	is.Equal(f, &FuturePlayMetadata{
		Play:               "L9 L(E)AFERY",
		NeededDraw:         []string{"E"},
		Score:              38,
		ProbabilityPercent: 13.24,
		RequiresOtherPlay:  "none",
	})

	f, err = an.GetFuturePlayMetadata("L9 L(E)AFY")
	is.NoErr(err)
	is.Equal(f, &FuturePlayMetadata{
		Play:               "L9 L(E)AFY",
		NeededDraw:         []string{},
		Score:              26,
		ProbabilityPercent: 10.94,
		RequiresOtherPlay:  "none",
	})

	// FLAYERS only exists once OO is on the board: that is what makes a play
	// a setup rather than a coincidence.
	f, err = an.GetFuturePlayMetadata("14G FLAYERS")
	is.NoErr(err)
	is.Equal(f, &FuturePlayMetadata{
		Play:               "14G FLAYERS",
		NeededDraw:         []string{"E", "S"},
		Score:              77,
		IsBingo:            true,
		ProbabilityPercent: 0.78,
		RequiresOtherPlay:  "requires us to play G12 OO first",
	})

	// LYRA doesn't score what the sim saw on either board, so the opponent
	// must have added something first.
	f, err = an.GetFuturePlayMetadata("I12 LYRA")
	is.NoErr(err)
	is.Equal(f.RequiresOtherPlay, "requires opponent play")
}

// A follow-up is only a setup if our play enables it *and* it is likely enough
// and big enough to be worth setting up for.
func TestSetupThresholds(t *testing.T) {
	is := is.New(t)
	an := newFollowupAnalyzer(t)

	ps := &stats.PlayStats{OurFollowups: []*stats.FollowupFamily{
		// enabled by our play, but a 0.78% chance of 77 points
		fam("14G FLAYERS", 0.78, 77, 77, way("14G FLAYERS", "ES", 77, 0.78, true)),
		// enabled by our play, likely and big
		fam("14G FRIARLY", 8.0, 71, 71, way("14G FRIARLY", "IR", 71, 8.0, false)),
		// available anyway, and no bigger than an ordinary turn
		fam("L9 L(E)AFERY", 13.24, 38, 38, way("L9 L(E)AFERY", "E", 38, 13.24, false)),
	}}
	facts, err := an.analyzeFollowups("G12 OO", 40, ps)
	is.NoErr(err)

	is.Equal(facts[0].Requirement(), "requires us to play G12 OO first")
	is.True(!facts[0].IsSetup) // too unlikely to be the reason for the play
	is.True(facts[1].IsSetup)
	is.True(!facts[2].IsSetup)
	is.True(!facts[2].IsBigChance) // 38 against a 40-point turn is not a chance

	f := factsWith("G12 OO", facts)
	f.Flags = computeFlags(&PositionFacts{
		Best: &montecarlo.CandidateStats{Play: "G12 OO", Leave: "AFLRY"}, Followups: facts,
	})
	is.True(f.Flags["has_setup"])
	is.True(f.Flags["has_needed_draw"])
	is.True(!f.Flags["has_grouped_followup"])
}

// The regression test for the rule this replaced. A flat "at least 10% of the
// time and at least 40 points" floor threw away exactly the plays worth
// building a turn around: it passed an 11% chance at 23 points and rejected a
// 6% chance at 130. These are the real figures from a position with ?AEIKUW.
func TestBigChanceWeighsSizeAgainstFrequency(t *testing.T) {
	is := is.New(t)

	fact := func(f *stats.FollowupFamily) *FollowupFact { return &FollowupFact{FollowupFamily: f} }
	// An ordinary next turn after E4 (X)U is worth about 44 points.
	const typical = 44.4

	weak := fact(fam("7F W(E)AK", 11.24, 23, 23))
	zwieback := fact(fam("1H (Z)WIEBACK", 6.01, 116, 134,
		way("1H (Z)WIEBAcK", "B", 134, 2.38, false),
		way("1H (Z)WIEbACK", "C", 125, 2.32, false),
		way("1H (Z)WIEbAcK", "?", 116, 1.31, false)))
	sawlike := fact(fam("I8 SAWLIKE", 5.71, 79, 81,
		way("I8 SAWlIKE", "S", 81, 3.03, true),
		way("I8 sAWLIKE", "L", 80, 2.56, true),
		way("I8 sAWlIKE", "?", 79, 0.12, true)))
	skiwear := fact(fam("I8 sKIWEAR", 4.58, 80, 80))

	// 15G PREAD(JUST) from the position in the manual: modest next to
	// ZWIEBACK, but still worth naming, and the rule has to keep it.
	preadjust := fact(fam("15G PREAD(JUST)", 11.07, 57, 57))

	for _, tc := range []struct {
		f   *FollowupFact
		big bool
	}{
		{weak, false}, // frequent, but half an ordinary turn
		{zwieback, true},
		{sawlike, true},
		{skiwear, true},
		{preadjust, true},
	} {
		upside, big := bigChance(tc.f, typical)
		if big != tc.big {
			t.Errorf("%s: big=%v, want %v (upside %.2f)", tc.f.Play, big, tc.big, upside)
		}
	}

	// A grouped play is judged on what its routes average, not on its
	// luckiest one: quoting 134 would overstate a play that lands on 116 a
	// fifth of the time.
	is.True(zwieback.AvgScore() < 134)
	is.True(zwieback.AvgScore() > 125)

	// And the ordering puts the biggest first, which is the order the
	// explanation should follow.
	zUpside, _ := bigChance(zwieback, typical)
	sUpside, _ := bigChance(sawlike, typical)
	is.True(zUpside > sUpside)

	// Nothing after the play the reader made clears the bar - which is the
	// contrast the explanation exists to draw.
	for _, f := range []*FollowupFact{
		fact(fam("3C OWE", 2.50, 26, 26)),
		fact(fam("5D WIN(D)", 1.61, 44, 44)),
		fact(fam("1H (Z)OWIE", 1.43, 18, 18)),
	} {
		if _, big := bigChance(f, 39.6); big {
			t.Errorf("%s should not count as a big chance", f.Play)
		}
	}
}

// Without a baseline there is nothing to call big, and claiming otherwise
// would make every follow-up look enormous.
func TestBigChanceNeedsABaseline(t *testing.T) {
	is := is.New(t)
	f := &FollowupFact{FollowupFamily: fam("1H (Z)WIEBACK", 6.01, 134, 134)}
	upside, big := bigChance(f, 0)
	is.Equal(upside, 0.0)
	is.True(!big)
}

// However the model names a grouped play, it gets the whole family back, so
// it can't mistake one route's chance for the chance of making the play.
func TestGroupedPlayAlwaysReturnsEveryWay(t *testing.T) {
	is := is.New(t)
	an := NewAnalyzer()
	leafery := &FollowupFact{
		FollowupFamily: fam("L9 L(E)AFERY", 14.26, 34, 38,
			way("L9 L(E)AFERY", "E", 38, 13.24, false),
			way("L9 L(E)AFERy", "E?", 34, 1.02, false)),
		WayRequirements: []string{"none", "none"},
	}
	pecky := &FollowupFact{
		FollowupFamily:  fam("1A (PEC)KY", 5.19, 39, 39, way("1A (PEC)KY", "K", 39, 5.19, false)),
		WayRequirements: []string{"none"},
	}
	an.facts = factsWith("G12 OO", []*FollowupFact{leafery, pecky})
	tool := NewGetOurFuturePlayMetadataTool(an)

	for _, q := range []string{"L9 L(E)AFERY", "L9 L(E)AFERy", "L9 LEAFERY"} {
		res, err := tool.Execute(context.Background(), `{"play_string": "`+q+`"}`)
		is.NoErr(err)

		var got struct {
			Note       string `json:"note"`
			AskedAbout string `json:"asked_about"`
			Family     struct {
				Play              string   `json:"play"`
				CombinedPercent   float64  `json:"combined_probability_percent"`
				ScoreRange        string   `json:"score_range"`
				NeededDrawOptions []string `json:"needed_draw_options"`
			} `json:"family"`
			Ways []struct {
				Play               string  `json:"play"`
				Score              int     `json:"score"`
				ProbabilityPercent float64 `json:"probability_percent"`
			} `json:"ways"`
		}
		is.NoErr(json.Unmarshal([]byte(res), &got))

		is.Equal(got.Family.Play, "L9 L(E)AFERY")
		is.Equal(got.Family.CombinedPercent, 14.26) // not 13.24, the top way's share
		is.Equal(got.Family.ScoreRange, "34-38")
		is.Equal(got.Family.NeededDrawOptions, []string{"E", "E?"})
		is.Equal(len(got.Ways), 2)
		is.Equal(got.Ways[1].Play, "L9 L(E)AFERy")
		is.Equal(got.Ways[1].Score, 34)
		is.True(strings.Contains(got.Note, "combined_probability_percent"))
	}

	// naming one route reports which one was asked about
	res, err := tool.Execute(context.Background(), `{"play_string": "L9 L(E)AFERy"}`)
	is.NoErr(err)
	is.True(strings.Contains(res, `"asked_about":"L9 L(E)AFERy"`))

	// a play with only one way keeps the flat shape
	res, err = tool.Execute(context.Background(), `{"play_string": "1A (PEC)KY"}`)
	is.NoErr(err)
	is.True(!strings.Contains(res, `"ways"`))
	is.True(strings.Contains(res, `"probability_percent":5.19`))
}

func TestMatchFollowup(t *testing.T) {
	is := is.New(t)
	fams := blankyFollowups()

	match := func(q string) (string, string) {
		fam, wayIdx := matchFollowup(fams, q)
		if fam == nil {
			return "", ""
		}
		if wayIdx < 0 {
			return fam.Play, ""
		}
		return fam.Play, fam.Ways[wayIdx].Play
	}

	// exact match on a play with only one way to make it: the family answers,
	// and no individual way was named
	fam, wayPlay := match("I8 sKIWEAR")
	is.Equal(fam, "I8 sKIWEAR")
	is.Equal(wayPlay, "")

	// leading whitespace, as it appears in the table
	fam, _ = match(" I8 sKIWEAR")
	is.Equal(fam, "I8 sKIWEAR")

	// the model uppercased the blank
	fam, _ = match("I8 SKIWEAR")
	is.Equal(fam, "I8 sKIWEAR")

	// the grouped name resolves to the family, with no single way named
	fam, wayPlay = match("1H (Z)WIEBACK")
	is.Equal(fam, "1H (Z)WIEBACK")
	is.Equal(wayPlay, "")

	// naming one way still identifies the family it belongs to
	fam, wayPlay = match("1H (Z)WIEbACK")
	is.Equal(fam, "1H (Z)WIEBACK")
	is.Equal(wayPlay, "1H (Z)WIEbACK")

	// the model also guessed at the playthrough parens
	fam, _ = match("I8 S(K)IWEAR")
	is.Equal(fam, "I8 sKIWEAR")

	fam, _ = match("12A (D)ISCOMBOB(U)LATE")
	is.Equal(fam, "12A (D)ISCOMBOB(U)LATE")

	fam, _ = match("8H QUIXOTIC")
	is.Equal(fam, "")
}

func TestFuturePlayNotFoundIsNotAnError(t *testing.T) {
	is := is.New(t)
	an := NewAnalyzer()
	an.facts = factsWith("G12 OO", blankyFollowups())

	_, err := an.LookupFuturePlay("8H QUIXOTIC")
	var notFound *PlayNotFoundError
	is.True(errors.As(err, &notFound))
	is.Equal(len(notFound.Available), 12) // 7 families plus 5 individual ways

	// The tool hands the model a normal result listing the plays it can ask
	// about, rather than an error it will just retry.
	tool := NewGetOurFuturePlayMetadataTool(an)
	res, err := tool.Execute(context.Background(), `{"play_string": "8H QUIXOTIC"}`)
	is.NoErr(err)
	is.True(strings.Contains(res, "is not one of the follow-up plays"))
	is.True(strings.Contains(res, "- I8 sKIWEAR"))
	is.True(strings.Contains(res, "- 1H (Z)WIEbAcK"))
}
