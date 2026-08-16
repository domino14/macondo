package explainer

import (
	"slices"
	"strings"
	"testing"

	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/matryer/is"
)

// A card whose trigger doesn't name a real flag would parse, embed, and never
// ship - silently. This is the check that keeps that from happening.
func TestEveryCardHasARealTrigger(t *testing.T) {
	is := is.New(t)
	cards, err := loadConcepts()
	is.NoErr(err)
	is.True(len(cards) > 0)

	ids := map[string]bool{}
	for _, c := range cards {
		is.True(!ids[c.ID]) // ids are unique
		ids[c.ID] = true
		for _, flag := range c.When {
			if !slices.Contains(knownFlags, flag) {
				t.Errorf("card %s triggers on %q, which computeFlags never sets", c.ID, flag)
			}
		}
	}
}

// Conversely, a flag nothing triggers on is a computation we're paying for and
// throwing away.
func TestEveryFlagTriggersSomething(t *testing.T) {
	is := is.New(t)
	cards, err := loadConcepts()
	is.NoErr(err)

	used := map[string]bool{}
	for _, c := range cards {
		for _, flag := range c.When {
			used[flag] = true
		}
	}
	for _, flag := range knownFlags {
		if !used[flag] {
			t.Errorf("flag %q is computed but no card triggers on it", flag)
		}
	}
}

func TestSelectConcepts(t *testing.T) {
	is := is.New(t)

	// A position with nothing special about it gets only the always-on cards.
	plain, err := SelectConcepts(Flags{"always": true})
	is.NoErr(err)
	plainIDs := ConceptIDs(plain)
	is.True(slices.Contains(plainIDs, "notation"))
	is.True(slices.Contains(plainIDs, "winpct"))
	is.True(!slices.Contains(plainIDs, "preendgame"))
	is.True(!slices.Contains(plainIDs, "grouped-blanks"))
	is.True(!slices.Contains(plainIDs, "turnover"))

	// Turning a flag on brings in exactly its card.
	late, err := SelectConcepts(Flags{"always": true, "pre_endgame": true})
	is.NoErr(err)
	is.True(slices.Contains(ConceptIDs(late), "preendgame"))
	is.Equal(len(late), len(plain)+1)

	// Cards come out in priority order, so notation always leads.
	is.Equal(plainIDs[0], "notation")
}

func TestParseConceptRejectsBadCards(t *testing.T) {
	is := is.New(t)

	_, err := parseConcept("no frontmatter here")
	is.True(err != nil)

	_, err = parseConcept("---\nid: x\n---\n")
	is.True(err != nil) // no trigger, no body

	_, err = parseConcept("---\nid: x\nwhen: always\nnonsense: 1\n---\nbody")
	is.True(err != nil) // unknown key: probably a typo for a real one

	c, err := parseConcept("---\nid: x\npriority: 5\nwhen: a, b\n---\nthe body\n")
	is.NoErr(err)
	is.Equal(c.ID, "x")
	is.Equal(c.Priority, 5)
	is.Equal(c.When, []string{"a", "b"})
	is.Equal(c.Body, "the body")
}

// fakeFacts is the smallest fact pack that renders, for prompt-shape tests.
func fakeFacts() *PositionFacts {
	best := montecarlo.CandidateStats{
		Play: "12K QU(ID)", Leave: "ACDEP", Score: 28, WinPct: 37.8, WinPctCI: 1.2,
		Equity: 12.4, EquityCI: 0.8,
		Plies: []montecarlo.CandidatePlyStats{
			{Ply: 1, MeanScore: 31.2, Stdev: 12.3, BingoPct: 8},
			{Ply: 2, Ours: true, MeanScore: 44.5, Stdev: 18.1, BingoPct: 24},
		},
	}
	second := montecarlo.CandidateStats{
		Play: "5D (S)CAP(A)", Leave: "DEQU", Score: 30, WinPct: 30.1, WinPctCI: 1.3,
		Equity: 13.9, EquityCI: 0.9,
		Plies: []montecarlo.CandidatePlyStats{
			{Ply: 1, MeanScore: 35.0, Stdev: 13.0, BingoPct: 9},
			{Ply: 2, Ours: true, MeanScore: 34.0, Stdev: 16.0, BingoPct: 21},
		},
	}
	f := &PositionFacts{
		Rack: "ACDEPQU", Lexicon: "CSW24", BagCount: 9, OppRackSize: 7, UnseenCount: 16,
		Spread: -40, Phase: PhaseLateMidgame,
		UnseenVowels: 8, UnseenConsonants: 8, UnseenPowerTiles: []string{"J", "S"},
		Iterations: 1200,
		Candidates: []montecarlo.CandidateStats{best, second},
		PlayStats:  &stats.PlayStats{OurBingoPct: 24, OppBingoPct: 12},
	}
	f.Best = &f.Candidates[0]
	f.BestByEquity = &f.Candidates[1]
	f.Followups = []*FollowupFact{{
		FollowupFamily: &stats.FollowupFamily{
			Play: "15G PREAD(JUST)", Pct: 11.5, MinScore: 57, MaxScore: 57,
			NeededDraws: []string{"R"},
			Ways:        []*stats.FollowupWay{{Play: "15G PREAD(JUST)", Score: 57, Pct: 11.5, NeededDraw: "R"}},
		},
		WayRequirements: []string{"requires us to play 12K QU(ID) first"},
		IsSetup:         true,
	}}
	f.Flags = computeFlags(f)
	return f
}

func TestBuildPrompt(t *testing.T) {
	is := is.New(t)
	f := fakeFacts()

	// The position drives which knowledge gets sent.
	is.True(f.Flags["has_setup"])
	is.True(f.Flags["pre_endgame"])
	is.True(f.Flags["equity_sacrifice"])
	is.True(!f.Flags["behind_early"]) // behind, but far too late in the game
	is.True(!f.Flags["has_grouped_followup"])

	p, err := BuildPrompt(f, false)
	is.NoErr(err)

	is.True(slices.Contains(p.Concepts, "setup"))
	is.True(slices.Contains(p.Concepts, "preendgame"))
	is.True(slices.Contains(p.Concepts, "equity"))
	is.True(!slices.Contains(p.Concepts, "grouped-blanks"))
	is.True(!slices.Contains(p.Concepts, "board-dynamics"))

	// Stable instructions in the system message, the position in the user
	// message - not one blob.
	is.True(strings.Contains(p.System, "You are a Scrabble coach"))
	is.True(!strings.Contains(p.System, "12K QU(ID)"))
	is.True(!strings.Contains(p.System, "{quirky}"))
	is.True(strings.Contains(p.User, "12K QU(ID)"))
	is.True(strings.Contains(p.User, "15G PREAD(JUST)"))
	is.True(strings.Contains(p.User, "SETUP"))
	is.True(strings.Contains(p.User, "We are behind by 40 points"))
	is.True(strings.Contains(p.User, "highest equity"))

	quirky, err := BuildPrompt(f, true)
	is.NoErr(err)
	is.True(strings.Contains(quirky.System, "wenue"))
}

// Both messages are markdown, so their own headings start with "#". Any dump
// convention that marked Macondo's commentary with a per-line prefix would
// contradict the content it wraps: "## What to say" lives in system.md and
// really is sent. Sections are delimited by banners for exactly this reason.
func TestMarkdownHeadingsInThePromptAreSent(t *testing.T) {
	is := is.New(t)
	p, err := BuildPrompt(fakeFacts(), false)
	is.NoErr(err)

	is.True(strings.Contains(p.System, "\n## What to say\n"))
	is.True(strings.Contains(p.User, "\n### Position\n"))

	// Both survive into the dump, inside the section each belongs to.
	dump := p.Notes("") + p.String()
	_, afterSystem, found := strings.Cut(dump, systemBanner)
	is.True(found)
	system, user, found := strings.Cut(afterSystem, userBanner)
	is.True(found)
	is.True(strings.Contains(system, "## What to say"))
	is.True(strings.Contains(user, "### Position"))
}

// The whole point of the split is that an ordinary position stops paying for
// knowledge it doesn't need.
func TestUnneededConceptsAreNotSent(t *testing.T) {
	is := is.New(t)

	f := fakeFacts()
	full, err := BuildPrompt(f, false)
	is.NoErr(err)

	bare := fakeFacts()
	bare.Followups = nil
	bare.BagCount = 60
	bare.BestByEquity = bare.Best
	bare.Flags = computeFlags(bare)
	small, err := BuildPrompt(bare, false)
	is.NoErr(err)

	is.True(len(small.Concepts) < len(full.Concepts))
	is.True(len(small.User) < len(full.User))
}
