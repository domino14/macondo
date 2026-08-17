package explainer

import (
	"strings"
	"testing"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/matryer/is"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// The position from docs/manual/ai-explainability.md: 40 points down with
// ACDEPQU, nine in the bag.
const examplePosition = "COXA2B2ROPING/3T1DINGY5/3t2ZO1AINEE1/3A3V2FOWTH/3S3A2FOE1U/" +
	"1LEKE2T1MY3T/2RELATIVE4I/ARED3N5BA/7g5R1/12WO1/12HA1/12ID1/12ME1/13N1/" +
	"11JUST ACDEPQU/ 400/440 0 lex CSW24;"

// simulate runs a small real simulation, which is the only way to exercise the
// parts that read the heat map log: the follow-up tables and the lane stats.
func simulate(t *testing.T, cgpStr string, plays, plies, iters int) (*Analyzer, *montecarlo.Simmer, *stats.SimStats) {
	t.Helper()
	is := is.New(t)

	cfg := config.DefaultConfig()
	g, err := cgp.ParseCGP(cfg, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)

	tp, err := bot.NewBotTurnPlayerFromGame(g.Game, &bot.BotConfig{Config: *cfg}, pb.BotRequest_HASTY_BOT)
	is.NoErr(err)

	calc, err := equity.NewCombinedStaticCalculator(tp.LexiconName(), cfg, "", equity.PEGAdjustmentFilename)
	is.NoErr(err)
	simmer := &montecarlo.Simmer{}
	simmer.Init(tp.Game, []equity.EquityCalculator{calc}, calc, cfg)
	is.NoErr(simmer.SetCollectHeatmap(true))
	t.Cleanup(simmer.CleanupTempFile)
	is.NoErr(simmer.PrepareSim(plies, tp.GenerateMoves(plays)))
	simmer.SimSingleThread(iters, plies)

	an := NewAnalyzer()
	an.SetConfig(cfg)
	an.SetGame(tp)
	return an, simmer, stats.NewSimStats(simmer, tp)
}

func TestBuildFactsEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("runs a simulation")
	}
	is := is.New(t)
	an, simmer, simStats := simulate(t, examplePosition, 8, 3, 120)

	f, err := an.BuildFacts(simmer, simStats, nil, nil)
	is.NoErr(err)

	// The position, read off the game rather than described in prose.
	is.Equal(f.Rack, "ACDEPQU")
	is.Equal(f.Lexicon, "CSW24")
	is.Equal(f.Spread, -40)
	// The CGP gives the opponent no rack, so all 16 unseen tiles count as bag.
	is.Equal(f.BagCount, 16)
	is.Equal(f.OppRackSize, 0)
	is.Equal(f.UnseenCount, 16)
	is.Equal(f.UnseenVowels+f.UnseenConsonants+f.UnseenBlanks, 16)
	is.Equal(f.Phase, PhaseMidgame)
	is.True(!f.Flags["pre_endgame"]) // 16 in the bag is never a pre-endgame
	is.True(f.Flags["behind_early"])
	is.True(f.Flags["turnover_relevant"]) // second half, power tiles unseen

	// Candidates come out sorted by win%, and the play strings carry no table
	// padding.
	is.True(len(f.Candidates) > 1)
	for _, c := range f.Candidates {
		is.Equal(c.Play, strings.TrimSpace(c.Play))
	}
	is.True(f.Candidates[0].WinPct >= f.Candidates[1].WinPct)
	is.Equal(f.Best.Play, f.Candidates[0].Play)

	// Follow-ups came from the sim log, and each one has a verdict attached.
	is.True(f.PlayStats != nil)
	for _, fu := range f.Followups {
		is.Equal(len(fu.WayRequirements), len(fu.Ways))
		for _, req := range fu.WayRequirements {
			is.True(req == "none" || req == "requires opponent play" ||
				strings.HasPrefix(req, "requires us to play "))
		}
	}

	// Lane stats were computed for the top candidates, and every reply is
	// accounted for one way or another.
	is.True(len(f.Lanes) > 1)
	for _, lc := range f.Lanes {
		attributed := lc.Stats.Placements + lc.Stats.SingleTile + lc.Stats.Scoreless
		is.True(attributed <= lc.Stats.Total)
		for _, l := range lc.Stats.Lanes {
			is.True(strings.HasPrefix(l.Label, "row ") || strings.HasPrefix(l.Label, "column "))
			is.True(l.MaxScore >= int(l.MeanScore))
			is.True(l.BestPlay != "")
		}
	}
}

func TestRenderedPromptHasEverySection(t *testing.T) {
	if testing.Short() {
		t.Skip("runs a simulation")
	}
	is := is.New(t)
	an, simmer, simStats := simulate(t, examplePosition, 8, 3, 120)

	f, err := an.BuildFacts(simmer, simStats, nil, nil)
	is.NoErr(err)
	p, err := BuildPrompt(f, false)
	is.NoErr(err)

	for _, want := range []string{
		"### Position",
		"### Candidate plays",
		"### Next two plies per candidate",
		"### Our follow-up play",
		"## Your explanation for why ",
	} {
		if !strings.Contains(p.User, want) {
			t.Errorf("prompt is missing %q", want)
		}
	}

	// The deep plies are simulated but not sent: nobody can reason about a
	// mean score five plies out, and the win% already accounts for them.
	is.True(!strings.Contains(p.User, "Ply 3"))

	// A follow-up play the model might ask about resolves through the tools,
	// with no parsing of the prompt involved.
	if len(f.Followups) > 0 {
		lookup, err := an.LookupFuturePlay(f.Followups[0].Play)
		is.NoErr(err)
		is.True(len(lookup.Ways) > 0)
	}
}
