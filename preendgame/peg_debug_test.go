package preendgame

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/move"
)

// TestDebugPegHiedGEI traces one specific bag outcome (opp=LEANERS, bag=GEI)
// through the full PEG recursion for play 13M P(AH). Remove the t.Skip to run.
// Output goes to preendgame/peg-debug-hied-GEI.txt.
func TestDebugPegHiedGEI(t *testing.T) {
	t.Skip("manual debug harness; comment out to run")

	const cgpStr = "BEDEL10/R1R9U2/9FLENCHES/15/15/15/15/15/15/15/15/INRO11/15/15/15 ?ANNOPY/AEEGLRS 344/368 0 lex CSW21;"

	is := is.New(t)
	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	is.NoErr(err)

	peg := &Solver{}
	is.NoErr(peg.Init(g.Game, gd))

	// 13M P(AH) scores 32; leave after playing P from ?ANNOPY is ?ANNOY.
	m := move.NewScoringMoveSimple(32, "13M", "P..", "?ANNOY", g.Alphabet())
	peg.SetSolveOnly([]*move.Move{m})
	peg.SetThreads(1)
	peg.SetEndgamePlies(4)
	peg.SetIterativeDeepening(false)
	peg.SetNestedDepthLimit(3)
	peg.SetSkipDeepPass(false)
	peg.SetSkipTiebreaker(true)

	// Trace only the perm where opp draws LEANERS and bag tail is GEI (in order).
	peg.SetTraceTargetFirstRack("AEELNRS") // LEANERS sorted
	peg.SetTraceTargetBagTail("GEI")
	peg.SetTraceOnce(true)

	f, err := os.Create("peg-debug-hied-GEI.txt")
	is.NoErr(err)
	defer f.Close()
	peg.SetTraceWriter(f)

	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Hour)
	defer cancel()

	winners, err := peg.Solve(ctx)
	is.True(err == nil || err == ErrCanceledEarly)

	for _, o := range winners[0].OutcomesArray() {
		tiles := tilemapping.MachineWord(o.Tiles()).UserVisible(g.Alphabet())
		t.Logf("bag=%s outcome=%s count=%d", tiles, o.OutcomeResult(), o.Count())
	}
}
