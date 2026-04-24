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

// TestDebugPegGEI traces one specific bag outcome (bag=GEI, draw order G first)
// through the full PEG recursion for play 13M P(AH). Remove the t.Skip to run.
// Output goes to preendgame/peg-debug-GEI.txt.
func TestDebugPegGEI(t *testing.T) {
	t.Skip("manual debug harness; comment out to run")

	const cgpStr = "BEDEL10/R1R9U2/O1IT1Q5OM2/W1BIDI4YUM2/N2XI5AT3/E3G4T1R3/S1VOILE2OKA3/T3T1DISPACED1/9AWE1O1/9Z1s1FA/14R/13GO/13AH/3JUVIE4UTA/INRO3FLENCHES ?ANNOPY/AEELNRS 344/368 0 lex CSW21;"

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

	// Trace only the perm where bag draw order is G, E, I (G drawn first by us).
	bagTail, err := tilemapping.ToMachineLetters("GEI", g.Alphabet())
	is.NoErr(err)
	peg.SetTraceTargetBagTail(bagTail)
	peg.SetTraceOnce(true)

	f, err := os.Create("peg-debug-GEI.txt")
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

// TestDebugPegGRL traces one specific bag outcome (bag={G,R,L}, draw order G
// first) through the full PEG recursion for play 13M P(AH). This perm is
// incorrectly marked LOSS (Ortograf says WIN). Remove the t.Skip to run.
// Output goes to preendgame/peg-debug-GRL.txt.
func TestDebugPegGRL(t *testing.T) {
	// t.Skip("manual debug harness; comment out to run")

	const cgpStr = "BEDEL10/R1R9U2/O1IT1Q5OM2/W1BIDI4YUM2/N2XI5AT3/E3G4T1R3/S1VOILE2OKA3/T3T1DISPACED1/9AWE1O1/9Z1s1FA/14R/13GO/13AH/3JUVIE4UTA/INRO3FLENCHES ?ANNOPY/AEELNRS 344/368 0 lex CSW21;"

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

	// Trace the perm where draw order is G, R, L (G drawn first by us).
	// bagTail="GRL" ↔ bagPeek=[L,R,G] ↔ MoveTilesToBeginning([L,R,G]) ↔ draw back→front = G,R,L.
	bagTail, err := tilemapping.ToMachineLetters("GRL", g.Alphabet())
	is.NoErr(err)
	peg.SetTraceTargetBagTail(bagTail)
	peg.SetTraceOnce(true)

	f, err := os.Create("peg-debug-GRL.txt")
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
