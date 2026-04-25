package preendgame

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/move"
)

const explainCGP = "BEDEL10/R1R9U2/O1IT1Q5OM2/W1BIDI4YUM2/N2XI5AT3/E3G4T1R3/S1VOILE2OKA3/T3T1DISPACED1/9AWE1O1/9Z1s1FA/14R/13GO/13AH/3JUVIE4UTA/INRO3FLENCHES ?ANNOPY/AEELNRS 344/368 0 lex CSW21;"

func makeExplainSolver(t *testing.T, bagTailStr string) *Solver {
	t.Helper()
	is := is.New(t)
	g, err := cgp.ParseCGP(DefaultConfig, explainCGP)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	is.NoErr(err)

	peg := &Solver{}
	is.NoErr(peg.Init(g.Game, gd))

	m := move.NewScoringMoveSimple(32, "13M", "P..", "?ANNOY", g.Alphabet())
	peg.SetSolveOnly([]*move.Move{m})
	peg.SetThreads(1)
	peg.SetEndgamePlies(4)
	peg.SetIterativeDeepening(false)
	peg.SetNestedDepthLimit(1)
	peg.SetSkipDeepPass(true)
	peg.SetSkipTiebreaker(true)

	bagTail, err := tilemapping.ToMachineLetters(bagTailStr, g.Alphabet())
	is.NoErr(err)
	peg.SetTraceTargetBagTail(bagTail)
	peg.SetTraceOnce(true)
	peg.SetExplainMode(true)

	return peg
}

func TestExplainModeGRL(t *testing.T) {
	peg := makeExplainSolver(t, "GRL")

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	_, err := peg.Solve(ctx)
	if err != nil && err != ErrCanceledEarly {
		t.Fatalf("Solve: %v", err)
	}

	e := peg.ExplainResult()
	if e == nil {
		t.Fatal("ExplainResult is nil")
	}
	t.Logf("OurPlay=%s Score=%d RackBefore=%s OppRack=%s BagBefore=%s",
		e.OurPlay, e.OurScore, e.OurRackBefore, e.OppRack, e.BagBefore)
	t.Logf("RackAfter=%s BagAfter=%s", e.OurRackAfter, e.BagAfter)
	t.Logf("TotalOppReplies=%d DecisiveOppPlay=%q OppRackAfter=%s BagAfterOppPlay=%s",
		e.TotalOppReplies, e.DecisiveOppPlay, e.OppRackAfter, e.BagAfterOppPlay)
	for _, n := range e.Nested {
		t.Logf("Nested depth=%d OurRack=%s OppRack=%s Bag=%s BestPlay=%q Verdict=%s",
			n.Depth, n.OurRack, n.OppRack, n.BagContent, n.BestPlay, n.Verdict)
		for _, sp := range n.SubPerms {
			marker := ""
			if sp.IsActual {
				marker = " ← ACTUAL"
			}
			t.Logf("  %s ×%d → %s%s", sp.Tiles, sp.Count, sp.Outcome, marker)
		}
	}
	fmt.Printf("GRL verdict: %s\n", e.Verdict)
	fmt.Printf("%s\n", peg.FormatExplanation())

	if e.OurPlay == "" {
		t.Error("OurPlay not populated")
	}
	if e.TotalOppReplies == 0 {
		t.Error("TotalOppReplies not set")
	}
	if e.Verdict != PEGLoss {
		t.Errorf("expected LOSS for GRL, got %s", e.Verdict)
	}
	if e.DecisiveOppPlay == "" {
		t.Error("DecisiveOppPlay not set for LOSS verdict")
	}
}

func TestExplainModeGEI(t *testing.T) {
	peg := makeExplainSolver(t, "GEI")

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	_, err := peg.Solve(ctx)
	if err != nil && err != ErrCanceledEarly {
		t.Fatalf("Solve: %v", err)
	}

	e := peg.ExplainResult()
	if e == nil {
		t.Fatal("ExplainResult is nil")
	}

	fmt.Printf("GEI verdict: %s\n", e.Verdict)
	fmt.Printf("%s\n", peg.FormatExplanation())

	if e.OurPlay == "" {
		t.Error("OurPlay not populated")
	}
	if e.TotalOppReplies == 0 {
		t.Error("TotalOppReplies not set")
	}
	t.Logf("GEI verdict: %s", e.Verdict)
}

func TestExplainModeNoRegression(t *testing.T) {
	// Verify the normal solve path is unaffected by the explain mode additions.
	is := is.New(t)
	g, err := cgp.ParseCGP(DefaultConfig, explainCGP)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	is.NoErr(err)

	peg := &Solver{}
	is.NoErr(peg.Init(g.Game, gd))

	m := move.NewScoringMoveSimple(32, "13M", "P..", "?ANNOY", g.Alphabet())
	peg.SetSolveOnly([]*move.Move{m})
	peg.SetThreads(1)
	peg.SetEndgamePlies(4)
	peg.SetIterativeDeepening(false)
	peg.SetSkipTiebreaker(true)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	winners, err := peg.Solve(ctx)
	is.True(err == nil || err == ErrCanceledEarly)
	is.True(len(winners) > 0)
	t.Logf("Normal solve winner: %s", winners[0].Play.ShortDescription())
}
