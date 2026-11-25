package movegen_test

import (
	"fmt"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"lukechampine.com/frand"

	"github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// TestShadowPreEndgameBound checks that the best-play search finds the best
// play when pre-endgame adjustments are in force. They can be positive (the
// Quackle table gives +10 for leaving 8 tiles unseen), so the shadow bound
// has to include them, as MAGPIE's static_eval_get_shadow_equity does, or
// plays they lift are pruned.
func TestShadowPreEndgameBound(t *testing.T) {
	calc, err := equity.NewCombinedStaticCalculator("CSW21", DefaultConfig, "", "quackle_preendgame.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(calc.PEGValues()) == 0 {
		t.Skip("no quackle_preendgame.json")
	}
	calcs := []equity.EquityCalculator{calc}
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	if err != nil {
		t.Fatal(err)
	}

	// First a rack known to lose its best play (a tie, 7G OE, found by
	// the Zig port's agreement test), then random ones.
	for seed := -1; seed < 600; seed++ {
		inBag := 1 + (seed+6)%6
		parsed, err := cgp.ParseCGP(DefaultConfig, kaOpening)
		if err != nil {
			t.Fatal(err)
		}
		g := parsed.Game
		g.RecalculateBoard()
		g.ThrowRacksInFor(0)
		if seed < 0 {
			inBag = 3
			rack := tilemapping.RackFromString("EEEIOS", g.Alphabet())
			if err := g.SetRackFor(0, rack); err != nil {
				t.Fatal(err)
			}
		} else {
			rng := frand.NewCustom([]byte(fmt.Sprintf("shadow-peg-%021d", seed)), 32, 12)
			if err := g.SetRandomRackSeeded(0, nil, rng); err != nil {
				t.Fatal(err)
			}
		}
		// A few tiles in the bag: with three, playing two leaves 8 unseen, +10.
		buf := make([]tilemapping.MachineLetter, 100)
		if err := g.Bag().Draw(g.Bag().TilesRemaining()-inBag, buf); err != nil {
			t.Fatal(err)
		}

		p, err := turnplayer.NewAIStaticTurnPlayerFromGame(g, DefaultConfig, calcs)
		if err != nil {
			t.Fatal(err)
		}
		top := turnplayer.GenBestStaticTurn(g, p, 0)

		// Every move, valued one by one as the search values them (score
		// plus leave value and adjustment), ties broken the same way.
		all := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
		peg := calc.PEGValues()
		var best *move.Move
		bestEq := 0.0
		for _, m := range all.GenAll(g.RackFor(0), false) {
			if m.Action() != move.MoveTypePlay {
				continue
			}
			adj := calc.LeaveValue(m.Leave())
			if i := inBag - m.TilesPlayed() + game.RackTileLimit; i < len(peg) {
				adj += peg[i]
			}
			eq := float64(m.Score()) + adj
			if best == nil || eq > bestEq || (eq == bestEq && m.TiebreaksBetter(best)) {
				best, bestEq = m, eq
			}
		}
		if best != nil && (top.Equity() != bestEq || top.ShortDescription() != best.ShortDescription()) {
			t.Errorf("seed %d, bag %d, rack %s: best play %s (%v), the search found %s (%v)",
				seed, inBag, g.RackLettersFor(0), best.ShortDescription(), bestEq,
				top.ShortDescription(), top.Equity())
		}
	}
}
