package movegen_test

import (
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/movegen"
	wmppkg "github.com/domino14/macondo/wmp"
)

// TestWMPGeneratorReuse checks that a generator that has just found a best
// play with the word map generates every move correctly afterwards: the
// word-map generator sets tilesPlayed for each anchor, and a later
// generation must not start from what it left.
func TestWMPGeneratorReuse(t *testing.T) {
	w, err := wmppkg.EnsureWMP(DefaultConfig.WGLConfig(), "NWL20")
	if err != nil {
		t.Skipf("no NWL20 word map: %v", err)
	}
	parsed, err := cgp.ParseCGP(DefaultConfig, "15/15/15/15/15/15/15/6KA7/15/15/15/15/15/15/15 AEINRST/ADEEGIL 0/0 0 lex NWL20;")
	if err != nil {
		t.Fatal(err)
	}
	g := parsed.Game
	g.RecalculateBoard()
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	if err != nil {
		t.Fatal(err)
	}
	ld := g.Bag().LetterDistribution()
	calc, err := equity.NewCombinedStaticCalculator("NWL20", DefaultConfig, "", "")
	if err != nil {
		t.Fatal(err)
	}
	rack := tilemapping.RackFromString("AEINRST", g.Alphabet())

	fresh := movegen.NewGordonGenerator(gd, g.Board(), ld)
	want := len(fresh.GenAll(rack, false))

	reused := movegen.NewGordonGenerator(gd, g.Board(), ld)
	reused.SetEquityCalculators([]equity.EquityCalculator{calc})
	reused.SetGame(g)
	reused.SetWMP(w)
	reused.SetPlayRecorderTopPlay()
	reused.GenAll(rack, false)
	reused.SetPlayRecorder(movegen.AllPlaysRecorder)
	if got := len(reused.GenAll(rack, false)); got != want {
		t.Errorf("after a word-map best play: %d moves, want %d", got, want)
	}
}
