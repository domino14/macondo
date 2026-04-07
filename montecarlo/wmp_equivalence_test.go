// wmp_equivalence_test.go: verifies that the WMP-on and WMP-off paths
// produce equivalent top plays for the same game / rack. If WMP
// integration is functionally correct, both paths must pick the same
// top move.
package montecarlo

import (
	"os"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/movegen"
	wmppkg "github.com/domino14/macondo/wmp"
)

// TestWMPEquivalentTopPlay loads a CGP, then for each of several test
// racks runs the GenAllWithShadow + TopPlayOnlyRecorder pipeline once
// with WMP off and once with WMP on. Both runs must pick the same
// top play (same word, same coords, same score, same equity).
func TestWMPEquivalentTopPlay(t *testing.T) {
	is := is.New(t)

	if _, err := os.Stat(benchMagpieCSW24WMP); err != nil {
		t.Skipf("CSW24.wmp not present at %s", benchMagpieCSW24WMP)
	}
	if _, err := os.Stat(benchMacondoDataPath + "/lexica/gaddag/CSW24.kwg"); err != nil {
		t.Skipf("CSW24.kwg not at %s/lexica/gaddag", benchMacondoDataPath)
	}
	DefaultConfig.Set("data-path", benchMacondoDataPath)

	w, err := wmppkg.LoadFromFile("CSW24", benchMagpieCSW24WMP)
	is.NoErr(err)

	// Build a fresh generator for each call so leave map / shadow
	// state doesn't bleed across runs.
	makeGenerator := func() (*movegen.GordonGenerator, *cgp.ParsedCGP, *equity.CombinedStaticCalculator) {
		game, err := cgp.ParseCGP(DefaultConfig, benchSimCGP)
		is.NoErr(err)
		game.RecalculateBoard()

		calc, err := equity.NewCombinedStaticCalculator(
			"CSW24", DefaultConfig, "", equity.PEGAdjustmentFilename)
		is.NoErr(err)

		gd, err := kwg.GetKWG(game.Config().WGLConfig(), game.LexiconName())
		is.NoErr(err)

		gen := movegen.NewGordonGenerator(gd, game.Board(), game.Rules().LetterDistribution())
		gen.SetEquityCalculators([]equity.EquityCalculator{calc})
		gen.SetGame(game.Game)
		gen.SetPlayRecorderTopPlay()
		return gen, game, calc
	}

	racks := []string{
		"EEEIILZ", // benchmark rack
		"AEINRST", // common bingo rack
		"AEILORS",
		"DGIQTUV",
		"BFHKMOR",
	}

	for _, rackStr := range racks {
		t.Run(rackStr, func(t *testing.T) {
			// No WMP path
			noGen, noGame, _ := makeGenerator()
			noRack := tilemapping.RackFromString(rackStr, noGame.Alphabet())
			noGen.GenAll(noRack, false)
			noTop := noGen.Plays()[0]

			// WMP path (separate generator instance)
			wmpGen, wmpGame, _ := makeGenerator()
			wmpGen.SetWMP(w)
			wmpRack := tilemapping.RackFromString(rackStr, wmpGame.Alphabet())
			wmpGen.GenAll(wmpRack, false)
			wmpTop := wmpGen.Plays()[0]

			if noTop.Score() != wmpTop.Score() {
				t.Errorf("rack %s top score mismatch: noWMP=%d (%s) WMP=%d (%s)",
					rackStr,
					noTop.Score(), noTop.ShortDescription(),
					wmpTop.Score(), wmpTop.ShortDescription())
			}
			if !movesEqual(noTop, wmpTop, noGame.Alphabet()) {
				t.Errorf("rack %s top play mismatch:\n  noWMP: %s (eq %.4f)\n  WMP:   %s (eq %.4f)",
					rackStr,
					noTop.ShortDescription(), noTop.Equity(),
					wmpTop.ShortDescription(), wmpTop.Equity())
			}
			if absDiff(noTop.Equity(), wmpTop.Equity()) > 1e-6 {
				t.Errorf("rack %s equity mismatch: noWMP=%.6f WMP=%.6f",
					rackStr, noTop.Equity(), wmpTop.Equity())
			}
		})
	}
}

func absDiff(a, b float64) float64 {
	if a > b {
		return a - b
	}
	return b - a
}

// movesEqual checks structural equality between two moves (the same
// word at the same coords with the same direction). Doesn't check
// score/equity — those have their own assertions in the caller.
func movesEqual(a, b interface {
	ShortDescription() string
}, alph *tilemapping.TileMapping) bool {
	return a.ShortDescription() == b.ShortDescription()
}
