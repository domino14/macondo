package montecarlo

import (
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/matryer/is"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/movegen"
)

// With one tile in the bag every rollout empties it, so a sim searches the
// endgames it reaches, and plays them out statically when told not to.
func TestEndgameLeaves(t *testing.T) {
	is := is.New(t)
	cgpStr := "15/3Q7U3/3U2TAURINE2/1CHANSONS2W3/2AI6JO3/DIRL1PO3IN3/E1D2EF3V4/F1I2p1TRAIK3/O1L2T4E4/ABy1PIT2BRIG2/ME1MOZELLE5/1GRADE1O1NOH3/WE3R1V7/AT5E7/G6D7 ENOSTXY/ACEISUY 356/378 0 lex NWL20;"
	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()
	is.True(g.Bag().TilesRemaining() <= EndgameLeafMaxBag)
	calcs, leaves := defaultSimCalculators("NWL20")
	gd, err := kwg.GetKWG(g.Config().WGLConfig(), g.LexiconName())
	is.NoErr(err)
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Rules().LetterDistribution())
	gen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
	plays := gen.Plays()[:10]

	leafCount := func() int64 {
		return EndgameLeafSolves.Load() + EndgameLeafCacheHits.Load() + EndgameLeafFallbacks.Load()
	}
	sim := func(on bool) int64 {
		s := &Simmer{}
		s.Init(g.Game, calcs, leaves.(*equity.CombinedStaticCalculator), DefaultConfig)
		s.SetThreads(1)
		s.SetEndgameLeaves(on)
		is.NoErr(s.PrepareSim(2, plays))
		before := leafCount()
		s.SimSingleThread(50, 2)
		is.True(s.WinningPlay() != nil)
		return leafCount() - before
	}
	solvesBefore := EndgameLeafSolves.Load()
	is.True(sim(true) > 0)
	is.True(EndgameLeafSolves.Load() > solvesBefore)
	is.Equal(sim(false), int64(0))
}
