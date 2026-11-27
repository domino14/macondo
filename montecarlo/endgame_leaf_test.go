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

// The leaf searches' shared table is made for a board size: a sim on the
// 21x21 board after one on the 15x15 board must use a table of its own
// (it indexed the other's position keys out of range).
func TestEndgameLeavesOnBothBoardSizes(t *testing.T) {
	is := is.New(t)
	for _, c := range []struct{ cgp, leaves string }{
		{"15/3Q7U3/3U2TAURINE2/1CHANSONS2W3/2AI6JO3/DIRL1PO3IN3/E1D2EF3V4/F1I2p1TRAIK3/O1L2T4E4/ABy1PIT2BRIG2/ME1MOZELLE5/1GRADE1O1NOH3/WE3R1V7/AT5E7/G6D7 ENOSTXY/ACEISUY 356/378 0 lex NWL20;", ""},
		{"17H2K/16VOLTI/5YARTO7R2L/9FILBERT1ACED/15EWHoW1/15T2N2/15T2S2/C12TREELINE/HM11W1R2S1T/OIS10A4T1A/CAA7CANNY1QIS1C/1OX4TOQUE1G5HI/OU1FAD4R1FILTH2US/P3MAJOR1A2e5MM/E4MOFETTE1R3B1B1/R1IDEE4I4W1I1L1/ABOON3V1ONCOGENS1ED/N2PERIGEAN2X1A1T2A/D7LAST1E1K4Z/SPITZ3U3NY2JEUNE/1ODEA1VUM3YE1GORP1R ?EEENNU/?DDIOSS 929/924 0 lex CSW24; ld english_super; bdn SuperCrosswordGame;", "super-leaves.klv2"},
	} {
		g, err := cgp.ParseCGP(DefaultConfig, c.cgp)
		is.NoErr(err)
		g.RecalculateBoard()
		is.True(g.Bag().TilesRemaining() <= EndgameLeafMaxBag)
		calc, err := equity.NewCombinedStaticCalculator(g.LexiconName(), DefaultConfig, c.leaves, equity.PEGAdjustmentFilename)
		is.NoErr(err)
		gd, err := kwg.GetKWG(g.Config().WGLConfig(), g.LexiconName())
		is.NoErr(err)
		gen := movegen.NewGordonGenerator(gd, g.Board(), g.Rules().LetterDistribution())
		gen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
		plays := gen.Plays()[:min(10, len(gen.Plays()))]
		s := &Simmer{}
		s.Init(g.Game, []equity.EquityCalculator{calc}, calc, DefaultConfig)
		s.SetThreads(1)
		is.NoErr(s.PrepareSim(2, plays))
		before := EndgameLeafSolves.Load() + EndgameLeafCacheHits.Load() + EndgameLeafFallbacks.Load()
		s.SimSingleThread(20, 2)
		is.True(EndgameLeafSolves.Load()+EndgameLeafCacheHits.Load()+EndgameLeafFallbacks.Load() > before)
	}
}
