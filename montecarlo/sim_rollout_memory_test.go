package montecarlo

import (
	"context"
	"testing"

	"github.com/rs/zerolog"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/movegen"
)

// TestRolloutMemoryAgrees sims positions from every stage of a game with
// the rollouts' shortcuts that remember earlier work switched on -- the
// move generators' memory of best plays, lanes and rack setup by rack
// (movegen/carry.go) and the cross-set caches -- and with every rollout move
// generated a second time by a generator that has none of them, on a board
// whose every cached cross-set is recomputed from the lexicon. Any
// difference panics. It is run with and without the word map, on one thread
// and on several.
func TestRolloutMemoryAgrees(t *testing.T) {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	defer zerolog.SetGlobalLevel(zerolog.DebugLevel)
	const lexicon = "NWL18"
	stops := []int{84, 70, 55, 40, 25, 12, 8}
	iters := 100
	if testing.Short() {
		stops = []int{70, 40, 12}
		iters = 30
	}
	positions := simSpeedPositions(t, lexicon, stops)
	for _, useWMP := range []bool{false, true} {
		var rackHits, lanesSkipped, seeded uint64
		for _, pos := range positions {
			for _, threads := range []int{1, 3} {
				parsed, err := cgp.ParseCGP(DefaultConfig, pos)
				if err != nil {
					t.Fatal(err)
				}
				parsed.RecalculateBoard()
				g := parsed.Game
				calcs, leaves := defaultSimCalculators(lexicon)
				p, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(g, DefaultConfig, calcs)
				if err != nil {
					t.Fatal(err)
				}
				plays := p.GenerateMoves(15)
				s := &Simmer{}
				s.Init(g, calcs, leaves.(*equity.CombinedStaticCalculator), DefaultConfig)
				s.SetThreads(threads)
				s.checkBestMoveCarry = true
				if useWMP {
					s.TryLoadWMP(DefaultConfig.WGLConfig(), lexicon)
					if s.WMP() == nil {
						t.Log("no word map; skipping that half")
						return
					}
				}
				if err := s.PrepareSim(5, plays); err != nil {
					t.Fatal(err)
				}
				if err := s.simulate(context.Background(), simRun{maxIters: iters, quiet: true}); err != nil {
					t.Fatal(err)
				}
				for _, ai := range s.aiplayers {
					_, hits, _, skipped, seeds := ai.MoveGenerator().(*movegen.GordonGenerator).BestMoveCarryStats()
					rackHits += hits
					lanesSkipped += skipped
					seeded += seeds
				}
				for _, c := range s.crossSetCaches {
					if c.Hits == 0 {
						t.Fatal("a cross-set cache was never hit")
					}
				}
			}
		}
		// The test means nothing unless the memory was used.
		if rackHits == 0 || lanesSkipped == 0 || seeded == 0 {
			t.Fatalf("word map %v: the memory was not used: %d rack hits, %d lanes skipped, %d generations seeded",
				useWMP, rackHits, lanesSkipped, seeded)
		}
		t.Logf("word map %v: %d rack hits, %d lanes skipped, %d generations seeded",
			useWMP, rackHits, lanesSkipped, seeded)
	}
}
