package movegen

import (
	"math"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// TestBestLeavesFromLeaveMap verifies that bestLeaves computed from
// leave map rawLeave values matches computeBestLeaves (enumerateLeaves).
// This test exposes the bug where the leave map's rawLeave-based
// bestLeaves diverges from the canonical computation.
func TestBestLeavesFromLeaveMap(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL23",
		board.CrosswordGameLayout, "English",
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL23")
	is.NoErr(err)
	calc, err := equity.NewCombinedStaticCalculator("NWL23", DefaultConfig, "", "")
	is.NoErr(err)
	klv := calc.KLV()
	playerInfos := []*pb.PlayerInfo{{Nickname: "p1", RealName: "P1"}, {Nickname: "p2", RealName: "P2"}}

	// Replay game 1 (which had disagreements) and check bestLeaves at each turn
	g, _ := game.NewGame(rules, playerInfos)
	g.SeedBag([32]byte{1})
	g.StartGame()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	gen := NewGordonGenerator(gd, bd, g.Bag().LetterDistribution())
	gen.klv = klv
	gen.pegValues = calc.PEGValues()
	gen.equityCalculators = []equity.EquityCalculator{calc}

	genPlay := NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())

	for turn := 0; turn < 25 && g.Playing() == pb.PlayState_PLAYING; turn++ {
		rack := g.RackFor(g.PlayerOnTurn())
		gen.tilesInBag = g.Bag().TilesRemaining()
		if gen.tilesInBag == 0 {
			break // endgame bestLeaves uses different formula; tested separately
		}
		gen.oppRackScore = 0
		oppRack := g.RackFor(g.NextPlayer())
		for ml := tilemapping.MachineLetter(1); int(ml) < len(oppRack.LetArr); ml++ {
			gen.oppRackScore += oppRack.LetArr[ml] * gen.letterDistribution.Score(ml)
		}
		gen.shadow.tilesInBag = gen.tilesInBag
		gen.shadow.oppRackScore = gen.oppRackScore

		// Populate leave map
		// Initialize bestLeaves (as GenAllWithShadow does)
		for i := range gen.shadow.bestLeaves {
			gen.shadow.bestLeaves[i] = math.Inf(-1)
		}
		gen.leavemap.init(rack)
		if !gen.leavemap.initialized {
			continue
		}
		gen.populateLeaveMap(rack)

		// bestLeaves was populated from rawLeave during populateLeaveMap
		lmBest := gen.shadow.bestLeaves

		// Compute bestLeaves via canonical path
		gen.computeBestLeaves(rack)
		computeBest := gen.shadow.bestLeaves

		for k := 0; k <= int(rack.NumTiles()); k++ {
			diff := lmBest[k] - computeBest[k]
			if math.Abs(diff) > 0.001 {
				t.Errorf("turn %d rack=%s bestLeaves[%d]: lm=%.4f compute=%.4f diff=%.4f tilesInBag=%d",
					turn, rack.String(), k, lmBest[k], computeBest[k], diff, gen.tilesInBag)
			}
		}

		// Advance game
		genPlay.SetPlayRecorder(AllPlaysRecorder)
		plays := genPlay.GenAll(rack, false)
		if len(plays) > 0 {
			g.PlayMove(plays[0], false, 0)
		}
	}
}
