package game_test

import (
	"slices"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// An RD29 game (29 letters with Ä Ö Ü) builds 94-plane, 78-scalar vectors,
// a FILE2017 game (28 tiles with CH LL Ñ RR) 91-plane, 76-scalar ones: the
// float and byte paths agree for every legal move, and every tile on the
// board, umlauts and digraphs included, is in its letter's plane.
func TestGermanMLVectors(t *testing.T) {
	testMLVectors(t, "RD29", "german", 94, 78, []string{"Ä", "Ö", "Ü"})
}

func TestSpanishMLVectors(t *testing.T) {
	testMLVectors(t, "FILE2017", "spanish", 91, 76, []string{"[CH]", "[LL]", "Ñ", "[RR]"})
}

func testMLVectors(t *testing.T, lexicon, dist string, nplanes, nscalars int, special []string) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(DefaultConfig, lexicon, board.CrosswordGameLayout, dist,
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	g, err := game.NewGame(rules, []*pb.PlayerInfo{{Nickname: "a"}, {Nickname: "b"}})
	is.NoErr(err)
	g.StartGame()
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	lay, err := g.MLLayout()
	is.NoErr(err)
	is.Equal(lay.NumPlanes, nplanes)
	is.Equal(lay.NumScalars, nscalars)
	leaves, err := equity.NewExhaustiveLeaveCalculator(lexicon, DefaultConfig, "")
	is.NoErr(err)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), lexicon)
	is.NoErr(err)

	var history []*move.Move
	specials := 0
	for turn := 0; turn < 14 && g.Playing() == pb.PlayState_PLAYING; turn++ {
		gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
		gen.SetGenPass(true)
		moves := gen.GenAll(g.RackFor(g.PlayerOnTurn()), g.Bag().TilesRemaining() >= game.RackTileLimit)
		planes, scalars, err := g.MLVectorsForMoves(moves, leaves, history)
		is.NoErr(err)
		is.Equal(len(planes), len(moves)*lay.PlaneLen)
		is.Equal(len(scalars), len(moves)*lay.NumScalars)
		boards, bscalars, err := g.MLByteVectorsForMoves(moves, leaves, history)
		is.NoErr(err)
		is.Equal(len(boards), len(planes))
		for j := range planes {
			if float32(boards[j]) != planes[j] {
				t.Fatalf("turn %d: plane value %d: float %v byte %v", turn, j, planes[j], boards[j])
			}
		}
		is.Equal(bscalars, scalars)

		best := moves[0]
		for _, m := range moves {
			if m.Score() > best.Score() {
				best = m
			}
		}
		is.NoErr(g.PlayMove(best, false, 0))
		history = append(history, best)
	}
	// The tiles on the board, as the net sees them before a pass.
	pass := move.NewPassMove(g.RackFor(g.PlayerOnTurn()).TilesOn(), g.Alphabet())
	planes, _, err := g.MLVectorsForMoves([]*move.Move{pass}, leaves, history)
	is.NoErr(err)
	b := g.Board()
	tiles := 0
	for r := range game.NN_H {
		for c := range game.NN_W {
			ml := b.GetLetter(r, c)
			if ml == 0 {
				continue
			}
			tiles++
			ll := int(ml.Unblank()) - 1
			if slices.Contains(special, g.Alphabet().Letter(ml.Unblank())) {
				specials++
			}
			is.Equal(planes[ll*game.NN_H*game.NN_W+r*game.NN_W+c], float32(1))
		}
	}
	t.Logf("%s: %d tiles on the board, %d of %v", lexicon, tiles, specials, special)
	is.True(tiles > 30)
}
