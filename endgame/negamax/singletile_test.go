package negamax

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/tinymove/conversions"
)

// Boards to scan every single-tile placement on. Two dense endgame boards plus
// one with an open lane, so squares with neighbours in one direction, the other,
// and both are all exercised.
var singleTileBoards = []struct {
	name string
	cgp  string
	lex  string
}{
	{"EldarVsNigel", stuckEndgameCGP, "CSW19"},
	{"PassFirst", openEndgameCGP, "CSW19"},
	{
		"Complicated1PEG",
		"13AW/11F1LI/10JURAT/9LINER1/8O1T4/5C1WAsTiNG1/4DAMAR1E4/3PARED2ROUEN/2YA1K9/1BERG1OATH4V/3COUP1I5E/3H1TESTILY2N/4FAN1I2OXID/9MIB2U/7ZEES3E EEILOSS/ 297/300 0 lex NWL20;",
		"NWL20",
	},
}

// TestBestSingleTilePlayMatchesMovegen is the correctness net for the one-tile
// board scan. For every tile in the alphabet, on every board, the scan's chosen
// placement must match what full move generation would pick: same score, and a
// TinyMove that decodes to the same tiles at the same coordinates.
//
// A wrong cross-score axis, a bad bonus multiplier, or a mis-encoded row/col all
// surface here.
func TestBestSingleTilePlayMatchesMovegen(t *testing.T) {
	is := is.New(t)
	for _, bd := range singleTileBoards {
		g, err := cgp.ParseCGP(DefaultConfig, bd.cgp)
		is.NoErr(err)
		g.RecalculateBoard()
		g.SetBackupMode(game.SimulationMode)
		ld := g.Bag().LetterDistribution()
		alph := g.Alphabet()

		checked, playable := 0, 0
		numLetters := int(alph.NumLetters())
		// ml 0 is the blank; cover it too.
		for ml := 0; ml < numLetters; ml++ {
			rack := tilemapping.NewRack(alph)
			rack.Add(tilemapping.MachineLetter(ml))
			if rack.NumTiles() != 1 {
				continue
			}
			checked++

			// Ground truth from full move generation.
			wantBest, haveTilePlay := bestTilePlayScore(t, g.Game, tilemapping.MachineLetter(ml))

			got, ok := bestSingleTilePlay(g.Board(), rack, ld)
			if !haveTilePlay {
				if ok {
					t.Errorf("%s tile=%s: scan found a play (%d) but movegen found none",
						bd.name, tilemapping.MachineLetter(ml).UserVisible(alph, false), got.Score())
				}
				continue
			}
			playable++
			if !ok {
				t.Errorf("%s tile=%s: scan found no play but movegen best is %d",
					bd.name, tilemapping.MachineLetter(ml).UserVisible(alph, false), wantBest)
				continue
			}
			if got.Score() != wantBest {
				t.Errorf("%s tile=%s: scan score %d != movegen best %d",
					bd.name, tilemapping.MachineLetter(ml).UserVisible(alph, false),
					got.Score(), wantBest)
				continue
			}
			// The encoding must decode back to a real play whose score, when
			// independently recomputed off the board, matches.
			var m tinymove.SmallMove = got
			full, err := conversions.TinyMoveToFullMove(m.TinyMove(), g.Board(), ld, rack)
			if err != nil {
				t.Errorf("%s tile=%s: TinyMove does not decode: %v",
					bd.name, tilemapping.MachineLetter(ml).UserVisible(alph, false), err)
				continue
			}
			if full.Score() != got.Score() {
				t.Errorf("%s tile=%s: re-scored %d != scan %d (play %s)",
					bd.name, tilemapping.MachineLetter(ml).UserVisible(alph, false),
					full.Score(), got.Score(), full.ShortDescription())
			}
			if full.TilesPlayed() != 1 {
				t.Errorf("%s tile=%s: decoded %d tiles played, want 1",
					bd.name, tilemapping.MachineLetter(ml).UserVisible(alph, false), full.TilesPlayed())
			}
			if got.PlayLength() != full.PlayLength() {
				t.Errorf("%s tile=%s: play length %d != decoded %d",
					bd.name, tilemapping.MachineLetter(ml).UserVisible(alph, false),
					got.PlayLength(), full.PlayLength())
			}
		}
		t.Logf("%s: checked %d tiles, %d playable", bd.name, checked, playable)
		is.True(checked > 20)
		is.True(playable > 0)
	}
}

// TestBestSingleTilePlayAcrossPositions widens the net: rather than three fixed
// boards, walk each one forward a dozen plies and re-check every tile at every
// position. Each play changes which squares are open, which cross sets are
// constrained, and which squares have neighbours in one direction versus both,
// so this covers far more square topologies than the static boards do.
func TestBestSingleTilePlayAcrossPositions(t *testing.T) {
	is := is.New(t)
	for _, bd := range singleTileBoards {
		g, err := cgp.ParseCGP(DefaultConfig, bd.cgp)
		is.NoErr(err)
		g.RecalculateBoard()
		g.SetBackupMode(game.SimulationMode)
		g.SetStateStackLength(30)
		g.SetEndgameMode(true)
		ld := g.Bag().LetterDistribution()
		alph := g.Alphabet()

		numLetters := int(alph.NumLetters())
		positions, comparisons := 0, 0
		played := 0
		for step := 0; step < 12; step++ {
			positions++
			for ml := 0; ml < numLetters; ml++ {
				probe := tilemapping.NewRack(alph)
				probe.Add(tilemapping.MachineLetter(ml))
				if probe.NumTiles() != 1 {
					continue
				}
				want, have := bestTilePlayScore(t, g.Game, tilemapping.MachineLetter(ml))
				got, ok := bestSingleTilePlay(g.Board(), probe, ld)
				if ok != have {
					t.Fatalf("%s step %d tile %s: scan found=%v, movegen found=%v",
						bd.name, step, tilemapping.MachineLetter(ml).UserVisible(alph, false), ok, have)
				}
				if have {
					comparisons++
					if got.Score() != want {
						t.Fatalf("%s step %d tile %s: scan %d != movegen %d",
							bd.name, step, tilemapping.MachineLetter(ml).UserVisible(alph, false),
							got.Score(), want)
					}
					// Re-score the encoded move off the board independently.
					full, err := conversions.TinyMoveToFullMove(got.TinyMove(), g.Board(), ld, probe)
					is.NoErr(err)
					if full.Score() != got.Score() {
						t.Fatalf("%s step %d tile %s: re-scored %d != %d (%s)",
							bd.name, step, tilemapping.MachineLetter(ml).UserVisible(alph, false),
							full.Score(), got.Score(), full.ShortDescription())
					}
				}
			}
			// Advance the board with a real play so the next position differs.
			onTurn := g.RackFor(g.PlayerOnTurn())
			if onTurn.NumTiles() == 0 {
				break
			}
			plays := onTurnSmallPlays(t, g.Game)
			bi, bs, found := 0, -1, false
			for j := range plays {
				if plays[j].IsPass() {
					continue
				}
				if plays[j].Score() > bs {
					bs, bi, found = plays[j].Score(), j, true
				}
			}
			if !found {
				break
			}
			_, err := g.PlaySmallMove(&plays[bi])
			is.NoErr(err)
			played++
			if g.Playing() != pb.PlayState_PLAYING {
				break
			}
		}
		// How many positions we get varies: a greedy 7-tile rack plays itself
		// out in a couple of plies, while a cramped board runs longer. The
		// comparison count is what matters.
		t.Logf("%s: %d positions, %d playable comparisons", bd.name, positions, comparisons)
		is.True(positions >= 2)
		is.True(comparisons > 40)
		for i := 0; i < played; i++ {
			g.UnplayLastMove()
		}
	}
}

// TestBestSingleTilePlayStuckTile: the lone V in EldarVsNigel is unplayable, and
// the scan has to say so rather than inventing a placement.
func TestBestSingleTilePlayStuckTile(t *testing.T) {
	is := is.New(t)
	g, err := cgp.ParseCGP(DefaultConfig, stuckEndgameCGP)
	is.NoErr(err)
	g.RecalculateBoard()
	ld := g.Bag().LetterDistribution()
	rack := tilemapping.RackFromString("V", g.Alphabet())
	_, ok := bestSingleTilePlay(g.Board(), rack, ld)
	is.True(!ok)
}

// bestTilePlayScore is the best score the player on turn could make holding
// only tile, from MAGPIE's full small-move generation.
func bestTilePlayScore(t *testing.T, g *game.Game, tile tilemapping.MachineLetter) (int, bool) {
	t.Helper()
	pos, err := g.MagpiePosition()
	if err != nil {
		t.Fatal(err)
	}
	var racks [2]tilemapping.MachineWord
	racks[g.PlayerOnTurn()] = tilemapping.MachineWord{tile}
	racks[1-g.PlayerOnTurn()] = g.RackFor(1 - g.PlayerOnTurn()).TilesOn()
	pos.SetRacksAndBag(racks[0], racks[1], g.Bag().Tiles(), g.PlayerOnTurn())
	best, have := 0, false
	for _, m := range pos.GenerateSmall(false, make([]magpie.SmallMove, 10000)) {
		if m.TM != 0 && (!have || int(m.Score) > best) {
			best, have = int(m.Score), true
		}
	}
	return best, have
}

// onTurnSmallPlays is every play for the player on turn, a pass included.
func onTurnSmallPlays(t *testing.T, g *game.Game) []tinymove.SmallMove {
	t.Helper()
	pos, err := g.MagpiePosition()
	if err != nil {
		t.Fatal(err)
	}
	var plays []tinymove.SmallMove
	for _, m := range pos.GenerateSmall(true, make([]magpie.SmallMove, 100000)) {
		if m.TM == 0 {
			plays = append(plays, tinymove.PassMove())
			continue
		}
		plays = append(plays, tinymove.TilePlayMove(tinymove.TinyMove(m.TM), m.Score, m.TilesPlayed, m.PlayLength))
	}
	return plays
}
