package magpie_test

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/alphadawg"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// WordSmog: MAGPIE generating from the .kad alpha graph gives exactly the
// plays Macondo's alpha generator does, with the same scores, and its
// cross-sets follow the game. Racks with a blank are not compared (a blank
// multiplies the permutations past what is worth enumerating twice).
func TestWordSmogAgreement(t *testing.T) {
	dataPath := defaultConfig.GetString(config.ConfigDataPath)
	kadPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.kad")
	klvPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.klv2")
	for _, p := range []string{kadPath, klvPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("%s not available", p)
		}
	}
	ld, _ := tilemapping.GetDistribution(defaultConfig.WGLConfig(), "english")
	kad, err := alphadawg.Get(defaultConfig.WGLConfig(), "NWL23", "english")
	if err != nil {
		t.Fatal(err)
	}
	d, err := magpie.NewData(magpie.DataOptions{KWGPath: kadPath, KLVPath: klvPath, Dist: ld,
		Layout: board.CrosswordGameBoard, Variant: magpie.VariantWordSmog, BingoBonus: 50})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	pos := magpie.NewPosition(d, 1, 1)
	defer pos.Close()
	mb := d.NewBoard()
	rng := rand.New(rand.NewPCG(8, 9))
	letters := uint64(1)<<len(ld.Distribution()) - 2
	compared, plays := 0, 0
	for g := 0; g < 6; g++ {
		bd := board.MakeBoard(board.CrosswordGameBoard)
		csgen := &cross_set.WordSmogCrossSetGenerator{Dist: ld, AlphaDawg: kad}
		csgen.GenerateAll(bd)
		bd.UpdateAllAnchors()
		gen := movegen.NewGordonGenerator(kad, bd, ld)
		gen.SetWordSmog(kad)
		gen.SetSortingParameter(movegen.SortByNone)
		bag := ld.MakeBag()
		bag.Shuffle()
		racks := [2][]tilemapping.MachineLetter{make([]tilemapping.MachineLetter, 7), make([]tilemapping.MachineLetter, 7)}
		bag.Draw(7, racks[0])
		bag.Draw(7, racks[1])
		onturn := 0
		for turn := 0; turn < 12 && bag.TilesRemaining() > 7; turn++ {
			squares := make([]tilemapping.MachineLetter, 225)
			for r := 0; r < 15; r++ {
				for c := 0; c < 15; c++ {
					squares[r*15+c] = bd.GetLetter(r, c)
				}
			}
			var mr [2]tilemapping.MachineWord
			mr[0], mr[1] = racks[0], racks[1]
			if err := pos.Set(magpie.State{Squares: squares, Racks: mr, Bag: bag.Peek(), OnTurn: onturn}); err != nil {
				t.Fatal(err)
			}
			// MAGPIE's cross-sets match Macondo's WordSmog ones.
			pos.ReadBoard(mb)
			for i := 0; i < 225; i++ {
				if squares[i] != 0 {
					continue
				}
				r, c := i/15, i%15
				for dir, md := range []board.BoardDirection{board.VerticalDirection, board.HorizontalDirection} {
					if uint64(bd.GetCrossSet(r, c, md))&letters != mb.CrossSets[dir*225+i]&letters {
						t.Fatalf("game %d turn %d: cross-set %d,%d dir %d differs", g, turn, r, c, dir)
					}
				}
			}
			rack := tilemapping.NewRack(ld.TileMapping())
			rack.Set(racks[onturn])
			var placements []*move.Move
			for _, m := range gen.GenAll(rack, false) {
				if m.Action() == move.MoveTypePlay {
					placements = append(placements, m)
				}
			}
			if rack.LetArr[0] == 0 {
				compared++
				want := map[string]int{}
				for _, m := range placements {
					want[playKey(m)] = m.Score()
				}
				got := map[string]int{}
				for _, m := range pos.GenerateN(magpie.RecordAll, magpie.SortScore, 0, 2000000) {
					if m.Type == magpie.MovePlacement {
						got[magpieKey(&m)] = int(m.Score / magpie.EquityResolution)
					}
				}
				plays += len(want)
				if len(got) != len(want) {
					t.Fatalf("game %d turn %d: %d plays in MAGPIE, %d in Macondo", g, turn, len(got), len(want))
				}
				for k, s := range want {
					if got[k] != s {
						t.Fatalf("game %d turn %d: %s scores %d in Macondo, %d in MAGPIE", g, turn, k, s, got[k])
					}
				}
			}
			if len(placements) == 0 {
				break
			}
			sort.Slice(placements, func(i, j int) bool { return placements[i].Score() > placements[j].Score() })
			m := placements[rng.IntN(min(5, len(placements)))]
			bd.PlayMove(m)
			csgen.UpdateForMove(bd, m)
			leave, _ := tilemapping.Leave(racks[onturn], m.Tiles(), true)
			drawn := make([]tilemapping.MachineLetter, m.TilesPlayed())
			bag.Draw(len(drawn), drawn)
			racks[onturn] = append(leave, drawn...)
			onturn = 1 - onturn
		}
	}
	t.Logf("compared %d positions, %d plays", compared, plays)
}
