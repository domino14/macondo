package magpie_test

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// At endgame positions (empty bag) MAGPIE's small moves are Gordon's, the
// pruned word graph generates exactly what the full one does, and
// PlaySmall/UnplaySmall round-trip.
func TestSmallMovesAgree(t *testing.T) {
	dataPath := defaultConfig.GetString(config.ConfigDataPath)
	kwgPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.kwg")
	klvPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.klv2")
	if _, err := os.Stat(klvPath); err != nil {
		t.Skip("NWL23 not available")
	}
	gd, _ := kwg.GetKWG(defaultConfig.WGLConfig(), "NWL23")
	ld, _ := tilemapping.GetDistribution(defaultConfig.WGLConfig(), "english")
	d, err := magpie.NewData(magpie.DataOptions{KWGPath: kwgPath, KLVPath: klvPath, Dist: ld,
		Layout: board.CrosswordGameBoard, Variant: magpie.VariantClassic, BingoBonus: 50})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	pos := magpie.NewPosition(d, 1, 1)
	defer pos.Close()
	rng := rand.New(rand.NewPCG(5, 6))
	buf := make([]magpie.SmallMove, 100000)
	buf2 := make([]magpie.SmallMove, 100000)
	strip := make([]tilemapping.MachineLetter, 21)
	before, after := d.NewBoard(), d.NewBoard()
	positions := 0
	for g := 0; positions < 60; g++ {
		bd := board.MakeBoard(board.CrosswordGameBoard)
		csgen := &cross_set.GaddagCrossSetGenerator{Dist: ld, Gaddag: gd}
		csgen.GenerateAll(bd)
		bd.UpdateAllAnchors()
		gen := movegen.NewGordonGenerator(gd, bd, ld)
		bag := ld.MakeBag()
		bag.Shuffle()
		racks := [2][]tilemapping.MachineLetter{make([]tilemapping.MachineLetter, 7), make([]tilemapping.MachineLetter, 7)}
		bag.Draw(7, racks[0])
		bag.Draw(7, racks[1])
		onturn := 0
		// Play top-ish moves until the bag is empty.
		for bag.TilesRemaining() > 0 {
			rack := tilemapping.NewRack(ld.TileMapping())
			rack.Set(racks[onturn])
			var plays []*move.Move
			for _, m := range gen.GenAll(rack, false) {
				if m.Action() == move.MoveTypePlay {
					plays = append(plays, m)
				}
			}
			if len(plays) == 0 {
				break
			}
			sort.Slice(plays, func(i, j int) bool { return plays[i].Score() > plays[j].Score() })
			m := plays[rng.IntN(min(5, len(plays)))]
			bd.PlayMove(m)
			csgen.UpdateForMove(bd, m)
			leave, _ := tilemapping.Leave(racks[onturn], m.Tiles(), true)
			n := min(m.TilesPlayed(), bag.TilesRemaining())
			drawn := make([]tilemapping.MachineLetter, n)
			bag.Draw(n, drawn)
			racks[onturn] = append(leave, drawn...)
			onturn = 1 - onturn
		}
		if bag.TilesRemaining() > 0 {
			continue
		}
		positions++
		squares := make([]tilemapping.MachineLetter, 225)
		for r := 0; r < 15; r++ {
			for c := 0; c < 15; c++ {
				squares[r*15+c] = bd.GetLetter(r, c)
			}
		}
		var mr [2]tilemapping.MachineWord
		mr[0], mr[1] = racks[0], racks[1]
		if err := pos.Set(magpie.State{Squares: squares, Racks: mr, OnTurn: onturn}); err != nil {
			t.Fatal(err)
		}

		rack := tilemapping.NewRack(ld.TileMapping())
		rack.Set(racks[onturn])
		gen.SetPlayRecorder(movegen.AllPlaysSmallRecorder)
		gen.SetGenPass(true)
		gen.GenAll(rack, false)
		want := map[uint64]int{}
		for _, sm := range gen.SmallPlays() {
			want[uint64(sm.TinyMove())] = sm.Score()
		}
		gen.SetPlayRecorder(movegen.AllPlaysRecorder)
		gen.SetGenPass(false)

		got := pos.GenerateSmall(true, buf)
		if len(got) != len(want) {
			t.Fatalf("game %d: %d small moves in MAGPIE, %d in Macondo", g, len(got), len(want))
		}
		for _, sm := range got {
			s, ok := want[sm.TM]
			if !ok || s != int(sm.Score) {
				t.Fatalf("game %d: MAGPIE move %x (score %d) not in Macondo's (%v %d)", g, sm.TM, sm.Score, ok, s)
			}
		}

		// The pruned word graph generates the same plays.
		k := pos.PrunedKWG()
		pos.SetOverrideKWG(k)
		pruned := pos.GenerateSmall(true, buf2)
		if len(pruned) != len(got) {
			t.Fatalf("game %d: %d plays with the pruned graph, %d with the full", g, len(pruned), len(got))
		}
		for _, sm := range pruned {
			if _, ok := want[sm.TM]; !ok {
				t.Fatalf("game %d: pruned graph generates %x", g, sm.TM)
			}
		}

		// Play and unplay each move.
		pos.ReadBoard(before)
		for _, sm := range pruned {
			n, err := pos.PlaySmall(sm.TM, int(sm.Score), strip)
			if err != nil {
				t.Fatal(err)
			}
			if sm.TM != 0 && n != int(sm.PlayLength) {
				t.Fatalf("game %d: strip of %d for a play %d long", g, n, sm.PlayLength)
			}
			pos.ReadBoard(after)
			if err := pos.UnplaySmall(); err != nil {
				t.Fatal(err)
			}
			pos.ReadBoard(after)
			if !boardsEqual(before, after) {
				t.Fatalf("game %d: unplaying %x did not restore the board", g, sm.TM)
			}
		}
		pos.SetOverrideKWG(nil)
		k.Close()
	}
}
