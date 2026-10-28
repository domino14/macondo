package magpie_test

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/magpie"
)

// At endgame positions (empty bag) the pruned word graph generates exactly
// what the full one does, and PlaySmall/UnplaySmall round-trip.
func TestSmallMoves(t *testing.T) {
	dataPath := defaultConfig.GetString(config.ConfigDataPath)
	kwgPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.kwg")
	klvPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.klv2")
	if _, err := os.Stat(klvPath); err != nil {
		t.Skip("NWL23 not available")
	}
	ld, _ := tilemapping.GetDistribution(defaultConfig.WGLConfig(), "english")
	d, err := magpie.NewData(magpie.DataOptions{KWGPath: kwgPath, KLVPath: klvPath, Dist: ld,
		Layout: board.CrosswordGameBoard, Variant: magpie.VariantClassic, BingoBonus: 50})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	pos := magpie.NewPosition(d, 5, 1)
	defer pos.Close()
	rng := rand.New(rand.NewPCG(5, 6))
	full := make([]magpie.SmallMove, 100000)
	pruned := make([]magpie.SmallMove, 100000)
	strip := make([]tilemapping.MachineLetter, 21)
	before, after := d.NewBoard(), d.NewBoard()
	drawn := make([]tilemapping.MachineLetter, 7)
	positions := 0
	for g := 0; positions < 60; g++ {
		bag := ld.MakeBag()
		bag.Shuffle()
		racks := [2][]tilemapping.MachineLetter{make([]tilemapping.MachineLetter, 7), make([]tilemapping.MachineLetter, 7)}
		bag.Draw(7, racks[0])
		bag.Draw(7, racks[1])
		if err := pos.Set(magpie.State{Squares: make([]tilemapping.MachineLetter, 225),
			Racks: [2]tilemapping.MachineWord{racks[0], racks[1]}, Bag: bag.Peek()}); err != nil {
			t.Fatal(err)
		}
		onturn := 0
		// Play one of the top few plays until the bag is empty.
		for bag.TilesRemaining() > 0 {
			moves := pos.GenerateN(magpie.RecordAll, magpie.SortScore, 0, 5)
			m := &moves[rng.IntN(len(moves))]
			if m.Type != magpie.MovePlacement {
				break
			}
			leave, err := tilemapping.Leave(racks[onturn], m.Word(), true)
			if err != nil {
				t.Fatal(err)
			}
			n := min(int(m.TilesPlayed), bag.TilesRemaining())
			bag.Draw(n, drawn)
			if err := pos.Play(m, drawn[:n]); err != nil {
				t.Fatal(err)
			}
			racks[onturn] = append(leave, drawn[:n]...)
			onturn = 1 - onturn
		}
		if bag.TilesRemaining() > 0 {
			continue
		}
		positions++

		got := pos.GenerateSmall(true, full)
		k := pos.PrunedKWG()
		pos.SetOverrideKWG(k)
		withPruned := pos.GenerateSmall(true, pruned)
		if len(withPruned) != len(got) {
			t.Fatalf("game %d: %d plays with the pruned graph, %d with the full", g, len(withPruned), len(got))
		}
		want := map[uint64]int16{}
		for _, sm := range got {
			want[sm.TM] = sm.Score
		}
		for _, sm := range withPruned {
			if s, ok := want[sm.TM]; !ok || s != sm.Score {
				t.Fatalf("game %d: pruned graph generates %x", g, sm.TM)
			}
		}

		// Play and unplay each move.
		pos.ReadBoard(before)
		for _, sm := range withPruned {
			n, err := pos.PlaySmall(sm.TM, int(sm.Score), strip)
			if err != nil {
				t.Fatal(err)
			}
			if sm.TM != 0 && n != int(sm.PlayLength) {
				t.Fatalf("game %d: strip of %d for a play %d long", g, n, sm.PlayLength)
			}
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

func boardsEqual(a, b *magpie.Board) bool {
	for i := range a.Letters {
		if a.Letters[i] != b.Letters[i] {
			return false
		}
	}
	for i := range a.CrossSets {
		if a.CrossSets[i] != b.CrossSets[i] || a.CrossScores[i] != b.CrossScores[i] ||
			a.Anchors[i] != b.Anchors[i] {
			return false
		}
	}
	return true
}
