package magpie_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/magpie"
)

// Macondo's .wmp files are MAGPIE's format: one loads, and generates what a
// word map MAGPIE builds itself does.
func TestMacondoWMPFile(t *testing.T) {
	dataPath := defaultConfig.GetString(config.ConfigDataPath)
	wmpPath := filepath.Join(dataPath, "lexica", "CSW24.wmp")
	klvPath := filepath.Join(dataPath, "lexica", "gaddag", "CSW24.klv2")
	for _, p := range []string{wmpPath, klvPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("%s not available", p)
		}
	}
	ld, err := tilemapping.GetDistribution(defaultConfig.WGLConfig(), "english")
	if err != nil {
		t.Fatal(err)
	}
	load := func(file string, build bool) *magpie.Data {
		d, err := magpie.NewData(magpie.DataOptions{
			KWGPath: filepath.Join(dataPath, "lexica", "gaddag", "CSW24.kwg"), KLVPath: klvPath,
			WMPPath: file, BuildWMP: build, Dist: ld, Layout: board.CrosswordGameBoard,
			Variant: magpie.VariantClassic, BingoBonus: 50})
		if err != nil {
			t.Fatal(err)
		}
		if !d.HasWMP() {
			t.Fatal("no word map loaded")
		}
		return d
	}
	fromFile, built := load(wmpPath, false), load("", true)
	defer fromFile.Close()
	defer built.Close()

	alph := ld.TileMapping()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	bd.SetToGame(alph, board.VsMatt)
	for i := 0; i < 20; i++ {
		// Seven unseen tiles, from a different place in the bag each time.
		st := stateFromBoardExt(bd, ld, nil)
		start := (i * 13) % (len(st.Bag) - 7)
		rack := append(tilemapping.MachineWord(nil), st.Bag[start:start+7]...)
		st.Bag = append(append(tilemapping.MachineWord(nil), st.Bag[:start]...), st.Bag[start+7:]...)
		st.Racks[0] = rack
		rs := rack.UserVisible(alph)
		var best [2]magpie.Move
		for i, d := range []*magpie.Data{fromFile, built} {
			p := magpie.NewPosition(d, 1, 1)
			if err := p.Set(st); err != nil {
				t.Fatal(err)
			}
			best[i] = p.Generate(magpie.RecordBest, magpie.SortEquity, 0)[0]
			p.Close()
		}
		if best[0] != best[1] {
			t.Fatalf("%s: file word map plays %+v, built one %+v", rs, best[0], best[1])
		}
	}
}

func stateFromBoardExt(bd *board.GameBoard, ld *tilemapping.LetterDistribution,
	rack tilemapping.MachineWord) magpie.State {
	dim := bd.Dim()
	unseen := append([]uint8(nil), ld.Distribution()...)
	squares := make([]tilemapping.MachineLetter, dim*dim)
	for r := 0; r < dim; r++ {
		for c := 0; c < dim; c++ {
			ml := bd.GetLetter(r, c)
			squares[r*dim+c] = ml
			if ml != 0 {
				unseen[ml.IntrinsicTileIdx()]--
			}
		}
	}
	for _, t := range rack {
		unseen[t]--
	}
	var bag tilemapping.MachineWord
	for ml, n := range unseen {
		for i := 0; i < int(n); i++ {
			bag = append(bag, tilemapping.MachineLetter(ml))
		}
	}
	return magpie.State{Squares: squares, Racks: [2]tilemapping.MachineWord{rack, nil}, Bag: bag}
}
