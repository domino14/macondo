package magpie

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
)

var DefaultConfig = config.DefaultConfig()

func lexiconPath(t *testing.T, lex, ext string) string {
	p := filepath.Join(DefaultConfig.GetString(config.ConfigDataPath), "lexica", "gaddag", lex+ext)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("lexicon file %s not available: %v", p, err)
	}
	return p
}

func englishData(t *testing.T, lex string) *Data {
	is := is.New(t)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.WGLConfig())
	is.NoErr(err)
	d, err := NewData(DataOptions{
		KWGPath:    lexiconPath(t, lex, ".kwg"),
		KLVPath:    lexiconPath(t, lex, ".klv2"),
		Dist:       ld,
		Layout:     board.CrosswordGameBoard,
		Variant:    VariantClassic,
		BingoBonus: 50,
	})
	is.NoErr(err)
	return d
}

// stateFromBoard puts rack on turn and every other unseen tile in the bag.
func stateFromBoard(bd *board.GameBoard, ld *tilemapping.LetterDistribution,
	rack tilemapping.MachineWord) State {
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
	return State{Squares: squares, Racks: [2]tilemapping.MachineWord{rack, nil}, Bag: bag}
}

func TestGenerateFullRack(t *testing.T) {
	is := is.New(t)
	d := englishData(t, "NWL20")
	defer d.Close()
	ld := d.dist
	alph := ld.TileMapping()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	bd.SetToGame(alph, board.VsMatt)
	rack, err := tilemapping.ToMachineWord("AABDELT", alph)
	is.NoErr(err)

	p := NewPosition(d, 10000, 1)
	defer p.Close()
	is.NoErr(p.Set(stateFromBoard(bd, ld, rack)))

	moves := p.Generate(RecordAll, SortScore, 0)
	scoring, exchanges, passes := 0, 0, 0
	for i := range moves {
		switch moves[i].Type {
		case MovePlacement:
			scoring++
		case MoveExchange:
			exchanges++
		case MovePass:
			passes++
		}
	}
	// The same counts Macondo's own generator gives (movegen
	// TestGenAllMovesFullRack).
	is.Equal(scoring, 667)
	is.Equal(exchanges, 95)
	t.Logf("passes: %d", passes)

	highestScores := []int32{38, 36, 36, 34, 34, 33, 30, 30, 30, 28}
	for i, s := range highestScores {
		is.Equal(moves[i].Score, s*EquityResolution)
	}

	best := p.Generate(RecordBest, SortEquity, 0)
	is.Equal(len(best), 1)
	t.Logf("best: type %d at %d,%d vertical=%d score %d equity %.3f",
		best[0].Type, best[0].Row, best[0].Col, best[0].Vertical,
		best[0].Score/EquityResolution, float64(best[0].Equity)/EquityResolution)
}
