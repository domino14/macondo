package magpie_test

import (
	"os"
	"path/filepath"
	"runtime"
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

// benchPosition is a mid-game position: a Macondo board and a MAGPIE
// position holding the same tiles, with the rack on turn.
type benchPosition struct {
	bd   *board.GameBoard
	rack *tilemapping.Rack
	pos  *magpie.Position
}

// benchPositions plays greedy (top-score) games and keeps every position
// with 30 to 60 tiles on the board.
func benchPositions(b *testing.B, wmp bool) ([]benchPosition, *movegen.GordonGenerator, func()) {
	return benchPositionsRIT(b, wmp, "")
}

func benchPositionsRIT(b *testing.B, wmp bool, rit string) ([]benchPosition, *movegen.GordonGenerator, func()) {
	dataPath := defaultConfig.GetString(config.ConfigDataPath)
	kwgPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.kwg")
	klvPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.klv2")
	if _, err := os.Stat(klvPath); err != nil {
		b.Skip("NWL23 not available")
	}
	gd, _ := kwg.GetKWG(defaultConfig.WGLConfig(), "NWL23")
	ld, _ := tilemapping.GetDistribution(defaultConfig.WGLConfig(), "english")
	d, err := magpie.NewData(magpie.DataOptions{KWGPath: kwgPath, KLVPath: klvPath,
		BuildWMP: wmp, RITPath: rit, Dist: ld, Layout: board.CrosswordGameBoard,
		Variant: magpie.VariantClassic, BingoBonus: 50})
	if err != nil {
		b.Fatal(err)
	}
	var out []benchPosition
	var gen *movegen.GordonGenerator
	for g := 0; len(out) < 200; g++ {
		bd := board.MakeBoard(board.CrosswordGameBoard)
		csgen := &cross_set.GaddagCrossSetGenerator{Dist: ld, Gaddag: gd}
		csgen.GenerateAll(bd)
		bd.UpdateAllAnchors()
		gen = movegen.NewGordonGenerator(gd, bd, ld)
		bag := ld.MakeBag()
		bag.Shuffle()
		racks := [2][]tilemapping.MachineLetter{make([]tilemapping.MachineLetter, 7), make([]tilemapping.MachineLetter, 7)}
		bag.Draw(7, racks[0])
		bag.Draw(7, racks[1])
		onturn, tiles := 0, 0
		for tiles < 60 && bag.TilesRemaining() > 7 {
			rack := tilemapping.NewRack(ld.TileMapping())
			rack.Set(racks[onturn])
			if tiles >= 30 {
				pos := magpie.NewPosition(d, 100000, 1)
				squares := make([]tilemapping.MachineLetter, 225)
				for r := 0; r < 15; r++ {
					for c := 0; c < 15; c++ {
						squares[r*15+c] = bd.GetLetter(r, c)
					}
				}
				var mr [2]tilemapping.MachineWord
				mr[onturn] = racks[onturn]
				mr[1-onturn] = racks[1-onturn]
				if err := pos.Set(magpie.State{Squares: squares, Racks: mr, Bag: bag.Peek(), OnTurn: onturn}); err != nil {
					b.Fatal(err)
				}
				out = append(out, benchPosition{bd: bd.Copy(), rack: rack, pos: pos})
			}
			plays := gen.GenAll(rack, false)
			var best *move.Move
			for _, m := range plays {
				if m.Action() == move.MoveTypePlay && (best == nil || m.Score() > best.Score()) {
					best = m
				}
			}
			if best == nil {
				break
			}
			bd.PlayMove(best)
			csgen.UpdateForMove(bd, best)
			tiles += best.TilesPlayed()
			leave, _ := tilemapping.Leave(racks[onturn], best.Tiles(), false)
			drawn := make([]tilemapping.MachineLetter, 7-len(leave))
			bag.Draw(len(drawn), drawn)
			racks[onturn] = append(leave, drawn...)
			onturn = 1 - onturn
		}
	}
	return out, gen, func() {
		for _, p := range out {
			p.pos.Close()
		}
		d.Close()
	}
}

func BenchmarkGenAllGordon(b *testing.B) {
	ps, _, done := benchPositions(b, false)
	defer done()
	gd, _ := kwg.GetKWG(defaultConfig.WGLConfig(), "NWL23")
	ld, _ := tilemapping.GetDistribution(defaultConfig.WGLConfig(), "english")
	gens := make([]*movegen.GordonGenerator, len(ps))
	for i, p := range ps {
		gens[i] = movegen.NewGordonGenerator(gd, p.bd, ld)
		gens[i].SetSortingParameter(movegen.SortByNone)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := i % len(ps)
		gens[j].GenAll(ps[j].rack, false)
	}
}

func BenchmarkGenAllMagpie(b *testing.B) {
	ps, _, done := benchPositions(b, false)
	defer done()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ps[i%len(ps)].pos.Generate(magpie.RecordAll, magpie.SortScore, 0)
	}
}

func BenchmarkBestMagpie(b *testing.B) {
	ps, _, done := benchPositions(b, false)
	defer done()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ps[i%len(ps)].pos.Generate(magpie.RecordBest, magpie.SortEquity, 0)
	}
}

func BenchmarkBestMagpieWMP(b *testing.B) {
	ps, _, done := benchPositions(b, true)
	defer done()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ps[i%len(ps)].pos.Generate(magpie.RecordBest, magpie.SortEquity, 0)
	}
}

// BenchmarkCgoCall is the cost of one trivial call into the shim.
func BenchmarkCgoCall(b *testing.B) {
	ps, _, done := benchPositions(b, false)
	defer done()
	p := ps[0].pos
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.SetOnTurn(0)
	}
}

func BenchmarkBestMagpieSamePosition(b *testing.B) {
	ps, _, done := benchPositions(b, false)
	defer done()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ps[0].pos.Generate(magpie.RecordBest, magpie.SortEquity, 0)
	}
}

// BenchmarkBestMagpieRIT needs MAGPIE_RIT: an NWL23 rack info table.
func BenchmarkBestMagpieRIT(b *testing.B) {
	rit := os.Getenv("MAGPIE_RIT")
	if rit == "" {
		b.Skip("set MAGPIE_RIT")
	}
	ps, _, done := benchPositionsRIT(b, true, rit)
	defer done()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ps[i%len(ps)].pos.Generate(magpie.RecordBest, magpie.SortEquity, 0)
	}
}

// playFullGame plays one game of best-equity moves on pos, with the bag held
// in Go as Macondo will hold it. It returns the number of turns.
func playFullGame(b *testing.B, pos *magpie.Position, ld *tilemapping.LetterDistribution) int {
	bag := ld.MakeBag()
	bag.Shuffle()
	var racks [2][]tilemapping.MachineLetter
	for i := range racks {
		racks[i] = make([]tilemapping.MachineLetter, 7)
		bag.Draw(7, racks[i])
	}
	if err := pos.Set(magpie.State{Squares: make([]tilemapping.MachineLetter, 225),
		Racks: [2]tilemapping.MachineWord{racks[0], racks[1]}, Bag: bag.Peek()}); err != nil {
		b.Fatal(err)
	}
	drawn := make([]tilemapping.MachineLetter, 7)
	onturn, scoreless, turns := 0, 0, 0
	for len(racks[0]) > 0 && len(racks[1]) > 0 && scoreless < 6 {
		turns++
		m := &pos.Generate(magpie.RecordBest, magpie.SortEquity, 0)[0]
		var n int
		switch m.Type {
		case magpie.MovePlacement:
			racks[onturn], _ = tilemapping.Leave(racks[onturn], m.Word(), false)
			n = min(int(m.TilesPlayed), bag.TilesRemaining())
			bag.Draw(n, drawn)
			scoreless = 0
		case magpie.MoveExchange:
			racks[onturn], _ = tilemapping.Leave(racks[onturn], m.Word(), true)
			n = int(m.TilesPlayed)
			bag.Exchange(m.Word(), drawn[:n])
			scoreless++
		default:
			scoreless++
		}
		if err := pos.Play(m, drawn[:n]); err != nil {
			b.Fatal(err)
		}
		racks[onturn] = append(racks[onturn], drawn[:n]...)
		onturn = 1 - onturn
	}
	return turns
}

func benchPlayFull(b *testing.B, wmp bool, rit string) {
	dataPath := defaultConfig.GetString(config.ConfigDataPath)
	klvPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.klv2")
	if _, err := os.Stat(klvPath); err != nil {
		b.Skip("NWL23 not available")
	}
	ld, _ := tilemapping.GetDistribution(defaultConfig.WGLConfig(), "english")
	d, err := magpie.NewData(magpie.DataOptions{
		KWGPath: filepath.Join(dataPath, "lexica", "gaddag", "NWL23.kwg"), KLVPath: klvPath,
		BuildWMP: wmp, RITPath: rit, Dist: ld, Layout: board.CrosswordGameBoard,
		Variant: magpie.VariantClassic, BingoBonus: 50})
	if err != nil {
		b.Fatal(err)
	}
	defer d.Close()
	pos := magpie.NewPosition(d, 1, 1)
	defer pos.Close()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	turns := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		turns += playFullGame(b, pos, ld)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(turns), "ns/turn")
}

// A full game of best-equity moves: compare with automatic's BenchmarkPlayFull.
func BenchmarkPlayFullMagpie(b *testing.B)    { benchPlayFull(b, false, "") }
func BenchmarkPlayFullMagpieWMP(b *testing.B) { benchPlayFull(b, true, "") }
func BenchmarkPlayFullMagpieRIT(b *testing.B) {
	rit := os.Getenv("MAGPIE_RIT")
	if rit == "" {
		b.Skip("set MAGPIE_RIT")
	}
	benchPlayFull(b, true, rit)
}
