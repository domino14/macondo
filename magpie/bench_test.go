package magpie_test

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/magpie"
)

// benchPosition is a mid-game position, with the rack on turn.
type benchPosition struct {
	pos *magpie.Position
}

func benchPositions(b *testing.B, wmp bool) ([]benchPosition, func()) {
	return benchPositionsRIT(b, wmp, "")
}

// benchPositionsRIT plays greedy (top-score) games and keeps every position
// with 30 to 60 tiles on the board, each in its own Position.
func benchPositionsRIT(b *testing.B, wmp bool, rit string) ([]benchPosition, func()) {
	dataPath := defaultConfig.GetString(config.ConfigDataPath)
	kwgPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.kwg")
	klvPath := filepath.Join(dataPath, "lexica", "gaddag", "NWL23.klv2")
	if _, err := os.Stat(klvPath); err != nil {
		b.Skip("NWL23 not available")
	}
	ld, _ := tilemapping.GetDistribution(defaultConfig.WGLConfig(), "english")
	d, err := magpie.NewData(magpie.DataOptions{KWGPath: kwgPath, KLVPath: klvPath,
		BuildWMP: wmp, RITPath: rit, Dist: ld, Layout: board.CrosswordGameBoard,
		Variant: magpie.VariantClassic, BingoBonus: 50})
	if err != nil {
		b.Fatal(err)
	}
	player := magpie.NewPosition(d, 1, 1)
	defer player.Close()
	mb := d.NewBoard()
	var out []benchPosition
	drawn := make([]tilemapping.MachineLetter, 7)
	for g := 0; len(out) < 200; g++ {
		bag := ld.MakeBag()
		bag.Shuffle()
		racks := [2][]tilemapping.MachineLetter{make([]tilemapping.MachineLetter, 7), make([]tilemapping.MachineLetter, 7)}
		bag.Draw(7, racks[0])
		bag.Draw(7, racks[1])
		if err := player.Set(magpie.State{Squares: make([]tilemapping.MachineLetter, 225),
			Racks: [2]tilemapping.MachineWord{racks[0], racks[1]}, Bag: bag.Peek()}); err != nil {
			b.Fatal(err)
		}
		onturn, tiles := 0, 0
		for tiles < 60 && bag.TilesRemaining() > 7 {
			if tiles >= 30 {
				player.ReadBoard(mb)
				pos := magpie.NewPosition(d, 100000, 1)
				var st magpie.State
				st.Squares = append([]tilemapping.MachineLetter(nil), mb.Letters...)
				st.Racks[0], st.Racks[1] = racks[0], racks[1]
				st.Bag, st.OnTurn = bag.Peek(), onturn
				if err := pos.Set(st); err != nil {
					b.Fatal(err)
				}
				out = append(out, benchPosition{pos: pos})
			}
			m := &player.Generate(magpie.RecordBest, magpie.SortScore, 0)[0]
			if m.Type != magpie.MovePlacement {
				break
			}
			leave, err := tilemapping.Leave(racks[onturn], m.Word(), true)
			if err != nil {
				b.Fatal(err)
			}
			n := min(int(m.TilesPlayed), bag.TilesRemaining())
			bag.Draw(n, drawn)
			if err := player.Play(m, drawn[:n]); err != nil {
				b.Fatal(err)
			}
			tiles += int(m.TilesPlayed)
			racks[onturn] = append(leave, drawn[:n]...)
			onturn = 1 - onturn
		}
	}
	return out, func() {
		for _, p := range out {
			p.pos.Close()
		}
		d.Close()
	}
}

func BenchmarkGenAllMagpie(b *testing.B) {
	ps, done := benchPositions(b, false)
	defer done()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ps[i%len(ps)].pos.Generate(magpie.RecordAll, magpie.SortScore, 0)
	}
}

func BenchmarkBestMagpie(b *testing.B) {
	ps, done := benchPositions(b, false)
	defer done()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ps[i%len(ps)].pos.Generate(magpie.RecordBest, magpie.SortEquity, 0)
	}
}

func BenchmarkBestMagpieWMP(b *testing.B) {
	ps, done := benchPositions(b, true)
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
	ps, done := benchPositions(b, false)
	defer done()
	p := ps[0].pos
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.SetOnTurn(0)
	}
}

func BenchmarkBestMagpieSamePosition(b *testing.B) {
	ps, done := benchPositions(b, false)
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
	ps, done := benchPositionsRIT(b, true, rit)
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
		var err error
		switch m.Type {
		case magpie.MovePlacement:
			racks[onturn], err = tilemapping.Leave(racks[onturn], m.Word(), true)
			n = min(int(m.TilesPlayed), bag.TilesRemaining())
			bag.Draw(n, drawn)
			scoreless = 0
		case magpie.MoveExchange:
			racks[onturn], err = tilemapping.Leave(racks[onturn], m.Word(), false)
			n = int(m.TilesPlayed)
			bag.Exchange(m.Word(), drawn[:n])
			scoreless++
		default:
			scoreless++
		}
		if err != nil {
			b.Fatal(err)
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
