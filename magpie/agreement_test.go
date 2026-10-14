package magpie_test

// Checks MAGPIE against Macondo's own move generator and cross-set code over
// random games: at every position both must generate the same plays with the
// same scores, and after every move MAGPIE's incrementally updated board and
// cross-sets must match Macondo's.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"math/rand/v2"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

var defaultConfig = config.DefaultConfig()

type agreementCase struct {
	lexicon string
	leaves  string
	dist    string
	layout  []string
}

func playKey(m *move.Move) string {
	r, c, v := m.CoordsAndVertical()
	return fmt.Sprintf("%d,%d,%v,%v", r, c, v, []tilemapping.MachineLetter(m.Tiles()))
}

func magpieKey(m *magpie.Move) string {
	return fmt.Sprintf("%d,%d,%v,%v", m.Row, m.Col, m.Vertical == 1, []tilemapping.MachineLetter(m.Word()))
}

func exchangeKey(tiles []tilemapping.MachineLetter) string {
	s := append([]tilemapping.MachineLetter(nil), tiles...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return fmt.Sprint(s)
}

func toMagpieMove(m *move.Move) magpie.Move {
	var mm magpie.Move
	switch m.Action() {
	case move.MoveTypePlay:
		mm.Type = magpie.MovePlacement
		r, c, v := m.CoordsAndVertical()
		mm.Row, mm.Col = uint8(r), uint8(c)
		if v {
			mm.Vertical = 1
		}
		mm.Score = int32(m.Score() * magpie.EquityResolution)
	case move.MoveTypeExchange:
		mm.Type = magpie.MoveExchange
	default:
		mm.Type = magpie.MovePass
	}
	mm.TilesPlayed = uint8(m.TilesPlayed())
	mm.TilesLength = uint8(len(m.Tiles()))
	copy(mm.Tiles[:], m.Tiles())
	return mm
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

func runAgreement(t *testing.T, tc agreementCase, games int, seed uint64) {
	dataPath := defaultConfig.GetString(config.ConfigDataPath)
	kwgPath := filepath.Join(dataPath, "lexica", "gaddag", tc.lexicon+".kwg")
	if _, err := os.Stat(kwgPath); err != nil {
		t.Skipf("lexicon %s not available", kwgPath)
	}
	gd, err := kwg.GetKWG(defaultConfig.WGLConfig(), tc.lexicon)
	if err != nil {
		t.Fatal(err)
	}
	ld, err := tilemapping.GetDistribution(defaultConfig.WGLConfig(), tc.dist)
	if err != nil {
		t.Fatal(err)
	}
	klvPath := filepath.Join(dataPath, "lexica", "gaddag", tc.leaves+".klv2")
	if _, err := os.Stat(klvPath); err != nil {
		t.Skipf("leaves %s not available", klvPath)
	}
	d, err := magpie.NewData(magpie.DataOptions{KWGPath: kwgPath, KLVPath: klvPath, Dist: ld,
		Layout: tc.layout, Variant: magpie.VariantClassic, BingoBonus: 50})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	pos := magpie.NewPosition(d, 200000, seed)
	defer pos.Close()
	pos.SetBackupMode(magpie.BackupStack)
	mb, before, after := d.NewBoard(), d.NewBoard(), d.NewBoard()
	dim := len(tc.layout)
	rng := rand.New(rand.NewPCG(seed, 0))

	// Cross-sets compare on the distribution's letters only: an unrestricted
	// square's set has every bit up to each program's alphabet limit.
	letters := uint64(1)<<len(ld.Distribution()) - 2
	positions, plays := 0, 0
	for g := 0; g < games; g++ {
		bd := board.MakeBoard(tc.layout)
		csgen := &cross_set.GaddagCrossSetGenerator{Dist: ld, Gaddag: gd}
		csgen.GenerateAll(bd)
		bd.UpdateAllAnchors()
		gen := movegen.NewGordonGenerator(gd, bd, ld)
		gen.SetSortingParameter(movegen.SortByNone)

		bag := ld.MakeBag()
		bag.Shuffle()
		var racks [2][]tilemapping.MachineLetter
		for i := range racks {
			racks[i] = make([]tilemapping.MachineLetter, 7)
			if err := bag.Draw(7, racks[i]); err != nil {
				t.Fatal(err)
			}
		}
		if err := pos.Set(magpie.State{
			Squares: make([]tilemapping.MachineLetter, dim*dim),
			Racks:   [2]tilemapping.MachineWord{racks[0], racks[1]},
			Bag:     bag.Peek(),
		}); err != nil {
			t.Fatal(err)
		}
		onturn, scoreless := 0, 0
		for turn := 0; len(racks[0]) > 0 && len(racks[1]) > 0 && scoreless < 6; turn++ {
			positions++
			rack := tilemapping.NewRack(gd.GetAlphabet())
			rack.Set(racks[onturn])
			canExchange := bag.TilesRemaining() >= 7
			gordon := gen.GenAll(rack, canExchange)
			mmoves := pos.Generate(magpie.RecordAll, magpie.SortScore, 0)

			want := map[string]int{}
			wantEx := map[string]bool{}
			var placements []*move.Move
			for _, m := range gordon {
				switch m.Action() {
				case move.MoveTypePlay:
					want[playKey(m)] = m.Score()
					placements = append(placements, m)
				case move.MoveTypeExchange:
					wantEx[exchangeKey(m.Tiles())] = true
				}
			}
			got := map[string]int{}
			gotEx := map[string]bool{}
			for i := range mmoves {
				switch mmoves[i].Type {
				case magpie.MovePlacement:
					got[magpieKey(&mmoves[i])] = int(mmoves[i].Score / magpie.EquityResolution)
				case magpie.MoveExchange:
					gotEx[exchangeKey(mmoves[i].Word())] = true
				}
			}
			plays += len(want)
			for k, s := range want {
				gs, ok := got[k]
				if !ok {
					t.Fatalf("game %d turn %d: MAGPIE is missing %s (score %d)", g, turn, k, s)
				}
				if gs != s {
					t.Fatalf("game %d turn %d: %s scores %d in MAGPIE, %d in Macondo", g, turn, k, gs, s)
				}
			}
			for k := range got {
				if _, ok := want[k]; !ok {
					t.Fatalf("game %d turn %d: MAGPIE has extra play %s", g, turn, k)
				}
			}
			if len(wantEx) != len(gotEx) {
				t.Fatalf("game %d turn %d: %d exchanges in Macondo, %d in MAGPIE", g, turn, len(wantEx), len(gotEx))
			}
			for k := range wantEx {
				if !gotEx[k] {
					t.Fatalf("game %d turn %d: MAGPIE is missing exchange %s", g, turn, k)
				}
			}

			// Pick a play among the 8 best by score, an exchange now and then,
			// or pass.
			var chosen *move.Move
			sort.Slice(placements, func(i, j int) bool { return placements[i].Score() > placements[j].Score() })
			switch {
			case len(placements) > 0 && rng.IntN(20) != 0:
				chosen = placements[rng.IntN(min(8, len(placements)))]
			case canExchange:
				n := 1 + rng.IntN(7)
				chosen = move.NewExchangeMove(append([]tilemapping.MachineLetter(nil), racks[onturn][:n]...), nil, ld.TileMapping())
			default:
				chosen = move.NewPassMove(nil, ld.TileMapping())
			}

			// The tiles left on the rack, then the draw.
			leave := append([]tilemapping.MachineLetter(nil), racks[onturn]...)
			used := chosen.Tiles()
			if chosen.Action() == move.MoveTypePass {
				used = nil
			}
			for _, tile := range used {
				if chosen.Action() == move.MoveTypePlay && tile == 0 {
					continue
				}
				if tile.IsBlanked() {
					tile = 0
				}
				for i, l := range leave {
					if l == tile {
						leave = append(leave[:i], leave[i+1:]...)
						break
					}
				}
			}
			ndraw := min(len(racks[onturn])-len(leave), bag.TilesRemaining())
			drawn := make([]tilemapping.MachineLetter, ndraw)
			switch chosen.Action() {
			case move.MoveTypePlay:
				if err := bag.Draw(ndraw, drawn); err != nil {
					t.Fatal(err)
				}
				bd.PlayMove(chosen)
				csgen.UpdateForMove(bd, chosen)
				scoreless = 0
			case move.MoveTypeExchange:
				if err := bag.Exchange(chosen.Tiles(), drawn); err != nil {
					t.Fatal(err)
				}
				scoreless++
			default:
				drawn = drawn[:0]
				scoreless++
			}
			mm := toMagpieMove(chosen)
			pos.ReadBoard(before)
			_, _, bagBefore := pos.RackCounts()
			if err := pos.Play(&mm, drawn); err != nil {
				t.Fatalf("game %d turn %d: %v", g, turn, err)
			}
			if rng.IntN(4) == 0 {
				// Unplay must restore the position exactly; then replay.
				pos.ReadBoard(after)
				if err := pos.Unplay(); err != nil {
					t.Fatal(err)
				}
				pos.ReadBoard(mb)
				_, _, bagNow := pos.RackCounts()
				if !boardsEqual(mb, before) || bagNow != bagBefore {
					t.Fatalf("game %d turn %d: unplay did not restore the position", g, turn)
				}
				if err := pos.Play(&mm, drawn); err != nil {
					t.Fatal(err)
				}
				pos.ReadBoard(mb)
				if !boardsEqual(mb, after) {
					t.Fatalf("game %d turn %d: replay differs from the first play", g, turn)
				}
			}
			racks[onturn] = append(leave, drawn...)
			onturn = 1 - onturn

			// MAGPIE's bag holds the same tiles as Macondo's.
			_, _, nbag := pos.RackCounts()
			if nbag != bag.TilesRemaining() {
				t.Fatalf("game %d turn %d: MAGPIE bag has %d tiles, Macondo's %d", g, turn, nbag, bag.TilesRemaining())
			}
			pos.ReadBoard(mb)
			for r := 0; r < dim; r++ {
				for c := 0; c < dim; c++ {
					i := r*dim + c
					if mb.Letters[i] != bd.GetLetter(r, c) {
						t.Fatalf("game %d turn %d: square %d,%d is %d in MAGPIE, %d in Macondo", g, turn, r, c, mb.Letters[i], bd.GetLetter(r, c))
					}
					if bd.GetLetter(r, c) != 0 {
						continue
					}
					for dir := 0; dir < 2; dir++ {
						// MAGPIE indexes cross-sets by the direction of the play
						// they constrain; Macondo by the direction of the
						// cross-word, so an across play reads its vertical set.
						md := board.VerticalDirection
						if dir == 1 {
							md = board.HorizontalDirection
						}
						want := uint64(bd.GetCrossSet(r, c, md))
						got := mb.CrossSets[dir*dim*dim+i]
						if want&letters != got&letters {
							t.Fatalf("game %d turn %d: cross-set %d,%d dir %d is %s in MAGPIE, %s in Macondo",
								g, turn, r, c, dir, strconv.FormatUint(got, 2), strconv.FormatUint(want, 2))
						}
						ws := bd.GetCrossScore(r, c, md)
						gs := int(mb.CrossScores[dir*dim*dim+i] / magpie.EquityResolution)
						if ws != gs {
							t.Fatalf("game %d turn %d: cross-score %d,%d dir %d is %d in MAGPIE, %d in Macondo", g, turn, r, c, dir, gs, ws)
						}
					}
				}
			}
		}
	}
	t.Logf("%s %dx%d: %d games, %d positions, %d plays agree", tc.lexicon, dim, dim, games, positions, plays)
}

func TestAgreementNWL23(t *testing.T) {
	games := 20
	if testing.Short() {
		games = 3
	}
	if s := os.Getenv("MAGPIE_AGREEMENT_GAMES"); s != "" {
		games, _ = strconv.Atoi(s)
	}
	runAgreement(t, agreementCase{"NWL23", "NWL23", "english", board.CrosswordGameBoard}, games, 42)
}

func TestAgreementSuper(t *testing.T) {
	games := 5
	if s := os.Getenv("MAGPIE_AGREEMENT_GAMES"); s != "" {
		games, _ = strconv.Atoi(s)
	}
	runAgreement(t, agreementCase{"NWL23", "super-NWL23", "english_super", board.SuperCrosswordGameBoard}, games, 7)
}
