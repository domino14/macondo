package crosscheck

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

var gamesFlag = flag.Int("crosscheck.games", 3,
	"random games per lexicon in the board cross-check; every position of every game is compared")

// A lexicon and board to cross-check.
type boardCase struct {
	lexicon, dist, layout string
}

var boardCases = []boardCase{
	{"NWL20", "english", board.CrosswordGameLayout},
	{"CSW24", "english", board.CrosswordGameLayout},
	{"CSW24", "english_super", board.SuperCrosswordGameLayout},
	{"OSPS49", "polish", board.CrosswordGameLayout},
	{"FRA20", "french", board.CrosswordGameLayout},
	{"RD29", "german", board.CrosswordGameLayout},
	{"FILE2017", "spanish", board.CrosswordGameLayout},
}

// TestPositionCrossCheck plays random games with the Go engine and, after
// every move, checks that the Zig engine computes the same cross-sets,
// cross-scores, anchors and extension sets for the position, the same
// moves with the same scores, and the same leave values for random leaves.
// It does the same on boards of random tiles with random racks.
func TestPositionCrossCheck(t *testing.T) {
	e := startEngine(t)
	defer e.Quit()
	cfg := config.DefaultConfig()
	dataPath := cfg.GetString(config.ConfigDataPath)
	if dataPath == "" {
		t.Skip("no data path; set MACONDO_DATA_PATH")
	}
	if err := e.Send("setoption name data-path value " + dataPath); err != nil {
		t.Fatal(err)
	}
	if got, err := e.Command("isready", "readyok"); err != nil || len(got) != 1 {
		t.Fatalf("isready: %q %v", got, err)
	}

	for _, bc := range boardCases {
		t.Run(bc.lexicon+"-"+bc.layout, func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dataPath, "lexica", "gaddag", bc.lexicon+".kwg")); err != nil {
				t.Skipf("no %s word graph", bc.lexicon)
			}
			positions := 0
			for gi := 0; gi < *gamesFlag; gi++ {
				positions += crossCheckGame(t, e, cfg, dataPath, bc, uint64(gi))
				if t.Failed() {
					return
				}
			}
			// Random tiles: phonies and shapes no game reaches.
			for i := 0; i < 50**gamesFlag; i++ {
				cgpStr := randomTilesCGP(t, cfg, bc, uint64(i))
				comparePosition(t, e, cfg, cgpStr)
				compareMoves(t, e, cfg, cgpStr)
				positions++
				if t.Failed() {
					return
				}
			}
			t.Logf("%d positions match", positions)
		})
	}
}

// randomTilesCGP lays a few runs of random tiles from the bag, words or
// not, on an empty board.
func randomTilesCGP(t *testing.T, cfg *config.Config, bc boardCase, seed uint64) string {
	t.Helper()
	ld, err := tilemapping.GetDistribution(cfg.WGLConfig(), bc.dist)
	if err != nil {
		t.Fatal(err)
	}
	rows := board.CrosswordGameBoard
	if bc.layout == board.SuperCrosswordGameLayout {
		rows = board.SuperCrosswordGameBoard
	}
	b := board.MakeBoard(rows)
	rng := rand.New(rand.NewPCG(seed, 0xb0a4d))

	var bag []tilemapping.MachineLetter
	for ml, n := range ld.Distribution() {
		for range n {
			bag = append(bag, tilemapping.MachineLetter(ml))
		}
	}
	rng.Shuffle(len(bag), func(i, j int) { bag[i], bag[j] = bag[j], bag[i] })
	numLetters := len(ld.Distribution())

	dim := b.Dim()
	for range 1 + rng.IntN(12) {
		vertical := rng.IntN(2) == 0
		row, col := rng.IntN(dim), rng.IntN(dim)
		for range 1 + rng.IntN(7) {
			if row >= dim || col >= dim || len(bag) == 0 {
				break
			}
			if b.GetLetter(row, col) == 0 {
				ml := bag[len(bag)-1]
				bag = bag[:len(bag)-1]
				if ml == 0 {
					ml = tilemapping.MachineLetter(1 + rng.IntN(numLetters-1)).Blank()
				}
				b.SetLetter(row, col, ml)
			}
			if vertical {
				row++
			} else {
				col++
			}
		}
	}
	// Racks from what is left: usually full, sometimes short (end of game).
	var racks [2]string
	for i := range racks {
		n := min(len(bag), 7)
		if rng.IntN(4) == 0 {
			n = min(len(bag), rng.IntN(8))
		}
		racks[i] = tilemapping.MachineWord(bag[len(bag)-n:]).UserVisible(ld.TileMapping())
		bag = bag[:len(bag)-n]
	}
	return fmt.Sprintf("%s %s/%s 0/0 0 lex %s; ld %s; bdn %s;",
		b.ToFEN(ld.TileMapping()), racks[0], racks[1], bc.lexicon, bc.dist, bc.layout)
}

// compareMoves generates every move for the player on turn in both engines
// and compares the lists (as multisets: the order is not compared).
func compareMoves(t *testing.T, e *Engine, cfg *config.Config, cgpStr string) {
	t.Helper()
	parsed, err := cgp.ParseCGP(cfg, cgpStr)
	if err != nil {
		t.Fatalf("Go can't parse %q: %v", cgpStr, err)
	}
	g := parsed.Game
	g.RecalculateBoard()
	ld := g.Bag().LetterDistribution()
	gd, err := kwg.GetKWG(cfg.WGLConfig(), g.LexiconName(), kwg.WithDistribution(ld.Name))
	if err != nil {
		t.Fatal(err)
	}
	gen := movegen.NewGordonGenerator(gd, g.Board(), ld)
	unseen := int(g.RackFor(1).NumTiles()) + g.Bag().TilesRemaining()
	exchAllowed := unseen-game.RackTileLimit >= g.ExchangeLimit()
	gen.SetMaxCanExchange(game.MaxCanExchange(unseen-game.RackTileLimit, g.ExchangeLimit()))
	plays := gen.GenAll(g.RackFor(0), exchAllowed)

	alph := ld.TileMapping()
	want := make([]string, 0, len(plays))
	for _, m := range plays {
		want = append(want, ucgiMove(m, alph)+" score "+strconv.Itoa(m.Score()))
	}

	lines, err := e.Command("gen", "genok")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]string, 0, len(lines))
	for _, l := range lines[:len(lines)-1] {
		got = append(got, strings.TrimPrefix(l, "move "))
	}
	if last := lines[len(lines)-1]; last != "genok count "+strconv.Itoa(len(got)) {
		t.Fatalf("bad gen trailer %q after %d moves", last, len(got))
	}
	missing, extra := multisetDiff(want, got)
	if len(missing) > 0 || len(extra) > 0 {
		t.Errorf("position %s: %d moves from Go, %d from Zig", cgpStr, len(want), len(got))
		for i, m := range missing {
			if i == 10 {
				break
			}
			t.Errorf("  only Go:  %s", m)
		}
		for i, m := range extra {
			if i == 10 {
				break
			}
			t.Errorf("  only Zig: %s", m)
		}
	}
}

// ucgiMove writes a Go move in UCGI notation (see zigcore/UCGI.md).
func ucgiMove(m *move.Move, alph *tilemapping.TileMapping) string {
	switch m.Action() {
	case move.MoveTypePlay:
		return m.BoardCoords() + "." + m.Tiles().UserVisiblePlayedTiles(alph)
	case move.MoveTypeExchange:
		return "exch." + m.Tiles().UserVisible(alph)
	case move.MoveTypePass:
		return "pass"
	}
	return "unhandled"
}

// multisetDiff returns the items of a not in b and of b not in a, counting
// repeats.
func multisetDiff(a, b []string) (onlyA, onlyB []string) {
	counts := map[string]int{}
	for _, s := range a {
		counts[s]++
	}
	for _, s := range b {
		counts[s]--
	}
	for s, n := range counts {
		for ; n > 0; n-- {
			onlyA = append(onlyA, s)
		}
		for ; n < 0; n++ {
			onlyB = append(onlyB, s)
		}
	}
	sort.Strings(onlyA)
	sort.Strings(onlyB)
	return onlyA, onlyB
}

// crossCheckGame plays one game and compares every position. It returns the
// number of positions compared.
func crossCheckGame(t *testing.T, e *Engine, cfg *config.Config, dataPath string, bc boardCase, seed uint64) int {
	t.Helper()
	rules, err := game.NewBasicGameRules(cfg, bc.lexicon, bc.layout, bc.dist, game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		t.Fatal(err)
	}
	players := []*pb.PlayerInfo{{Nickname: "a", RealName: "a"}, {Nickname: "b", RealName: "b"}}
	g, err := game.NewGame(rules, players)
	if err != nil {
		t.Fatal(err)
	}
	var bagSeed [32]byte
	copy(bagSeed[:], fmt.Sprintf("crosscheck-%s-%d", bc.lexicon, seed))
	g.SeedBag(bagSeed)
	g.StartGame()
	rng := rand.New(rand.NewPCG(seed, 0x5eed))

	gd, err := kwg.GetKWG(cfg.WGLConfig(), bc.lexicon, kwg.WithDistribution(bc.dist))
	if err != nil {
		t.Fatal(err)
	}
	klv := loadKLV(t, dataPath, bc.lexicon)
	gen := movegen.NewGordonGenerator(gd, g.Board(), rules.LetterDistribution())

	positions := 0
	for g.Playing() == pb.PlayState_PLAYING {
		cgpStr := g.ToCGP(false)
		if bc.layout != board.CrosswordGameLayout {
			cgpStr += " bdn " + bc.layout + ";"
		}
		comparePosition(t, e, cfg, cgpStr)
		compareMoves(t, e, cfg, cgpStr)
		if klv != nil {
			compareLeaves(t, e, klv, g.RackFor(g.PlayerOnTurn()), rules.LetterDistribution().TileMapping(), rng)
		}
		positions++
		if t.Failed() {
			return positions
		}

		// Half the time the top-scoring play, for realistic boards; the
		// rest a random one, for odd shapes.
		plays := gen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
		var tilePlays []*move.Move
		for _, m := range plays {
			if m.Action() == move.MoveTypePlay {
				tilePlays = append(tilePlays, m)
			}
		}
		if len(tilePlays) == 0 {
			break
		}
		pick := tilePlays[rng.IntN(len(tilePlays))]
		if rng.IntN(2) == 0 {
			for _, m := range tilePlays {
				if m.Score() > pick.Score() {
					pick = m
				}
			}
		}
		chosen := new(move.Move)
		chosen.CopyFrom(pick)
		if err := g.PlayMove(chosen, false, 0); err != nil {
			t.Fatal(err)
		}
	}
	return positions
}

// comparePosition sets the position up in both engines and compares every
// square.
func comparePosition(t *testing.T, e *Engine, cfg *config.Config, cgpStr string) {
	t.Helper()
	parsed, err := cgp.ParseCGP(cfg, cgpStr)
	if err != nil {
		t.Fatalf("Go can't parse its own CGP %q: %v", cgpStr, err)
	}
	parsed.RecalculateBoard()
	b := parsed.Board()

	if err := e.Send("position cgp " + cgpStr); err != nil {
		t.Fatal(err)
	}
	lines, err := e.Command("crosssets", "crosssetsok")
	if err != nil {
		t.Fatal(err)
	}
	dim := b.Dim()
	if want := 2*dim*dim + 1; len(lines) != want {
		t.Fatalf("%s\ngot %d lines, want %d; first: %q", cgpStr, len(lines), want, lines[0])
	}
	mismatches := 0
	for _, line := range lines[:len(lines)-1] {
		f := strings.Fields(line)
		if len(f) != 14 || f[0] != "cs" {
			t.Fatalf("bad line %q", line)
		}
		row, _ := strconv.Atoi(f[1])
		col, _ := strconv.Atoi(f[2])
		dir := board.HorizontalDirection
		if f[3] == "v" {
			dir = board.VerticalDirection
		}
		want := fmt.Sprintf("set %x score %d anchor %d lext %x rext %x",
			uint64(b.GetCrossSet(row, col, dir)), b.GetCrossScore(row, col, dir),
			boolInt(b.IsAnchor(row, col, dir)),
			b.GetLeftExtSet(row, col, dir), b.GetRightExtSet(row, col, dir))
		if got := strings.Join(f[4:], " "); got != want {
			if mismatches == 0 {
				t.Errorf("position %s", cgpStr)
			}
			t.Errorf("square %d,%d %s:\n  zig %s\n  go  %s", row, col, f[3], got, want)
			mismatches++
			if mismatches == 10 {
				t.FailNow()
			}
		}
	}
}

// compareLeaves checks the leave values of a few random subsets of rack.
func compareLeaves(t *testing.T, e *Engine, klv *equity.KLV, rack *tilemapping.Rack, tm *tilemapping.TileMapping, rng *rand.Rand) {
	t.Helper()
	tiles := rack.TilesOn()
	for range 4 {
		var leave tilemapping.MachineWord
		for _, ml := range tiles {
			if rng.IntN(2) == 0 {
				leave = append(leave, ml)
			}
		}
		text := leave.UserVisible(tm)
		got, err := e.Command("leave "+text, "leave")
		if err != nil {
			t.Fatal(err)
		}
		zig, err := strconv.ParseFloat(strings.TrimPrefix(got[len(got)-1], "leave "), 32)
		if err != nil {
			t.Fatalf("leave %q: %q", text, got)
		}
		want := klv.LeaveValue(append(tilemapping.MachineWord(nil), leave...))
		if float64(float32(zig)) != want {
			t.Errorf("leave %q: zig %v, go %v", text, float32(zig), want)
		}
	}
}

func loadKLV(t *testing.T, dataPath, lexicon string) *equity.KLV {
	t.Helper()
	f, err := os.Open(filepath.Join(dataPath, "lexica", "gaddag", lexicon+".klv2"))
	if err != nil {
		return nil
	}
	defer f.Close()
	klv, err := equity.ReadKLV(f)
	if err != nil {
		t.Fatal(err)
	}
	return klv
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
