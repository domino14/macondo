// wmp_shadow_test.go — ports the WMP-specific tests in MAGPIE's
// test/shadow_test.c (test_shadow_wmp_*). They drive the GordonGenerator
// shadow pass with a real CSW21 KWG and CSW21 WMP loaded directly from
// MAGPIE's data directory, then verify the per-anchor outputs match
// MAGPIE's expected counts and per-anchor fields.
//
// The tests are skipped automatically if the MAGPIE data files are not
// present at the expected absolute paths.
package movegen_test

import (
	"os"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/movegen"
	wmppkg "github.com/domino14/macondo/wmp"
)

// loadCSW21 loads the CSW21 KWG and WMP from the standard data directory.
// Skips the calling test if MACONDO_DATA_PATH is unset or the files are absent.
//
// Required layout under $MACONDO_DATA_PATH:
//
//	lexica/gaddag/CSW21.kwg
//	lexica/CSW21.wmp
//	letterdistributions/english  (for tilemapping.ProbableLetterDistribution)
//
// Example:
//
//	MACONDO_DATA_PATH=./data go test ./movegen/ -run TestWMPShadow
func loadCSW21(t *testing.T) (*kwg.KWG, *wmppkg.WMP, *tilemapping.LetterDistribution) {
	t.Helper()
	macondoDataPath := os.Getenv("MACONDO_DATA_PATH")
	if macondoDataPath == "" {
		t.Skip("MACONDO_DATA_PATH not set; skipping CSW21 shadow tests")
	}

	magpieCSW21KWG := macondoDataPath + "/lexica/gaddag/CSW21.kwg"
	magpieCSW21WMP := macondoDataPath + "/lexica/CSW21.wmp"

	if _, err := os.Stat(magpieCSW21KWG); err != nil {
		t.Skipf("CSW21.kwg not present at %s", magpieCSW21KWG)
	}
	if _, err := os.Stat(magpieCSW21WMP); err != nil {
		t.Skipf("CSW21.wmp not present at %s", magpieCSW21WMP)
	}

	DefaultConfig.Set("data-path", macondoDataPath)
	wglCfg := DefaultConfig.WGLConfig()

	gd, err := kwg.LoadKWG(wglCfg, magpieCSW21KWG)
	if err != nil {
		t.Fatalf("loading CSW21.kwg: %v", err)
	}

	w, err := wmppkg.LoadFromFile("CSW21", magpieCSW21WMP)
	if err != nil {
		t.Fatalf("loading CSW21.wmp: %v", err)
	}

	ld, err := tilemapping.NamedLetterDistribution(wglCfg, "english")
	if err != nil {
		t.Fatalf("loading english letter distribution: %v", err)
	}
	return gd, w, ld
}

// makeBoardWithLetters builds a fresh CrosswordGame board, places the
// given (row, col, "tile string") tuples on it, then runs cross-set
// generation and anchor updates so the result is ready for shadow.
//
// SetLetter does not bump the board's internal tilesPlayed counter,
// so we set it explicitly via SetTilesPlayed before UpdateAllAnchors;
// otherwise UpdateAllAnchors falls into the empty-board branch and
// only places the center anchor.
func makeBoardWithLetters(t *testing.T, gd *kwg.KWG, ld *tilemapping.LetterDistribution, tiles []boardTile) *board.GameBoard {
	t.Helper()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	alph := gd.GetAlphabet()
	for _, bt := range tiles {
		ml, err := alph.Val(bt.letter)
		if err != nil {
			t.Fatalf("alph.Val(%q): %v", bt.letter, err)
		}
		bd.SetLetter(bt.row, bt.col, ml)
	}
	bd.TestSetTilesPlayed(len(tiles))
	cross_set.GenAllCrossSets(bd, gd, ld)
	bd.UpdateAllAnchors()
	return bd
}

type boardTile struct {
	row, col int
	letter   string
}

// runWMPShadow constructs a GordonGenerator with the given KWG/board/LD
// + WMP, runs the shadow pass, and returns sorted anchors. Mirrors
// MAGPIE's load_and_shadow path with -wmp true and MOVE_SORT_SCORE.
//
// SubAnchor recording is turned on so the test can inspect per-(blocks,
// tiles) WMP anchor properties. Production sims (BenchmarkSimCSW24WMP)
// leave it off because each sub-anchor would trigger a redundant
// recursive_gen pass.
func runWMPShadow(t *testing.T, gd *kwg.KWG, bd *board.GameBoard,
	ld *tilemapping.LetterDistribution, w *wmppkg.WMP, rackStr string) []movegen.Anchor {
	t.Helper()
	gen := movegen.NewGordonGenerator(gd, bd, ld)
	gen.SetWMP(w)
	gen.SetWMPRecordSubAnchors(true)
	gen.SetSortingParameter(movegen.SortByScore)

	rack := tilemapping.RackFromString(rackStr, gd.GetAlphabet())
	// MAGPIE's tests run with the bag still mostly full; we just
	// need a positive count so shadowRecord uses the mid-game branch.
	const tilesInBag = 100
	const oppRackScore = 0
	return gen.RunShadowOnlyWMP(rack, tilesInBag, oppRackScore)
}

// assertSortedDescending asserts that anchors are in descending equity
// order, mirroring MAGPIE's load_and_shadow post-condition.
func assertSortedDescending(t *testing.T, anchors []movegen.Anchor) {
	t.Helper()
	for i := 1; i < len(anchors); i++ {
		if anchors[i].HighestPossibleEquity > anchors[i-1].HighestPossibleEquity {
			t.Errorf("anchors[%d].equity = %v > anchors[%d].equity = %v",
				i, anchors[i].HighestPossibleEquity, i-1, anchors[i-1].HighestPossibleEquity)
		}
	}
}

// TestShadowWMPNonplaythroughExistence ports MAGPIE's
// test_shadow_wmp_nonplaythrough_existence. Empty board with three
// different racks: MUZJIKS (full bingo + 6 sub-anchors), TRONGLE
// (no 7s), VVWWXYZ (no plays at all).
func TestShadowWMPNonplaythroughExistence(t *testing.T) {
	gd, w, ld := loadCSW21(t)

	// MUZJIKS — 6 anchors for lengths 2..7. Top is 7 tiles for 128.
	bd := makeBoardWithLetters(t, gd, ld, nil)
	anchors := runWMPShadow(t, gd, bd, ld, w, "MUZJIKS")
	assertSortedDescending(t, anchors)
	if len(anchors) != 6 {
		t.Errorf("MUZJIKS anchor count = %d, want 6", len(anchors))
	}

	expectedMUZJIKS := []struct {
		score       int
		tilesToPlay int
		blocks      int
	}{
		{128, 7, 0}, // MUZJIKS bingo
		{76, 6, 0},  // ZJKMIS-like 6
		{74, 5, 0},
		{52, 4, 0},
		{46, 3, 0},
		{36, 2, 0},
	}
	for i, want := range expectedMUZJIKS {
		if i >= len(anchors) {
			break
		}
		got := anchors[i]
		if int(got.HighestPossibleScore) != want.score {
			t.Errorf("MUZJIKS anchor[%d].score = %d, want %d", i, got.HighestPossibleScore, want.score)
		}
		if int(got.TilesToPlay) != want.tilesToPlay {
			t.Errorf("MUZJIKS anchor[%d].tilesToPlay = %d, want %d", i, got.TilesToPlay, want.tilesToPlay)
		}
		if int(got.PlaythroughBlocks) != want.blocks {
			t.Errorf("MUZJIKS anchor[%d].playthroughBlocks = %d, want %d", i, got.PlaythroughBlocks, want.blocks)
		}
		if got.Row != 7 || got.Col != 7 {
			t.Errorf("MUZJIKS anchor[%d] (row,col) = (%d,%d), want (7,7)", i, got.Row, got.Col)
		}
		if got.Dir != board.HorizontalDirection {
			t.Errorf("MUZJIKS anchor[%d].dir = %v, want horizontal", i, got.Dir)
		}
	}

	// TRONGLE — 5 anchors for lengths 2..6 (no 7-letter words).
	bd = makeBoardWithLetters(t, gd, ld, nil)
	anchors = runWMPShadow(t, gd, bd, ld, w, "TRONGLE")
	assertSortedDescending(t, anchors)
	if len(anchors) != 5 {
		t.Errorf("TRONGLE anchor count = %d, want 5", len(anchors))
	}

	expectedTRONGLE := []struct {
		score       int
		tilesToPlay int
	}{
		{18, 6},
		{16, 5},
		{10, 4},
		{8, 3},
		{6, 2},
	}
	for i, want := range expectedTRONGLE {
		if i >= len(anchors) {
			break
		}
		got := anchors[i]
		if int(got.HighestPossibleScore) != want.score {
			t.Errorf("TRONGLE anchor[%d].score = %d, want %d", i, got.HighestPossibleScore, want.score)
		}
		if int(got.TilesToPlay) != want.tilesToPlay {
			t.Errorf("TRONGLE anchor[%d].tilesToPlay = %d, want %d", i, got.TilesToPlay, want.tilesToPlay)
		}
		if got.PlaythroughBlocks != 0 {
			t.Errorf("TRONGLE anchor[%d].playthroughBlocks = %d, want 0", i, got.PlaythroughBlocks)
		}
	}

	// VVWWXYZ — no valid words at any length, so no anchors.
	bd = makeBoardWithLetters(t, gd, ld, nil)
	anchors = runWMPShadow(t, gd, bd, ld, w, "VVWWXYZ")
	if len(anchors) != 0 {
		t.Errorf("VVWWXYZ anchor count = %d, want 0", len(anchors))
	}
}

// TestShadowWMPPlaythroughBingoExistence ports MAGPIE's
// test_shadow_wmp_playthrough_bingo_existence. Board has a QI/QIS
// fragment; FRUITED and AOUNS?? produce specific top anchors.
//
// Board layout (CGP "15/15/15/15/15/15/15/6QI7/6I8/6S8/15/15/15/15/15"):
//
//	row 7: Q at col 6, I at col 7
//	row 8: I at col 6
//	row 9: S at col 6
func TestShadowWMPPlaythroughBingoExistence(t *testing.T) {
	is := is.New(t)
	gd, w, ld := loadCSW21(t)

	tiles := []boardTile{
		{7, 6, "Q"},
		{7, 7, "I"},
		{8, 6, "I"},
		{9, 6, "S"},
	}

	// FRUITED — top anchor is f9 UFTRIDE for 88, 7 tiles, vertical,
	// row 5 col 8, no playthrough blocks.
	bd := makeBoardWithLetters(t, gd, ld, tiles)
	anchors := runWMPShadow(t, gd, bd, ld, w, "FRUITED")
	assertSortedDescending(t, anchors)
	is.Equal(len(anchors), 46)

	top := anchors[0]
	if top.HighestPossibleScore != 88 {
		t.Errorf("FRUITED top score = %d, want 88", top.HighestPossibleScore)
	}
	if top.TilesToPlay != 7 {
		t.Errorf("FRUITED top tilesToPlay = %d, want 7", top.TilesToPlay)
	}
	if top.Row != 5 || top.Col != 8 {
		t.Errorf("FRUITED top (row,col) = (%d,%d), want (5,8)", top.Row, top.Col)
	}
	if top.Dir != board.VerticalDirection {
		t.Errorf("FRUITED top dir = %v, want vertical", top.Dir)
	}
	if top.PlaythroughBlocks != 0 {
		t.Errorf("FRUITED top playthroughBlocks = %d, want 0", top.PlaythroughBlocks)
	}

	// AOUNS?? — top anchor is 8g (QI)NghAOSU for 101, 7 tiles,
	// horizontal, row 7 col 7, 1 playthrough block.
	bd = makeBoardWithLetters(t, gd, ld, tiles)
	anchors = runWMPShadow(t, gd, bd, ld, w, "AOUNS??")
	assertSortedDescending(t, anchors)
	is.Equal(len(anchors), 56)

	top = anchors[0]
	if top.HighestPossibleScore != 101 {
		t.Errorf("AOUNS?? top score = %d, want 101", top.HighestPossibleScore)
	}
	if top.TilesToPlay != 7 {
		t.Errorf("AOUNS?? top tilesToPlay = %d, want 7", top.TilesToPlay)
	}
	if top.Row != 7 || top.Col != 7 {
		t.Errorf("AOUNS?? top (row,col) = (%d,%d), want (7,7)", top.Row, top.Col)
	}
	if top.Dir != board.HorizontalDirection {
		t.Errorf("AOUNS?? top dir = %v, want horizontal", top.Dir)
	}
	if top.PlaythroughBlocks != 1 {
		t.Errorf("AOUNS?? top playthroughBlocks = %d, want 1", top.PlaythroughBlocks)
	}
}

// TestShadowWMPOneTile ports MAGPIE's test_shadow_wmp_one_tile.
// Board has a QI horizontal at row 7 cols 6-7, plus a leftover I at
// row 8 col 6 (QI_QI_CGP). Rack "D" produces exactly two anchors: one
// horizontal, one vertical.
func TestShadowWMPOneTile(t *testing.T) {
	is := is.New(t)
	gd, w, ld := loadCSW21(t)

	tiles := []boardTile{
		{7, 6, "Q"},
		{7, 7, "I"},
		{8, 6, "I"},
	}
	bd := makeBoardWithLetters(t, gd, ld, tiles)
	anchors := runWMPShadow(t, gd, bd, ld, w, "D")
	assertSortedDescending(t, anchors)
	is.Equal(len(anchors), 2)

	// Anchor 0: max score 6 for 9F (I)D, horizontal at row 8 col 6,
	// 1 tile, 1 playthrough block, leftmost=5, rightmost=6.
	a0 := anchors[0]
	if a0.HighestPossibleScore != 6 {
		t.Errorf("anchor[0].score = %d, want 6", a0.HighestPossibleScore)
	}
	if a0.Row != 8 || a0.Col != 6 {
		t.Errorf("anchor[0] (row,col) = (%d,%d), want (8,6)", a0.Row, a0.Col)
	}
	if a0.Dir != board.HorizontalDirection {
		t.Errorf("anchor[0].dir = %v, want horizontal", a0.Dir)
	}
	if a0.TilesToPlay != 1 {
		t.Errorf("anchor[0].tilesToPlay = %d, want 1", a0.TilesToPlay)
	}
	if a0.PlaythroughBlocks != 1 {
		t.Errorf("anchor[0].playthroughBlocks = %d, want 1", a0.PlaythroughBlocks)
	}
	if a0.LeftmostStartCol != 5 {
		t.Errorf("anchor[0].leftmostStartCol = %d, want 5", a0.LeftmostStartCol)
	}
	if a0.RightmostStartCol != 6 {
		t.Errorf("anchor[0].rightmostStartCol = %d, want 6", a0.RightmostStartCol)
	}

	// Anchor 1: max score 3 for H7 D(I), vertical at row 7 col 7,
	// 1 tile, 1 playthrough block, leftmost=6, rightmost=6.
	a1 := anchors[1]
	if a1.HighestPossibleScore != 3 {
		t.Errorf("anchor[1].score = %d, want 3", a1.HighestPossibleScore)
	}
	if a1.Row != 7 || a1.Col != 7 {
		t.Errorf("anchor[1] (row,col) = (%d,%d), want (7,7)", a1.Row, a1.Col)
	}
	if a1.Dir != board.VerticalDirection {
		t.Errorf("anchor[1].dir = %v, want vertical", a1.Dir)
	}
	if a1.TilesToPlay != 1 {
		t.Errorf("anchor[1].tilesToPlay = %d, want 1", a1.TilesToPlay)
	}
	if a1.PlaythroughBlocks != 1 {
		t.Errorf("anchor[1].playthroughBlocks = %d, want 1", a1.PlaythroughBlocks)
	}
	if a1.LeftmostStartCol != 6 {
		t.Errorf("anchor[1].leftmostStartCol = %d, want 6", a1.LeftmostStartCol)
	}
	if a1.RightmostStartCol != 6 {
		t.Errorf("anchor[1].rightmostStartCol = %d, want 6", a1.RightmostStartCol)
	}
}

