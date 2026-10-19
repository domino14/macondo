package cross_set

import (
	"os"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/alphadawg"
	"github.com/domino14/macondo/board"
)

func alphaDawgFromLexicon(t *testing.T, lex string) *kwg.KWG {
	t.Helper()
	if os.Getenv("MACONDO_DATA_PATH") == "" {
		t.Skip("MACONDO_DATA_PATH not set")
	}
	k, err := alphadawg.Get(DefaultConfig.WGLConfig(), lex, "english")
	if err != nil {
		t.Skip(lex + ".kad not available: " + err.Error())
	}
	return k
}

// TestGenAlphaCrossSet mirrors magpie's test_alpha_cross_set. The point of
// most of these cases is that the cross set is identical no matter where the
// gap sits inside a block: only the multiset matters.
func TestGenAlphaCrossSet(t *testing.T) {
	is := is.New(t)

	kad := alphaDawgFromLexicon(t, "CSW21")
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.WGLConfig())
	is.NoErr(err)
	alph := dist.TileMapping()

	b := board.MakeBoard(board.CrosswordGameBoard)

	testCases := []crossSetEdgeTestCase{
		{0, " A", board.CrossSetFromString("ABDEFGHIJKLMNPRSTWXYZ", alph), 1},
		{1, "A ", board.CrossSetFromString("ABDEFGHIJKLMNPRSTWXYZ", alph), 1},
		{0, " Q", board.CrossSetFromString("I", alph), 10},
		{1, "Q ", board.CrossSetFromString("I", alph), 10},
		{0, " V", board.CrossSet(0), 4},
		{1, "V ", board.CrossSet(0), 4},
		{0, " T", board.CrossSetFromString("AEIOSU", alph), 1},
		{1, "T ", board.CrossSetFromString("AEIOSU", alph), 1},

		{0, " ABCDE", board.CrossSetFromString("BHKLRU", alph), 10},
		{1, "A BCDE", board.CrossSetFromString("BHKLRU", alph), 10},
		{2, "AB CDE", board.CrossSetFromString("BHKLRU", alph), 10},
		{3, "ABC DE", board.CrossSetFromString("BHKLRU", alph), 10},
		{4, "ABCD E", board.CrossSetFromString("BHKLRU", alph), 10},
		{5, "ABCDE ", board.CrossSetFromString("BHKLRU", alph), 10},

		{0, " WORKBLUH", board.CrossSetFromString("S", alph), 20},
		{4, "WORK BLUH", board.CrossSetFromString("S", alph), 20},
		{8, "WORKBLUH ", board.CrossSetFromString("S", alph), 20},

		// Same tiles, but some are blanks: same cross set, lower cross score.
		{0, " WORKbLUh", board.CrossSetFromString("S", alph), 13},
		{4, "WORK bLUh", board.CrossSetFromString("S", alph), 13},
		{8, "WORKbLUh ", board.CrossSetFromString("S", alph), 13},

		// No alphagram extends this one, wherever the gap is.
		{0, " TRONGLE", board.CrossSet(0), 8},
		{4, "TRON GLE", board.CrossSet(0), 8},
		{7, "TRONGLE ", board.CrossSet(0), 8},

		{14, "       ZYZZVAY ", board.CrossSetFromString("S", alph), 43},
		{14, "        ZYZZVY ", board.CrossSetFromString("A", alph), 42},
	}

	row := 4
	for _, tc := range testCases {
		b.SetRow(row, tc.rowContents, alph)
		GenAlphaCrossSet(b, row, tc.col, board.HorizontalDirection, kad, dist)
		if b.GetCrossSet(row, tc.col, board.HorizontalDirection) != tc.crossSet {
			t.Errorf("For row=%q col=%v, expected cross-set %v, got %v",
				tc.rowContents, tc.col, tc.crossSet,
				b.GetCrossSet(row, tc.col, board.HorizontalDirection))
		}
		if b.GetCrossScore(row, tc.col, board.HorizontalDirection) != tc.score {
			t.Errorf("For row=%q col=%v, expected cross-score %v, got %v",
				tc.rowContents, tc.col, tc.score,
				b.GetCrossScore(row, tc.col, board.HorizontalDirection))
		}
	}
}

// TestGenAlphaCrossSetExtensionSets checks the invariant the shadow player
// depends on: extension sets are meaningless in WordSmog but must never be
// left at zero, or shadowPlayForAnchor discards the anchor.
func TestGenAlphaCrossSetExtensionSets(t *testing.T) {
	is := is.New(t)

	kad := alphaDawgFromLexicon(t, "CSW21")
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.WGLConfig())
	is.NoErr(err)
	alph := dist.TileMapping()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetRow(4, "   ABCDE   ", alph)
	GenAllAlphaCrossSets(b, kad, dist)

	// The empty squares flanking the block, and the block's end tiles, are
	// where shadow reads extension sets from.
	for _, col := range []int{2, 3, 7, 8} {
		is.Equal(b.GetLeftExtSet(4, col, board.VerticalDirection), uint64(board.TrivialCrossSet))
		is.Equal(b.GetRightExtSet(4, col, board.VerticalDirection), uint64(board.TrivialCrossSet))
	}
}
