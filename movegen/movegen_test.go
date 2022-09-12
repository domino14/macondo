package movegen

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/testcommon"
	"github.com/matryer/is"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/stretchr/testify/assert"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	testcommon.CreateGaddags(DefaultConfig, []string{"America"})
	os.Exit(m.Run())
}

func GaddagFromLexicon(lex string) (*gaddag.SimpleGaddag, error) {
	return gaddag.LoadGaddag(filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag"))
}

func Filter(moves []*move.Move, f func(*move.Move) bool) []*move.Move {
	ms := make([]*move.Move, 0)
	for _, m := range moves {
		if f(m) {
			ms = append(ms, m)
		}
	}
	return ms
}

func scoringPlays(moves []*move.Move) []*move.Move {
	return Filter(moves, func(m *move.Move) bool {
		return m.Action() == move.MoveTypePlay
	})
}

func nonScoringPlays(moves []*move.Move) []*move.Move {
	return Filter(moves, func(m *move.Move) bool {
		return m.Action() != move.MoveTypePlay
	})
}

func TestGenBase(t *testing.T) {
	// Sanity check. A board with no cross checks should generate nothing.
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	rack := alphabet.RackFromString("AEINRST", gd.GetAlphabet())
	board := board.MakeBoard(board.CrosswordGameBoard)
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, board, dist)
	generator.csetGen.CS.ClearAll()
	generator.curAnchorCol = 8
	// Row 4 for shiz and gig
	generator.curRowIdx = 4
	generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true)

	if len(generator.plays) != 0 {
		t.Errorf("Generated %v plays, expected %v", len(generator.plays), 0)
	}
}

type SimpleGenTestCase struct {
	rack          string
	curAnchorCol  int
	row           int
	rowString     string
	expectedPlays int
}

func TestSimpleRowGen(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	var cases = []SimpleGenTestCase{
		{"P", 11, 2, "     REGNANT", 1},
		{"O", 9, 2, "  PORTOLAN", 1},
		{"S", 9, 2, "  PORTOLAN", 1},
		{"?", 9, 2, "  PORTOLAN", 2},
		{"TY", 7, 2, "  SOVRAN", 1},
		{"ING", 6, 2, "  LAUGH", 1},
		{"ZA", 3, 4, "  BE", 0},
		{"AENPPSW", 14, 4, "        CHAWING", 1}, // wappenschawing
		{"ABEHINT", 9, 4, "   THERMOS  A", 2},    // nethermost hithermost
		{"ABEHITT", 8, 4, "  THERMOS A   ", 1},   // anchor on s: thermostat
		{"TT", 10, 4, "  THERMOS A   ", 3},       // anchor on a: thermostat, at, att
		{"A", 1, 4, " B", 1},
		{"A", 1, 4, " b", 1},
	}
	board := board.MakeBoard(board.CrosswordGameBoard)
	board.Clear()
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	for idx, tc := range cases {
		generator := NewGordonGenerator(gd, board, dist)
		generator.csetGen.CS.SetAll()
		generator.curAnchorCol = tc.curAnchorCol
		rack := alphabet.RackFromString(tc.rack, gd.GetAlphabet())
		board.SetRow(tc.row, tc.rowString, gd.GetAlphabet())
		generator.curRowIdx = tc.row
		generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true)
		if len(generator.plays) != tc.expectedPlays {
			t.Errorf("%v Generated %v plays, expected %v", idx, generator.plays, tc.expectedPlays)
		}
	}
}

func TestGenThroughBothWaysAllowedLetters(t *testing.T) {
	// Basically, allow HITHERMOST but not NETHERMOST.
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	rack := alphabet.RackFromString("ABEHINT", gd.GetAlphabet())
	bd := board.MakeBoard(board.CrosswordGameBoard)
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, dist)
	generator.csetGen.CS.SetAll()
	generator.curAnchorCol = 9
	bd.SetRow(4, "   THERMOS  A", gd.GetAlphabet())
	generator.curRowIdx = 4
	ml, err := gd.GetAlphabet().Val('I')
	is.NoErr(err)
	generator.csetGen.CS.Clear(int(generator.curRowIdx), 2, board.VerticalDirection)
	generator.csetGen.CS.Add(int(generator.curRowIdx), 2, ml, board.VerticalDirection)
	generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true)
	// it should generate HITHERMOST only
	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
	m := generator.plays[0]
	if m.Tiles().UserVisible(gd.GetAlphabet()) != "HI.......T" {
		t.Errorf("Got the wrong word: %v", m.Tiles().UserVisible(gd.GetAlphabet()))
	}
	if m.Leave().UserVisible(gd.GetAlphabet()) != "ABEN" {
		t.Errorf("Got the wrong leave: %v", m.Leave().UserVisible(gd.GetAlphabet()))
	}
}

func TestRowGen(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	generator.ResetCrossesAndAnchors()
	rack := alphabet.RackFromString("AAEIRST", gd.GetAlphabet())
	generator.curRowIdx = 4
	generator.curAnchorCol = 8
	generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true)

	sort.Slice(generator.plays, func(i, j int) bool {
		return generator.plays[i].Score() > generator.plays[j].Score()
	})

	// Should generate AIRGLOWS and REGLOWS only
	if len(generator.plays) != 2 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 2)
	}
	if generator.plays[0].Leave().UserVisible(gd.GetAlphabet()) != "AEST" {
		t.Errorf("Leave was wrong: %v",
			generator.plays[0].Leave().UserVisible(gd.GetAlphabet()))
	}
	if generator.plays[1].Leave().UserVisible(gd.GetAlphabet()) != "AAIST" {
		t.Errorf("Leave was wrong: %v",
			generator.plays[1].Leave().UserVisible(gd.GetAlphabet()))
	}
}

func TestOtherRowGen(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	generator.ResetCrossesAndAnchors()

	rack := alphabet.RackFromString("A", gd.GetAlphabet())
	generator.curRowIdx = 14
	generator.curAnchorCol = 8
	generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true)
	// Should generate AVENGED only
	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
	m := generator.plays[0]
	if m.Tiles().UserVisible(gd.GetAlphabet()) != "A......" {
		t.Errorf("Expected proper play-through markers (A......), got %v",
			m.Tiles().UserVisible(gd.GetAlphabet()))
	}
}

func TestOneMoreRowGen(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)

	rack := alphabet.RackFromString("A", gd.GetAlphabet())
	generator.curRowIdx = 0
	generator.curAnchorCol = 11
	generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true)

	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
	m := generator.plays[0]
	if m.ShortDescription() != " 1L .A" {
		t.Errorf("Expected 1L .A, got %v", m.ShortDescription())
	}
}

func TestGenMoveJustOnce(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	generator.ResetCrossesAndAnchors()
	bd.Transpose()

	rack := alphabet.RackFromString("AELT", gd.GetAlphabet())
	// We want to generate TAEL parallel to ABANDON (making RESPONDED)
	// See VsMatt board definition above.
	generator.curRowIdx = 10
	generator.vertical = true
	generator.lastAnchorCol = 100
	for anchorCol := 8; anchorCol <= 12; anchorCol++ {
		generator.curAnchorCol = anchorCol
		generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, false)
		generator.lastAnchorCol = anchorCol
	}
	if len(generator.plays) != 34 {
		t.Errorf("Expected %v, got %v plays", 34, generator.plays)
	}
}

func TestGenAllMovesSingleTile(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	generator.ResetCrossesAndAnchors()

	generator.GenAll(alphabet.RackFromString("A", alph), false)
	log.Debug().Interface("plays", generator.plays).Msg("plays")
	assert.Equal(t, 24, len(scoringPlays(generator.plays)))
	// t.Fail()
}

func TestGenAllMovesFullRack(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	generator.ResetCrossesAndAnchors()

	generator.GenAll(alphabet.RackFromString("AABDELT", alph), true)
	// There should be 673 unique scoring plays and 95 exchanges.
	assert.Equal(t, 673, len(scoringPlays(generator.plays)))
	assert.Equal(t, 95, len(nonScoringPlays(generator.plays)))

	highestScores := []int{38, 36, 36, 34, 34, 33, 30, 30, 30, 28}
	for idx, score := range highestScores {
		assert.Equal(t, score, generator.plays[idx].Score())
	}
}

func TestGenAllMovesFullRackAgain(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	generator.ResetCrossesAndAnchors()

	generator.GenAll(alphabet.RackFromString("AFGIIIS", alph), true)
	// There should be 219 unique plays
	assert.Equal(t, 219, len(scoringPlays(generator.plays)))
	assert.Equal(t, 63, len(nonScoringPlays(generator.plays)))
}

// func TestGenMoveWithOverlaps(t *testing.T) {
// 	gd, err := GaddagFromLexicon("America")
// 	if err != nil {
// 		t.Errorf("Expected error to be nil, got %v", err)
// 	}
// 	alph := gd.GetAlphabet()

// 	bd := board.MakeBoard(board.CrosswordGameBoard)
// 	ld := alphabet.EnglishLetterDistribution(gd.GetAlphabet())
// 	generator := NewGordonGenerator(gd, bd, ld)
// 	bd.SetToGame(gd.GetAlphabet(), board.MavenVsMacondo)
// 	cross_set.GenAllCrossSets(bd, gd, ld)
// 	fmt.Println(bd.ToDisplayText(alph))

// 	generator.GenAll(alphabet.RackFromString("AEEORS?", alph), true)
// 	fmt.Println(generator.Plays())
// 	assert.True(t, false)
// }

func TestGenAllMovesSingleBlank(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	generator.ResetCrossesAndAnchors()

	generator.GenAll(alphabet.RackFromString("?", alph), true)
	// There should be 166 unique plays. Quackle does not generate all blank
	// plays, even when told to generate all plays!!
	assert.Equal(t, 166, len(scoringPlays(generator.plays)))
	// Exch ?
	assert.Equal(t, 1, len(nonScoringPlays(generator.plays)))
}

func TestGenAllMovesTwoBlanksOnly(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	generator.ResetCrossesAndAnchors()

	generator.GenAll(alphabet.RackFromString("??", alph), true)
	// Quackle generates 1827 unique plays. (my movegen generates 1958)
	// With one blank (the test above), Quackle generates 35 moves, I generate
	// 166 by hand. The difference is 131. It seems Quackle does not generate
	// all plays for one blank, only the first one alphabetically for every position.
	// The difference between 1827 and 1958 is also 131, so I think this is
	// ok.
	assert.Equal(t, 1958, len(scoringPlays(generator.plays)))
	assert.Equal(t, 2, len(nonScoringPlays(generator.plays)))
}

func TestGenAllMovesWithBlanks(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
	generator.ResetCrossesAndAnchors()

	generator.GenAll(alphabet.RackFromString("DDESW??", alph), false)
	// If I do DDESW? in quackle i generate 1483 moves. My movegen generates
	// 1586, possibly by the same logic as the above.
	// If I add 103 to the Quackle-generated 8194 moves for both blanks (DDESW??)
	// I get 8297, so there should be 8297 unique plays
	assert.Equal(t, 8297, len(scoringPlays(generator.plays)))
	assert.Equal(t, 0, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, 106, generator.plays[0].Score()) // hEaDW(OR)DS!
	assert.Equal(t, "", generator.plays[0].Leave().UserVisible(alph))
	assert.Equal(t, "S", generator.plays[1].Leave().UserVisible(alph))
	// There are 7 plays worth 32 pts.
	rewards := 0
	for i := 2; i < 9; i++ {
		assert.Equal(t, 32, generator.plays[i].Score())
		if generator.plays[i].Tiles().UserVisible(alph) == "rEW..DS" {
			rewards = i
		}
	}
	assert.NotEqual(t, 0, rewards)
	assert.Equal(t, "D?", generator.plays[rewards].Leave().UserVisible(alph))
}

func TestGiantTwentySevenTimer(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsOxy)
	generator.ResetCrossesAndAnchors()

	generator.GenAll(alphabet.RackFromString("ABEOPXZ", alph), false)
	assert.Equal(t, 519, len(scoringPlays(generator.plays)))
	// Bag has 5 tiles so no exchanges should be generated.
	assert.Equal(t, 0, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, 1780, generator.plays[0].Score()) // oxyphenbutazone
	assert.Equal(t, "", generator.plays[0].Leave().UserVisible(alph))
}

func TestGenerateEmptyBoard(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	generator.ResetCrossesAndAnchors()

	generator.GenAll(alphabet.RackFromString("DEGORV?", alph), true)
	assert.Equal(t, 3313, len(scoringPlays(generator.plays)))
	assert.Equal(t, 127, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, 80, generator.plays[0].Score())
	assert.Equal(t, "", generator.plays[0].Leave().UserVisible(alph))
}

func TestGenerateOtherEmptyBoard(t *testing.T) {
	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	generator := NewGordonGenerator(gd, bd, ld)
	generator.ResetCrossesAndAnchors()

	generator.GenAll(alphabet.RackFromString("AAADERW", alph), true)
	assert.Equal(t, 141, len(scoringPlays(generator.plays)))
	assert.Equal(t, 63, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, 26, generator.plays[0].Score()) // highest scoring wared
}

func TestGenerateNoPlays(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
	generator.ResetCrossesAndAnchors()

	generator.GenAll(alphabet.RackFromString("V", alph), false)
	// V won't play anywhere
	assert.Equal(t, 0, len(scoringPlays(generator.plays)))
	assert.Equal(t, 1, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, move.MoveTypePass, generator.plays[0].Action())
}

func TestRowEquivalent(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	bd.SetToGame(gd.GetAlphabet(), board.TestDupe)
	cs := cross_set.MakeBoardCrossSets(bd)
	a := MakeAnchors(bd)
	cross_set.GenAllCrossSets(bd, cs, gd, ld)
	a.UpdateAllAnchors()

	bd2 := board.MakeBoard(board.CrosswordGameBoard)

	bd2.SetRow(7, " INCITES", alph)
	bd2.SetRow(8, "IS", alph)
	bd2.SetRow(9, "T", alph)

	cs2 := cross_set.MakeBoardCrossSets(bd2)
	a2 := MakeAnchors(bd2)
	cross_set.GenAllCrossSets(bd2, cs2, gd, ld)
	a2.UpdateAllAnchors()

	assert.True(t, bd.Equals(bd2))
}

// Note about the comments on the following benchmarks:
// The benchmarks are a bit slower now. This largely comes
// from the sorting / equity stuff that wasn't there before.

func BenchmarkGenEmptyBoard(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		b.Error(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 1.67ms per operation

		bd := board.MakeBoard(board.CrosswordGameBoard)
		generator := NewGordonGenerator(gd, bd, ld)
		generator.ResetCrossesAndAnchors()
		generator.GenAll(alphabet.RackFromString("AEINRST", alph), true)
	}
}

func BenchmarkGenFullRack(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		b.Error(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 930 μs per operation on my macbook pro!! amazing!!!
		// update: 550 μs per operation on M1 Macbook Pro after some refactoring improvements
		// Benchmark run 2022-08-09 on M1 MBP (Docker not running):
		// go 1.18

		// 2190	    513518 ns/op	  204738 B/op	    3884 allocs/op
		bd := board.MakeBoard(board.CrosswordGameBoard)

		generator := NewGordonGenerator(gd, bd, ld)
		bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
		generator.ResetCrossesAndAnchors()

		generator.GenAll(alphabet.RackFromString("AABDELT", alph), true)
	}
}

func BenchmarkJustMovegen(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	// generator.SetPlayRecorder(NullPlayRecorder)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)
	b.ReportAllocs()
	b.ResetTimer()
	rack := alphabet.RackFromString("AABDELT", alph)
	for i := 0; i < b.N; i++ {
		// Benchmark run 2022-08-13 on M1 MBP (Docker not running):
		// go 1.18

		// 3349	    334262 ns/op	  147152 B/op	    2991 allocs/op
		generator.GenAll(rack, true)
	}
}

func BenchmarkGenOneBlank(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		b.Error(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 5.43 ms per operation on my macbook pro.
		bd := board.MakeBoard(board.CrosswordGameBoard)

		generator := NewGordonGenerator(gd, bd, ld)
		bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
		generator.ResetCrossesAndAnchors()

		generator.GenAll(alphabet.RackFromString("ADDESW?", alph), false)
	}
}

func BenchmarkGenBothBlanks(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		b.Error(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// ~16.48ms per operation on my macbook pro.
		bd := board.MakeBoard(board.CrosswordGameBoard)

		generator := NewGordonGenerator(gd, bd, ld)
		bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
		generator.ResetCrossesAndAnchors()

		generator.GenAll(alphabet.RackFromString("DDESW??", alph), false)
	}
}

func TestGenExchange(t *testing.T) {
	// rack := alphabet.RackFromString("AAABCCD", alphabet.EnglishAlphabet())
	rack := alphabet.RackFromString("ABCDEF?", alphabet.EnglishAlphabet())

	bd := board.MakeBoard(board.CrosswordGameBoard)
	gd, _ := GaddagFromLexicon("America")
	ld, _ := alphabet.EnglishLetterDistribution(&DefaultConfig)
	gen := NewGordonGenerator(gd, bd, ld)

	gen.generateExchangeMoves(rack, 0, 0)
	assert.Equal(t, len(gen.plays), 127)
}
