package movegen

import (
	"log"
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/move"
	"github.com/stretchr/testify/assert"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	for _, lex := range []string{"America"} {
		gdgPath := filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag")
		if _, err := os.Stat(gdgPath); os.IsNotExist(err) {
			gaddagmaker.GenerateGaddag(filepath.Join(DefaultConfig.LexiconPath, lex+".txt"), true, true)
			err = os.Rename("out.gaddag", gdgPath)
			if err != nil {
				panic(err)
			}
		}
	}
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

	gd, err := GaddagFromLexicon("America")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	rack := alphabet.RackFromString("AEINRST", gd.GetAlphabet())
	board := board.MakeBoard(board.CrosswordGameBoard)
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	generator := NewGordonGenerator(gd, board, dist)

	board.ClearAllCrosses()
	generator.curAnchorCol = 8
	// Row 4 for shiz and gig
	generator.curRowIdx = 4
	generator.recursiveGen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())

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

	gd, err := GaddagFromLexicon("America")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
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
	if err != nil {
		t.Error(err)
	}
	for idx, tc := range cases {
		log.Printf("Case %v", idx)
		generator := NewGordonGenerator(gd, board, dist)
		generator.curAnchorCol = tc.curAnchorCol
		rack := alphabet.RackFromString(tc.rack, gd.GetAlphabet())
		board.SetRow(tc.row, tc.rowString, gd.GetAlphabet())
		generator.curRowIdx = tc.row
		generator.recursiveGen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
			gd.GetRootNodeIndex())
		if len(generator.plays) != tc.expectedPlays {
			t.Errorf("%v Generated %v plays, expected %v", idx, generator.plays, tc.expectedPlays)
		}
	}
}

func TestGenThroughBothWaysAllowedLetters(t *testing.T) {
	// Basically, allow HITHERMOST but not NETHERMOST.

	gd, err := GaddagFromLexicon("America")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	rack := alphabet.RackFromString("ABEHINT", gd.GetAlphabet())
	bd := board.MakeBoard(board.CrosswordGameBoard)
	dist, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	generator := NewGordonGenerator(gd, bd, dist)
	generator.curAnchorCol = 9

	bd.SetRow(4, "   THERMOS  A", gd.GetAlphabet())
	generator.curRowIdx = 4
	ml, _ := gd.GetAlphabet().Val('I')
	bd.ClearCrossSet(int(generator.curRowIdx), 2, board.VerticalDirection)
	bd.SetCrossSetLetter(int(generator.curRowIdx), 2, board.VerticalDirection, ml)
	generator.recursiveGen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
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

	gd, err := GaddagFromLexicon("America")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	cross_set.GenAllCrossSets(bd, gd, ld)
	rack := alphabet.RackFromString("AAEIRST", gd.GetAlphabet())
	generator.curRowIdx = 4
	generator.curAnchorCol = 8
	generator.recursiveGen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	generator.dedupeAndSortPlays()
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
			generator.plays[0].Leave().UserVisible(gd.GetAlphabet()))
	}
}

func TestOtherRowGen(t *testing.T) {

	gd, err := GaddagFromLexicon("America")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)

	rack := alphabet.RackFromString("A", gd.GetAlphabet())
	generator.curRowIdx = 14
	generator.curAnchorCol = 8
	generator.recursiveGen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
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

func TestGenMoveJustOnce(t *testing.T) {
	gd, err := GaddagFromLexicon("America")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)
	bd.Transpose()

	rack := alphabet.RackFromString("AELT", gd.GetAlphabet())
	// We want to generate TAEL parallel to ABANDON (making RESPONDED)
	// See VsMatt board definition above.
	generator.curRowIdx = 10
	generator.vertical = true
	generator.lastAnchorCol = 100
	for anchorCol := 8; anchorCol <= 12; anchorCol++ {
		generator.curAnchorCol = anchorCol
		generator.recursiveGen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
			gd.GetRootNodeIndex())
		generator.lastAnchorCol = anchorCol
	}
	if len(generator.plays) != 34 {
		t.Errorf("Expected %v, got %v plays", 34, generator.plays)
	}
}

func TestGenAllMovesSingleTile(t *testing.T) {
	gd, err := GaddagFromLexicon("America")
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
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(alphabet.RackFromString("A", alph), false)
	assert.Equal(t, 24, len(scoringPlays(generator.plays)))
	// t.Fail()
}

func TestGenAllMovesFullRack(t *testing.T) {
	gd, err := GaddagFromLexicon("America")
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
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(alphabet.RackFromString("AABDELT", alph), true)
	// There should be 673 unique scoring plays, 95 exchanges and 1 pass.
	assert.Equal(t, 673, len(scoringPlays(generator.plays)))
	assert.Equal(t, 95, len(nonScoringPlays(generator.plays)))

	highestScores := []int{38, 36, 36, 34, 34, 33, 30, 30, 30, 28}
	for idx, score := range highestScores {
		assert.Equal(t, score, generator.plays[idx].Score())
	}
}

func TestGenAllMovesFullRackAgain(t *testing.T) {
	gd, err := GaddagFromLexicon("America")
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
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	cross_set.GenAllCrossSets(bd, gd, ld)

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
	gd, err := GaddagFromLexicon("America")
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
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(alphabet.RackFromString("?", alph), true)
	// There should be 166 unique plays. Quackle does not generate all blank
	// plays, even when told to generate all plays!!
	assert.Equal(t, 166, len(scoringPlays(generator.plays)))
	// Exch ?
	assert.Equal(t, 1, len(nonScoringPlays(generator.plays)))
}

func TestGenAllMovesTwoBlanksOnly(t *testing.T) {
	gd, err := GaddagFromLexicon("America")
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
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	cross_set.GenAllCrossSets(bd, gd, ld)

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
	gd, err := GaddagFromLexicon("America")
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
	bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
	cross_set.GenAllCrossSets(bd, gd, ld)

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
	gd, err := GaddagFromLexicon("America")
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
	bd.SetToGame(gd.GetAlphabet(), board.VsOxy)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(alphabet.RackFromString("ABEOPXZ", alph), false)
	assert.Equal(t, 519, len(scoringPlays(generator.plays)))
	// Bag has 5 tiles so no exchanges should be generated.
	assert.Equal(t, 0, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, 1780, generator.plays[0].Score()) // oxyphenbutazone
	assert.Equal(t, "", generator.plays[0].Leave().UserVisible(alph))
}

func TestGenerateEmptyBoard(t *testing.T) {
	gd, err := GaddagFromLexicon("America")
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
	bd.UpdateAllAnchors()

	generator.GenAll(alphabet.RackFromString("DEGORV?", alph), true)
	assert.Equal(t, 3313, len(scoringPlays(generator.plays)))
	assert.Equal(t, 127, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, 80, generator.plays[0].Score())
	assert.Equal(t, "", generator.plays[0].Leave().UserVisible(alph))
}

func TestGenerateNoPlays(t *testing.T) {
	gd, err := GaddagFromLexicon("America")
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
	bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(alphabet.RackFromString("V", alph), false)
	// V won't play anywhere
	assert.Equal(t, 0, len(scoringPlays(generator.plays)))
	assert.Equal(t, 1, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, move.MoveTypePass, generator.plays[0].Action())
}

func TestGenerateDupes(t *testing.T) {
	gd, err := GaddagFromLexicon("America")
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
	bd.SetToGame(gd.GetAlphabet(), board.TestDupe)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(alphabet.RackFromString("Z", alph), false)
	s := scoringPlays(generator.plays)
	assert.Equal(t, 1, len(s))
	assert.True(t, s[0].HasDupe())
}

func TestRowEquivalent(t *testing.T) {
	// XXX: This test should probably be in the board.
	gd, err := GaddagFromLexicon("America")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bd.SetToGame(gd.GetAlphabet(), board.TestDupe)
	cross_set.GenAllCrossSets(bd, gd, ld)

	bd2 := board.MakeBoard(board.CrosswordGameBoard)

	bd2.SetRow(7, " INCITES", alph)
	bd2.SetRow(8, "IS", alph)
	bd2.SetRow(9, "T", alph)
	bd2.UpdateAllAnchors()
	cross_set.GenAllCrossSets(bd2, gd, ld)

	assert.True(t, bd.Equals(bd2))
}

// Note about the comments on the following benchmarks:
// The benchmarks are a bit slower now. This largely comes
// from the sorting / equity stuff that wasn't there before.

func BenchmarkGenEmptyBoard(b *testing.B) {
	gd, err := GaddagFromLexicon("America")
	if err != nil {
		b.Errorf("Expected error to be nil, got %v", err)
	}
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 1.67ms per operation

		bd := board.MakeBoard(board.CrosswordGameBoard)
		ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
		if err != nil {
			b.Error(err)
		}
		generator := NewGordonGenerator(gd, bd, ld)
		bd.UpdateAllAnchors()
		generator.GenAll(alphabet.RackFromString("AEINRST", alph), true)
	}
}

func BenchmarkGenFullRack(b *testing.B) {
	gd, err := GaddagFromLexicon("America")
	if err != nil {
		b.Errorf("Expected error to be nil, got %v", err)
	}
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 930 μs per operation on my macbook pro!! amazing!!!

		bd := board.MakeBoard(board.CrosswordGameBoard)
		ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
		if err != nil {
			b.Error(err)
		}
		generator := NewGordonGenerator(gd, bd, ld)
		bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
		cross_set.GenAllCrossSets(bd, gd, ld)

		generator.GenAll(alphabet.RackFromString("AABDELT", alph), true)
	}
}

func BenchmarkGenOneBlank(b *testing.B) {
	gd, err := GaddagFromLexicon("America")
	if err != nil {
		b.Errorf("Expected error to be nil, got %v", err)
	}
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 5.43 ms per operation on my macbook pro.
		bd := board.MakeBoard(board.CrosswordGameBoard)
		ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
		if err != nil {
			b.Error(err)
		}
		generator := NewGordonGenerator(gd, bd, ld)
		bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
		cross_set.GenAllCrossSets(bd, gd, ld)

		generator.GenAll(alphabet.RackFromString("ADDESW?", alph), false)
	}
}

func BenchmarkGenBothBlanks(b *testing.B) {
	gd, err := GaddagFromLexicon("America")
	if err != nil {
		b.Errorf("Expected error to be nil, got %v", err)
	}
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// ~16.48ms per operation on my macbook pro.
		bd := board.MakeBoard(board.CrosswordGameBoard)
		ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
		if err != nil {
			b.Error(err)
		}
		generator := NewGordonGenerator(gd, bd, ld)
		bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
		cross_set.GenAllCrossSets(bd, gd, ld)

		generator.GenAll(alphabet.RackFromString("DDESW??", alph), false)
	}
}
