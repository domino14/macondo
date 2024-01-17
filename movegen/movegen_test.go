package movegen

import (
	"fmt"
	"os"
	"sort"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/testhelpers"
	"github.com/domino14/macondo/tinymove/conversions"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	os.Exit(m.Run())
}

func GaddagFromLexicon(lex string) (gaddag.WordGraph, error) {
	return kwg.Get(DefaultConfig.AllSettings(), lex)
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

	gd, err := GaddagFromLexicon("NWL18")
	is.NoErr(err)
	rack := tilemapping.RackFromString("AEINRST", gd.GetAlphabet())
	board := board.MakeBoard(board.CrosswordGameBoard)
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, board, dist)

	board.ClearAllCrosses()
	generator.curAnchorCol = 8
	// Row 4 for shiz and gig
	generator.curRowIdx = 4
	generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true,
		0, 0, 1)

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

	gd, err := GaddagFromLexicon("NWL20")
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
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	for idx, tc := range cases {
		generator := NewGordonGenerator(gd, board, dist)
		generator.curAnchorCol = tc.curAnchorCol
		rack := tilemapping.RackFromString(tc.rack, gd.GetAlphabet())
		board.SetRow(tc.row, tc.rowString, gd.GetAlphabet())
		generator.curRowIdx = tc.row
		generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true,
			0, 0, 1)
		if len(generator.plays) != tc.expectedPlays {
			t.Errorf("%v Generated %v plays, expected %v", idx, generator.plays, tc.expectedPlays)
		}
	}
}

func TestGenThroughBothWaysAllowedLetters(t *testing.T) {
	// Basically, allow HITHERMOST but not NETHERMOST.
	is := is.New(t)

	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)
	rack := tilemapping.RackFromString("ABEHINT", gd.GetAlphabet())
	bd := board.MakeBoard(board.CrosswordGameBoard)
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, dist)
	generator.curAnchorCol = 9
	bd.SetRow(4, "   THERMOS  A", gd.GetAlphabet())
	generator.curRowIdx = 4
	ml, err := gd.GetAlphabet().Val("I")
	is.NoErr(err)
	bd.ClearCrossSet(int(generator.curRowIdx), 2, board.VerticalDirection)
	bd.SetCrossSetLetter(int(generator.curRowIdx), 2, board.VerticalDirection, ml)
	generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true,
		0, 0, 1)
	// it should generate HITHERMOST only
	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
	m := generator.plays[0]
	if m.Tiles().UserVisiblePlayedTiles(gd.GetAlphabet()) != "HI.......T" {
		t.Errorf("Got the wrong word: %v", m.Tiles().UserVisiblePlayedTiles(gd.GetAlphabet()))
	}
	if m.Leave().UserVisiblePlayedTiles(gd.GetAlphabet()) != "ABEN" {
		t.Errorf("Got the wrong leave: %v", m.Leave().UserVisiblePlayedTiles(gd.GetAlphabet()))
	}
}

func TestRowGen(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	cross_set.GenAllCrossSets(bd, gd, ld)
	rack := tilemapping.RackFromString("AAEIRST", gd.GetAlphabet())
	generator.curRowIdx = 4
	generator.curAnchorCol = 8
	generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true,
		0, 0, 1)

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

	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)

	rack := tilemapping.RackFromString("A", gd.GetAlphabet())
	generator.curRowIdx = 14
	generator.curAnchorCol = 8
	generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true,
		0, 0, 1)
	// Should generate AVENGED only
	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
	m := generator.plays[0]
	if m.Tiles().UserVisiblePlayedTiles(gd.GetAlphabet()) != "A......" {
		t.Errorf("Expected proper play-through markers (A......), got %v",
			m.Tiles().UserVisiblePlayedTiles(gd.GetAlphabet()))
	}
}

func TestOneMoreRowGen(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)

	rack := tilemapping.RackFromString("A", gd.GetAlphabet())
	generator.curRowIdx = 0
	generator.curAnchorCol = 11
	generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, true,
		0, 0, 1)

	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
	m := generator.plays[0]
	d := m.ShortDescription()
	if d != " 1L .A" {
		t.Errorf("Expected 1L .A, got %v", d)
	}
}

func TestGenMoveJustOnce(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)
	bd.Transpose()

	rack := tilemapping.RackFromString("AELT", gd.GetAlphabet())
	// We want to generate TAEL parallel to ABANDON (making RESPONDED)
	// See VsMatt board definition above.
	generator.curRowIdx = 10
	generator.vertical = true
	generator.lastAnchorCol = 100
	for anchorCol := 8; anchorCol <= 12; anchorCol++ {
		generator.curAnchorCol = anchorCol
		generator.recursiveGen(generator.curAnchorCol, rack, gd.GetRootNodeIndex(), generator.curAnchorCol, generator.curAnchorCol, false,
			0, 0, 1)
		generator.lastAnchorCol = anchorCol
	}
	if len(generator.plays) != 34 {
		t.Errorf("Expected %v, got %v (%d) plays", 34, generator.plays, len(generator.plays))
	}
}

func TestGenAllMovesSingleTile(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(tilemapping.RackFromString("A", alph), false)
	assert.Equal(t, 24, len(scoringPlays(generator.plays)))
	// t.Fail()
}

func TestGenAllMovesFullRack(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(tilemapping.RackFromString("AABDELT", alph), true)

	// There should be 667 unique scoring plays and 95 exchanges.
	assert.Equal(t, 667, len(scoringPlays(generator.plays)))
	assert.Equal(t, 95, len(nonScoringPlays(generator.plays)))

	highestScores := []int{38, 36, 36, 34, 34, 33, 30, 30, 30, 28}
	for idx, score := range highestScores {
		fmt.Println(generator.plays[idx].String())
		assert.Equal(t, score, generator.plays[idx].Score())
	}
}

func TestGenAllMovesFullRackAgain(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(tilemapping.RackFromString("AFGIIIS", alph), true)
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
// 	ld := tilemapping.EnglishLetterDistribution(gd.GetAlphabet())
// 	generator := NewGordonGenerator(gd, bd, ld)
// 	bd.SetToGame(gd.GetAlphabet(), board.MavenVsMacondo)
// 	cross_set.GenAllCrossSets(bd, gd, ld)
// 	fmt.Println(bd.ToDisplayText(alph))

// 	generator.GenAll(tilemapping.RackFromString("AEEORS?", alph), true)
// 	fmt.Println(generator.Plays())
// 	assert.True(t, false)
// }

func TestGenAllMovesSingleBlank(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(tilemapping.RackFromString("?", alph), true)
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
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsEd)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(tilemapping.RackFromString("??", alph), true)
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
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(tilemapping.RackFromString("DDESW??", alph), false)
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
		if generator.plays[i].Tiles().UserVisiblePlayedTiles(alph) == "rEW..DS" {
			rewards = i
		}
	}
	assert.NotEqual(t, 0, rewards)
	assert.Equal(t, "?D", generator.plays[rewards].Leave().UserVisible(alph))
}

func TestSmallMoveRecorder(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
	cross_set.GenAllCrossSets(bd, gd, ld)
	generator.SetPlayRecorder(AllPlaysSmallRecorder)
	rack := tilemapping.RackFromString("DDESW??", alph)
	generator.GenAll(rack, false)
	// If I do DDESW? in quackle i generate 1483 moves. My movegen generates
	// 1586, possibly by the same logic as the above.
	// If I add 103 to the Quackle-generated 8194 moves for both blanks (DDESW??)
	// I get 8297, so there should be 8297 unique plays
	assert.Equal(t, 8297, len(generator.smallPlays))

	sort.Slice(generator.smallPlays, func(i, j int) bool {
		return generator.smallPlays[i].Score() > generator.smallPlays[j].Score()
	})

	assert.Equal(t, 106, generator.smallPlays[0].Score()) // hEaDW(OR)DS!
	// There are 7 plays worth 32 pts.
	rewards := 0
	for i := 2; i < 9; i++ {
		assert.Equal(t, 32, generator.smallPlays[i].Score())
		m := &move.Move{}
		conversions.SmallMoveToMove(&generator.smallPlays[i], m, alph, bd, rack)

		if m.Tiles().UserVisiblePlayedTiles(alph) == "rEW..DS" {
			rewards = i
		}
	}
	assert.NotEqual(t, 0, rewards)

	m := &move.Move{}
	conversions.SmallMoveToMove(&generator.smallPlays[rewards], m, alph, bd, rack)

	assert.Equal(t, "?D", m.Leave().UserVisible(alph))
}

func TestTopPlayOnlyRecorder(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	// VsJeremy but as a CGP
	g, err := cgp.ParseCGP(&DefaultConfig,
		"7N6M/5ZOON4AA/7B5UN/2S4L3LADY/2T4E2QI1I1/2A2PORN3NOR/2BICE2AA1DA1E/6GUVS1OP1F/8ET1LA1U/5J3R1E1UT/4VOTE1I1R1NE/5G1MICKIES1/6FE1T1THEW/6OR3E1XI/6OY6G DDESW??/AHIILR 299/352 0 lex America;")
	is.NoErr(err)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, g.Board(), ld)
	generator.SetGame(g.Game)
	g.RecalculateBoard()
	generator.SetPlayRecorder(TopPlayOnlyRecorder)
	elc, err := equity.NewExhaustiveLeaveCalculator("America", &DefaultConfig, "")
	is.NoErr(err)
	generator.SetEquityCalculators([]equity.EquityCalculator{
		&equity.EndgameAdjustmentCalculator{},
		elc,
		&equity.OpeningAdjustmentCalculator{}})
	generator.GenAll(tilemapping.RackFromString("DDESW??", alph), false)
	assert.Equal(t, 1, len(scoringPlays(generator.plays)))
	assert.Equal(t, 106, generator.plays[0].Score()) // hEaDW(OR)DS!
	assert.Equal(t, 0, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, "", generator.plays[0].Leave().UserVisible(alph))
	assert.Equal(t, generator.plays[0].Tiles().UserVisiblePlayedTiles(alph), "hEaDW..DS")
}

func TestGiantTwentySevenTimer(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsOxy)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(tilemapping.RackFromString("ABEOPXZ", alph), false)
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
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.UpdateAllAnchors()

	generator.GenAll(tilemapping.RackFromString("DEGORV?", alph), true)
	assert.Equal(t, 3313, len(scoringPlays(generator.plays)))
	assert.Equal(t, 127, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, 80, generator.plays[0].Score())
	assert.Equal(t, "", generator.plays[0].Leave().UserVisible(alph))
}

func TestGenerateNoPlays(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
	cross_set.GenAllCrossSets(bd, gd, ld)

	generator.GenAll(tilemapping.RackFromString("V", alph), false)
	// V won't play anywhere
	assert.Equal(t, 0, len(scoringPlays(generator.plays)))
	assert.Equal(t, 1, len(nonScoringPlays(generator.plays)))
	assert.Equal(t, move.MoveTypePass, generator.plays[0].Action())
}

func TestRowEquivalent(t *testing.T) {
	is := is.New(t)

	// XXX: This test should probably be in the board.
	gd, err := GaddagFromLexicon("NWL20")
	is.NoErr(err)
	alph := gd.GetAlphabet()

	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
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

func TestAtLeastOneTileMove(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)

	rack := tilemapping.RackFromString("AABDELT", alph)

	is.True(generator.AtLeastOneTileMove(rack))
	rackQ := tilemapping.RackFromString("Q", alph)
	is.True(!generator.AtLeastOneTileMove(rackQ))

}

func TestMaxTileUsage(t *testing.T) {
	is := is.New(t)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	cross_set.GenAllCrossSets(bd, gd, ld)

	rack := tilemapping.RackFromString("VIVIFIC", alph)
	plays := generator.GenAll(rack, false)
	// 7 orientations of VIVIFIC and 2 of IF
	is.Equal(len(plays), 9)
	generator.SetMaxTileUsage(6)
	// only allow at most 6 tiles to be used from the rack: both orientations of IF
	plays = generator.GenAll(rack, false)
	is.Equal(len(plays), 2)
}

// Note about the comments on the following benchmarks:
// The benchmarks are a bit slower now. This largely comes
// from the sorting / equity stuff that wasn't there before.

func BenchmarkGenEmptyBoard(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 1.67ms per operation

		bd := board.MakeBoard(board.CrosswordGameBoard)
		ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
		is.NoErr(err)
		generator := NewGordonGenerator(gd, bd, ld)
		bd.UpdateAllAnchors()
		generator.GenAll(tilemapping.RackFromString("AEINRST", alph), true)
	}
}

func BenchmarkGenFullRack(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Benchmark run 2022-08-09 on M1 MBP (Docker not running):
		// go 1.18

		// 2190	    513518 ns/op	  204738 B/op	    3884 allocs/op
		bd := board.MakeBoard(board.CrosswordGameBoard)
		ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
		is.NoErr(err)
		generator := NewGordonGenerator(gd, bd, ld)
		bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
		cross_set.GenAllCrossSets(bd, gd, ld)

		generator.GenAll(tilemapping.RackFromString("AABDELT", alph), true)
	}
}

func BenchmarkJustMovegen(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	generator.SetPlayRecorder(TopPlayOnlyRecorder)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)
	b.ReportAllocs()
	b.ResetTimer()
	rack := tilemapping.RackFromString("AABDELT", alph)
	for i := 0; i < b.N; i++ {
		// themonolith
		// go 1.20

		// 9517	    117747 ns/op	       0 B/op	       0 allocs/op
		generator.GenAll(rack, true)
	}
}

func BenchmarkAtLeastOneTileMove(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	is.NoErr(err)
	generator := NewGordonGenerator(gd, bd, ld)
	bd.SetToGame(gd.GetAlphabet(), board.VsMatt)
	cross_set.GenAllCrossSets(bd, gd, ld)
	b.ReportAllocs()
	b.ResetTimer()
	rack := tilemapping.RackFromString("AABDELT", alph)
	for i := 0; i < b.N; i++ {
		// themonolith
		// go 1.20

		// 2416358	       485.4 ns/op	      16 B/op	       1 allocs/op
		generator.AtLeastOneTileMove(rack)
	}
}

func BenchmarkGenOneBlank(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 5.43 ms per operation on my macbook pro.
		bd := board.MakeBoard(board.CrosswordGameBoard)
		ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
		is.NoErr(err)
		generator := NewGordonGenerator(gd, bd, ld)
		bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
		cross_set.GenAllCrossSets(bd, gd, ld)

		generator.GenAll(tilemapping.RackFromString("ADDESW?", alph), false)
	}
}

func BenchmarkGenBothBlanks(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// ~16.48ms per operation on my macbook pro.
		bd := board.MakeBoard(board.CrosswordGameBoard)
		ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
		is.NoErr(err)
		generator := NewGordonGenerator(gd, bd, ld)
		bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
		cross_set.GenAllCrossSets(bd, gd, ld)

		generator.GenAll(tilemapping.RackFromString("DDESW??", alph), false)
	}
}

func BenchmarkGenBothBlanksSmallPlayRecorder(b *testing.B) {
	is := is.New(b)

	gd, err := GaddagFromLexicon("America")
	is.NoErr(err)
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// ~16.48ms per operation on my macbook pro.
		bd := board.MakeBoard(board.CrosswordGameBoard)
		ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
		is.NoErr(err)
		generator := NewGordonGenerator(gd, bd, ld)
		generator.SetPlayRecorder(AllPlaysSmallRecorder)
		bd.SetToGame(gd.GetAlphabet(), board.VsJeremy)
		cross_set.GenAllCrossSets(bd, gd, ld)

		generator.GenAll(tilemapping.RackFromString("DDESW??", alph), false)
	}
}

func TestGenExchange(t *testing.T) {
	// rack := tilemapping.RackFromString("AAABCCD", testhelpers.EnglishAlphabet())
	rack := tilemapping.RackFromString("ABCDEF?", testhelpers.EnglishAlphabet())

	bd := board.MakeBoard(board.CrosswordGameBoard)
	gd, _ := GaddagFromLexicon("NWL20")
	ld, _ := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	gen := NewGordonGenerator(gd, bd, ld)

	gen.generateExchangeMoves(rack, 0, 0)
	assert.Equal(t, len(gen.plays), 127)
}
