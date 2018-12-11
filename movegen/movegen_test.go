package movegen

import (
	"log"
	"os"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/move"
)

var LexiconDir = os.Getenv("LEXICON_DIR")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/gen_america.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(LexiconDir+"America.txt", true, true)
		os.Rename("out.gaddag", "/tmp/gen_america.gaddag")
	}
	os.Exit(m.Run())
}

// This is going to be a big file; it tests the main move generation
// recursive algorithm

func TestGenBase(t *testing.T) {
	// Sanity check. A board with no cross checks should generate nothing.
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("AEINRST", gd.GetAlphabet())

	generator := newGordonGenHardcode(gd)
	generator.board.ClearAllCrosses()
	generator.curAnchorCol = 8
	// Row 4 for shiz and gig
	generator.curRowIdx = 4
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())

	if len(generator.plays) != 0 {
		t.Errorf("Generated %v plays, expected %v", len(generator.plays), 0)
	}
}

type SimpleGenTestCase struct {
	rack          string
	curAnchorCol  int8
	row           int8
	rowString     string
	expectedPlays int
}

func TestSimpleRowGen(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")

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
	for idx, tc := range cases {
		log.Printf("Case %v", idx)
		generator := newGordonGenHardcode(gd)
		generator.curAnchorCol = tc.curAnchorCol
		rack.Initialize(tc.rack, gd.GetAlphabet())
		generator.board.SetRow(tc.row, tc.rowString, gd.GetAlphabet())
		generator.curRowIdx = tc.row
		generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
			gd.GetRootNodeIndex())
		if len(generator.plays) != tc.expectedPlays {
			t.Errorf("%v Generated %v plays, expected %v", idx, generator.plays, tc.expectedPlays)
		}
	}
}

func TestGenThroughBothWaysAllowedLetters(t *testing.T) {
	// Basically, allow HITHERMOST but not NETHERMOST.
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("ABEHINT", gd.GetAlphabet())

	generator := newGordonGenHardcode(gd)
	generator.curAnchorCol = 9

	generator.board.SetRow(4, "   THERMOS  A", gd.GetAlphabet())
	generator.curRowIdx = 4
	ml, _ := gd.GetAlphabet().Val('I')
	generator.board.ClearCrossSet(int(generator.curRowIdx), 2, board.VerticalDirection)
	generator.board.SetCrossSetLetter(int(generator.curRowIdx), 2, board.VerticalDirection, ml)
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
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

func TestUpdateAnchors(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(gd.GetAlphabet(), board.VsEd)

	generator.board.UpdateAllAnchors()

	if generator.board.IsAnchor(3, 3, board.HorizontalDirection) ||
		generator.board.IsAnchor(3, 3, board.VerticalDirection) {
		t.Errorf("Should not be an anchor at all")
	}
	if !generator.board.IsAnchor(12, 12, board.HorizontalDirection) ||
		!generator.board.IsAnchor(12, 12, board.VerticalDirection) {
		t.Errorf("Should be a two-way anchor")
	}
	if !generator.board.IsAnchor(4, 3, board.VerticalDirection) ||
		generator.board.IsAnchor(4, 3, board.HorizontalDirection) {
		t.Errorf("Should be a vertical but not horizontal anchor")
	}
	// I could do more but it's all right for now?
}

func TestRowGen(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(gd.GetAlphabet(), board.VsEd)
	generator.board.UpdateAllAnchors()

	rack := &Rack{}
	rack.Initialize("AAEIRST", gd.GetAlphabet())
	generator.curRowIdx = 4
	generator.curAnchorCol = 8
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
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
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(gd.GetAlphabet(), board.VsMatt)
	generator.board.UpdateAllAnchors()

	rack := &Rack{}
	rack.Initialize("A", gd.GetAlphabet())
	generator.curRowIdx = 14
	generator.curAnchorCol = 8
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
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
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(alph, board.VsMatt)
	generator.board.UpdateAllAnchors()
	generator.board.GenAllCrossSets(gd, generator.bag)
	generator.board.Transpose()

	rack := &Rack{}
	rack.Initialize("AELT", gd.GetAlphabet())
	// We want to generate TAEL parallel to ABANDON (making RESPONDED)
	// See VsMatt board definition above.
	generator.curRowIdx = 10
	generator.vertical = true
	generator.lastAnchorCol = 100
	for anchorCol := int8(8); anchorCol <= int8(12); anchorCol++ {
		generator.curAnchorCol = anchorCol
		generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
			gd.GetRootNodeIndex())
		generator.lastAnchorCol = anchorCol
	}
	if len(generator.plays) != 34 {
		t.Errorf("Expected %v, got %v plays", 34, generator.plays)
	}
}

func TestGenAllMovesSingleTile(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(alph, board.VsMatt)
	generator.board.UpdateAllAnchors()
	generator.board.GenAllCrossSets(gd, generator.bag)
	generator.GenAll("A")
	if len(generator.plays) != 24 {
		t.Errorf("Expected %v, got %v (%v) plays", 24, generator.plays,
			len(generator.plays))
	}
}

func TestGenAllMovesFullRack(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(alph, board.VsMatt)
	generator.board.UpdateAllAnchors()
	generator.board.GenAllCrossSets(gd, generator.bag)
	generator.GenAll("AABDELT")
	// There should be 673 unique plays
	if len(generator.plays) != 673 {
		t.Errorf("Expected %v, got %v (%v) plays", 673, generator.plays,
			len(generator.plays))
	}
	highestScores := []int{38, 36, 36, 34, 34, 33, 30, 30, 30, 28}
	for idx, score := range highestScores {
		if generator.plays[idx].Score() != score {
			t.Errorf("Expected %v, got %v", score, generator.plays[idx].Score())
		}
	}
}

func TestGenAllMovesFullRackAgain(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(alph, board.VsEd)
	generator.board.UpdateAllAnchors()
	generator.board.GenAllCrossSets(gd, generator.bag)
	generator.GenAll("AFGIIIS")
	// There should be 219 unique plays
	if len(generator.plays) != 219 {
		t.Errorf("Expected %v, got %v (%v) plays", 219, generator.plays,
			len(generator.plays))
	}
}

func TestGenAllMovesSingleBlank(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(alph, board.VsEd)
	generator.board.UpdateAllAnchors()
	generator.board.GenAllCrossSets(gd, generator.bag)
	generator.GenAll("?")
	// There should be 166 unique plays. Quackle does not generate all blank
	// plays, even when told to generate all plays!!
	if len(generator.plays) != 166 {
		t.Errorf("Expected %v, got %v (%v) plays", 166, generator.plays,
			len(generator.plays))
	}
}
func TestGenAllMovesTwoBlanksOnly(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(alph, board.VsEd)
	generator.board.UpdateAllAnchors()
	generator.board.GenAllCrossSets(gd, generator.bag)
	generator.GenAll("??")
	// Quackle generates 1827 unique plays. (my movegen generates 1958)
	// With one blank (the test above), Quackle generates 35 moves, I generate
	// 166 by hand. The difference is 131. It seems Quackle does not generate
	// all plays for one blank, only the first one alphabetically for every position.
	// The difference between 1827 and 1958 is also 131, so I think this is
	// ok.
	if len(generator.plays) != 1958 {
		t.Errorf("Expected %v, got %v (%v) plays", 1958, generator.plays,
			len(generator.plays))
	}
}

func TestGenAllMovesWithBlanks(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(gd.GetAlphabet(), board.VsJeremy)
	generator.board.UpdateAllAnchors()
	generator.board.GenAllCrossSets(gd, generator.bag)
	generator.GenAll("DDESW??")
	// If I do DDESW? in quackle i generate 1483 moves. My movegen generates
	// 1586, possibly by the same logic as the above.
	// If I add 103 to the Quackle-generated 8194 moves for both blanks (DDESW??)
	// I get 8297, so there should be 8297 unique plays
	if len(generator.plays) != 8297 {
		t.Errorf("Expected %v, got %v (%v) plays", 8297, generator.plays,
			len(generator.plays))
	}
	if generator.plays[0].Score() != 106 { // hEaDW(OR)DS!
		t.Errorf("Expected %v, got %v", 106, generator.plays[0].Score())
	}
	if generator.plays[0].Leave().UserVisible(alph) != "" {
		t.Errorf("Expected bingo leave to be empty!")
	}
	if generator.plays[1].Leave().UserVisible(alph) != "S" {
		t.Errorf("Expected second-highest play to keep an S, leave was: %v",
			generator.plays[1].Leave().UserVisible(alph))
	}
	// There are 7 plays worth 32 pts.
	rewards := 0
	for i := 2; i < 9; i++ {
		if generator.plays[i].Score() != 32 {
			t.Errorf("Expected play to be worth 32 pts.")
		}
		if generator.plays[i].Tiles().UserVisible(alph) == "rEW..DS" {
			rewards = i
		}
	}
	if rewards == 0 {
		t.Errorf("shoulda found rewards")
	}
	if generator.plays[rewards].Leave().UserVisible(alph) != "D?" {
		t.Errorf("Leave was wrong, got %v",
			generator.plays[rewards].Leave().UserVisible(alph))
	}
}

func TestGiantTwentySevenTimer(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(gd.GetAlphabet(), board.VsOxy)
	generator.board.UpdateAllAnchors()
	generator.board.GenAllCrossSets(gd, generator.bag)
	generator.GenAll("ABEOPXZ")
	if len(generator.plays) != 519 {
		t.Errorf("Expected %v, got %v (%v) plays", 519, generator.plays,
			len(generator.plays))
	}
	if generator.plays[0].Score() != 1780 { // oxyphenbutazone
		t.Errorf("Expected %v, got %v", 1780, generator.plays[0].Score())
	}
	if generator.plays[0].Leave().UserVisible(alph) != "" {
		t.Errorf("Expected bingo leave to be empty!")
	}
}

func TestGenerateEmptyBoard(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenHardcode(gd)
	generator.board.UpdateAllAnchors()
	generator.GenAll("DEGORV?")
	if len(generator.plays) != 3313 {
		t.Errorf("Expected %v, got %v (%v) plays", 3313, generator.plays,
			len(generator.plays))
	}
	if generator.plays[0].Score() != 80 { // overdog
		t.Errorf("Expected %v, got %v", 80, generator.plays[0].Score())
	}
	if generator.plays[0].Leave().UserVisible(gd.GetAlphabet()) != "" {
		t.Errorf("Expected bingo leave to be empty!")
	}
}

func TestGenerateNoPlays(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenHardcode(gd)
	generator.board.SetBoardToGame(alph, board.VsJeremy)
	generator.board.UpdateAllAnchors()
	generator.board.GenAllCrossSets(gd, generator.bag)
	generator.GenAll("V")
	// V won't play anywhere
	if len(generator.plays) != 1 {
		t.Errorf("Expected %v, got %v (%v) plays", 1, generator.plays,
			len(generator.plays))
	}
	if generator.plays[0].Action() != move.MoveTypePass {
		t.Errorf("Expected %v, got %v", move.MoveTypePass, generator.plays[0].Action())
	}

}

func BenchmarkGenEmptyBoard(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 1.67ms per operation
		generator := newGordonGenHardcode(gd)
		generator.board.UpdateAllAnchors()
		generator.GenAll("AEINRST")
	}
}

func BenchmarkGenFullRack(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 930 μs per operation on my macbook pro!! amazing!!!
		generator := newGordonGenHardcode(gd)
		generator.board.SetBoardToGame(alph, board.VsMatt)
		generator.board.UpdateAllAnchors()
		generator.board.GenAllCrossSets(gd, generator.bag)
		generator.GenAll("AABDELT")
	}
}

func BenchmarkGenOneBlank(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 5.43 ms per operation on my macbook pro.
		generator := newGordonGenHardcode(gd)
		generator.board.SetBoardToGame(alph, board.VsJeremy)
		generator.board.UpdateAllAnchors()
		generator.board.GenAllCrossSets(gd, generator.bag)
		generator.GenAll("ADDESW?")
	}
}

func BenchmarkGenBothBlanks(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// ~16.48ms per operation on my macbook pro.
		generator := newGordonGenHardcode(gd)
		generator.board.SetBoardToGame(alph, board.VsJeremy)
		generator.board.UpdateAllAnchors()
		generator.board.GenAllCrossSets(gd, generator.bag)
		generator.GenAll("DDESW??")
	}
}
func BenchmarkGenCrossSets(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	b.ResetTimer()
	// 159 us
	for i := 0; i < b.N; i++ {
		generator := newGordonGenHardcode(gd)
		generator.board.SetBoardToGame(alph, board.VsOxy)
		generator.board.UpdateAllAnchors()
		generator.board.GenAllCrossSets(gd, generator.bag)
	}
}
