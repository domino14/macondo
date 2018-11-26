package movegen

import (
	"log"
	"os"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
)

type VsWho uint8

const (
	VsEd VsWho = iota
	VsMatt
	VsJeremy
)

var LexiconDir = os.Getenv("LEXICON_DIR")

func setBoardRow(board GameBoard, rowNum int8, letters string, alph *alphabet.Alphabet) {
	// Set the row in board to the passed in letters array.
	var err error
	for idx := 0; idx < board.dim(); idx++ {
		board.squares[rowNum][idx].letter = alphabet.EmptySquareMarker
	}
	for idx, r := range letters {
		if r != ' ' {
			board.squares[rowNum][idx].letter, err = alph.Val(r)
			if err != nil {
				log.Fatalf(err.Error())
			}
		}
	}
}

func getOnlyMove(moves map[string]Move) *Move {
	for _, move := range moves {
		return &move
	}
	return nil
}

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

	generator := newGordonGenerator(gd)
	generator.board.clearAllCrosses()
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
		generator := newGordonGenerator(gd)
		generator.curAnchorCol = tc.curAnchorCol
		rack.Initialize(tc.rack, gd.GetAlphabet())
		setBoardRow(generator.board, tc.row, tc.rowString, gd.GetAlphabet())
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

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 9

	setBoardRow(generator.board, 4, "   THERMOS  A", gd.GetAlphabet())
	generator.curRowIdx = 4
	ml, _ := gd.GetAlphabet().Val('I')
	generator.board.squares[generator.curRowIdx][2].vcrossSet.clear()
	generator.board.squares[generator.curRowIdx][2].vcrossSet.set(ml)
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate HITHERMOST only
	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
	m := getOnlyMove(generator.plays)
	if m.word.UserVisible(gd.GetAlphabet()) != "HI.......T" {
		t.Errorf("Got the wrong word: %v", m.word.UserVisible(gd.GetAlphabet()))
	}
}

func setBoardToGame(generator *GordonGenerator, alph *alphabet.Alphabet,
	game VsWho) {
	// Set the board to a game
	if game == VsEd {

		// club games - 20150127vEd, beginning of turn 8
		// Quackle generates 219 total unique moves with a rack of AFGIIIS
		generator.board.setFromPlaintext(`
cesar: Turn 8
   A B C D E F G H I J K L M N O   -> cesar                    AFGIIIS   182
   ------------------------------     ed                       ADEILNV   226
 1|=     '       =       '     E| --Tracking-----------------------------------
 2|  -       "       "       - N| ?AAAAACCDDDEEIIIIKLNOOOQRRRRSTTTTUVVZ  37
 3|    -       '   '       -   d|
 4|'     -       '       -     U|
 5|        G L O W S   -       R|
 6|  "       "     P E T     " E|
 7|    '       ' F A X I N G   R|
 8|=     '     J A Y   T E E M S|
 9|    B     B O Y '       N    |
10|  " L   D O E     "     U "  |
11|    A N E W         - P I    |
12|'   M O   L E U       O N   '|
13|    E H     '   '     H E    |
14|  -       "       "       -  |
15|=     '       =       '     =|
   ------------------------------
`, alph)
	} else if game == VsMatt {
		// tourney, 2018 Lake George vs Matt G
		generator.board.setFromPlaintext(`
cesar: Turn 10
   A B C D E F G H I J K L M N O      matt g                   AEEHIIL   341
   ------------------------------  -> cesar                    AABDELT   318
 1|=     '       Z E P   F     =| --Tracking-----------------------------------
 2|  F L U K Y       R   R   -  | AEEEGHIIIILMRUUWWY  18
 3|    -     E X   ' A   U -    |
 4|'   S C A R I E S T   I     '|
 5|        -         T O T      |
 6|  "       " G O   L O     "  |
 7|    '       O R ' E T A '    | ↓
 8|=     '     J A B S   b     =|
 9|    '     Q I   '     A '    | ↓
10|  "       I   N   "   N   "  | ↓
11|      R e S P O N D - D      | ↓
12|' H O E       V       O     '| ↓
13|  E N C O M I A '     N -    | ↓
14|  -       "   T   "       -  |
15|=     V E N G E D     '     =|
   ------------------------------
`, alph)
	} else if game == VsJeremy {
		// tourney, 2018 nov Manhattan vs Jeremy H
		generator.board.setFromPlaintext(`
jeremy hall: Turn 13
   A B C D E F G H I J K L M N O   -> jeremy hall              DDESW??   299
   ------------------------------     cesar                    AHIILR    352
 1|=     '       N       '     M| --Tracking-----------------------------------
 2|  -       Z O O N "       A A| AHIILR  6
 3|    -       ' B '       - U N|
 4|'   S -       L       L A D Y|
 5|    T   -     E     Q I   I  |
 6|  " A     P O R N "     N O R|
 7|    B I C E '   A A   D A   E|
 8|=     '     G U V S   O P   F|
 9|    '       '   E T   L A   U|
10|  "       J       R   E   U T|
11|        V O T E   I - R   N E|
12|'     -   G   M I C K I E S '|
13|    -       F E ' T   T H E W|
14|  -       " O R   "   E   X I|
15|=     '     O Y       '     G|
   ------------------------------
`, alph)
	}
}

func TestUpdateAnchors(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsEd)

	generator.board.updateAllAnchors()

	if generator.board.squares[3][3].hAnchor ||
		generator.board.squares[3][3].vAnchor {
		t.Errorf("Should not be an anchor at all")
	}
	if !generator.board.squares[12][12].hAnchor ||
		!generator.board.squares[12][12].vAnchor {
		t.Errorf("Should be a two-way anchor")
	}
	if !generator.board.squares[4][3].vAnchor ||
		generator.board.squares[4][3].hAnchor {
		t.Errorf("Should be a vertical but not horizontal anchor")
	}
	// I could do more but it's all right for now?
}

func TestRowGen(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsEd)
	generator.board.updateAllAnchors()

	rack := &Rack{}
	rack.Initialize("AAEIRST", gd.GetAlphabet())
	generator.curRowIdx = 4
	generator.curAnchorCol = 8
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// Should generate AIRGLOWS and REGLOWS only
	if len(generator.plays) != 2 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 2)
	}
}

func TestRowGenWithBlanks(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsEd)
	generator.board.updateAllAnchors()

	rack := &Rack{}
	rack.Initialize("AAEIRST", gd.GetAlphabet())
	generator.curRowIdx = 4
	generator.curAnchorCol = 8
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// Should generate AIRGLOWS and REGLOWS only
	if len(generator.plays) != 2 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 2)
	}
}

func TestOtherRowGen(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsMatt)
	generator.board.updateAllAnchors()

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
	m := getOnlyMove(generator.plays)
	if m.word.UserVisible(gd.GetAlphabet()) != "A......" {
		t.Errorf("Expected proper play-through markers (A......), got %v",
			m.word.UserVisible(gd.GetAlphabet()))
	}
}

type crossSetTestCase struct {
	row      int
	col      int
	crossSet CrossSet
	dir      BoardDirection
	score    int
}

func TestGenCrossSetLoadedGame(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsMatt)
	alph := gd.GetAlphabet()
	// All horizontal for now.
	var testCases = []crossSetTestCase{
		{10, 10, crossSetFromString("E", alph), HorizontalDirection, 11},
		{2, 4, crossSetFromString("DKHLRSV", alph), HorizontalDirection, 9},
		{8, 7, crossSetFromString("S", alph), HorizontalDirection, 11},
		// suffix - no hooks:
		{12, 8, CrossSet(0), HorizontalDirection, 11},
		// prefix - no hooks:
		{3, 1, CrossSet(0), HorizontalDirection, 10},
		// prefix and suffix, no path
		{6, 8, CrossSet(0), HorizontalDirection, 5},
		// More in-between
		{2, 10, crossSetFromString("M", alph), HorizontalDirection, 2},
	}

	for _, tc := range testCases {
		generator.genCrossSet(tc.row, tc.col, tc.dir)
		if generator.board.squares[tc.row][tc.col].hcrossSet != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				tc.row, tc.col, tc.crossSet,
				generator.board.squares[tc.row][tc.col].hcrossSet)
		}
		if generator.board.squares[tc.row][tc.col].hcrossScore != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				generator.board.squares[tc.row][tc.col].hcrossScore)
		}
	}
}

type crossSetEdgeTestCase struct {
	col         int
	rowContents string
	crossSet    CrossSet
	score       int
}

func TestGenCrossSetEdges(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenerator(gd)
	alph := gd.GetAlphabet()
	var testCases = []crossSetEdgeTestCase{
		{0, " A", crossSetFromString("ABDFHKLMNPTYZ", alph), 1},
		{1, "A", crossSetFromString("ABDEGHILMNRSTWXY", alph), 1},
		{13, "              F", crossSetFromString("EIO", alph), 4},
		{14, "             F ", crossSetFromString("AE", alph), 4},
		{14, "          WECH ", crossSetFromString("T", alph), 12}, // phony!
		{14, "           ZZZ ", CrossSet(0), 30},
		{14, "       ZYZZYVA ", crossSetFromString("S", alph), 43},
		{14, "        ZYZZYV ", crossSetFromString("A", alph), 42}, // phony!
		{14, "       Z Z Y A ", crossSetFromString("ABDEGHILMNRSTWXY", alph), 1},
		{12, "       z z Y A ", crossSetFromString("E", alph), 5},
		{14, "OxYpHeNbUTAzON ", crossSetFromString("E", alph), 15},
		{6, "OXYPHE BUTAZONE", crossSetFromString("N", alph), 40},
		// Should still calculate score correctly despite no gaddag path.
		{0, " YHJKTKHKTLV", CrossSet(0), 42},
		{14, "   YHJKTKHKTLV ", CrossSet(0), 42},
		{6, "YHJKTK HKTLV", CrossSet(0), 42},
	}
	row := 4
	for _, tc := range testCases {
		setBoardRow(generator.board, int8(row), tc.rowContents, alph)
		generator.genCrossSet(row, tc.col, HorizontalDirection)
		if generator.board.squares[row][tc.col].hcrossSet != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				row, tc.col, tc.crossSet,
				generator.board.squares[row][tc.col].hcrossSet)
		}
		if generator.board.squares[row][tc.col].hcrossScore != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				row, tc.col, tc.score,
				generator.board.squares[row][tc.col].hcrossScore)
		}
	}
}

func TestGenAllCrossSets(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsEd)
	generator.genAllCrossSets()
	var testCases = []crossSetTestCase{
		{8, 8, crossSetFromString("OS", alph), HorizontalDirection, 8},
		{8, 8, crossSetFromString("S", alph), VerticalDirection, 9},
		{5, 11, crossSetFromString("S", alph), HorizontalDirection, 5},
		{5, 11, crossSetFromString("AO", alph), VerticalDirection, 2},
		{8, 13, crossSetFromString("AEOU", alph), HorizontalDirection, 1},
		{8, 13, crossSetFromString("AEIMOUY", alph), VerticalDirection, 3},
		{9, 13, crossSetFromString("HMNPST", alph), HorizontalDirection, 1},
		{9, 13, TrivialCrossSet, VerticalDirection, 0},
		{14, 14, TrivialCrossSet, HorizontalDirection, 0},
		{14, 14, TrivialCrossSet, VerticalDirection, 0},
		{12, 12, CrossSet(0), HorizontalDirection, 0},
		{12, 12, CrossSet(0), VerticalDirection, 0},
	}

	for idx, tc := range testCases {
		// Compare values
		if generator.board.getCrossSet(tc.row, tc.col, tc.dir) != tc.crossSet {
			t.Errorf("Test=%v For row=%v col=%v, Expected cross-set to be %v, got %v",
				idx, tc.row, tc.col, tc.crossSet,
				generator.board.getCrossSet(tc.row, tc.col, tc.dir))
		}
		if generator.board.getCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				generator.board.getCrossScore(tc.row, tc.col, tc.dir))
		}
	}
	// This one has more nondeterministic (in-between LR) crosssets
	setBoardToGame(generator, gd.GetAlphabet(), VsMatt)
	generator.genAllCrossSets()
	testCases = []crossSetTestCase{
		{8, 7, crossSetFromString("S", alph), HorizontalDirection, 11},
		{8, 7, CrossSet(0), VerticalDirection, 12},
		{5, 11, crossSetFromString("BGOPRTWX", alph), HorizontalDirection, 2},
		{5, 11, CrossSet(0), VerticalDirection, 15},
		{8, 13, TrivialCrossSet, HorizontalDirection, 0},
		{8, 13, TrivialCrossSet, VerticalDirection, 0},
		{11, 4, crossSetFromString("DRS", alph), HorizontalDirection, 6},
		{11, 4, crossSetFromString("CGM", alph), VerticalDirection, 1},
		{2, 2, TrivialCrossSet, HorizontalDirection, 0},
		{2, 2, crossSetFromString("AEI", alph), VerticalDirection, 2},
		{7, 12, crossSetFromString("AEIOY", alph), HorizontalDirection, 0}, // it's a blank
		{7, 12, TrivialCrossSet, VerticalDirection, 0},
		{11, 8, CrossSet(0), HorizontalDirection, 4},
		{11, 8, crossSetFromString("AEOU", alph), VerticalDirection, 1},
		{1, 8, crossSetFromString("AEO", alph), HorizontalDirection, 1},
		{1, 8, crossSetFromString("DFHLMNRSTX", alph), VerticalDirection, 1},
	}
	for idx, tc := range testCases {
		// Compare values
		if generator.board.getCrossSet(tc.row, tc.col, tc.dir) != tc.crossSet {
			t.Errorf("Test=%v For row=%v col=%v, Expected cross-set to be %v, got %v",
				idx, tc.row, tc.col, tc.crossSet,
				generator.board.getCrossSet(tc.row, tc.col, tc.dir))
		}
		if generator.board.getCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				generator.board.getCrossScore(tc.row, tc.col, tc.dir))
		}
	}
}

func TestGenMoveJustOnce(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenerator(gd)
	setBoardToGame(generator, alph, VsMatt)
	generator.board.updateAllAnchors()
	generator.genAllCrossSets()
	generator.board.transpose()

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

	generator := newGordonGenerator(gd)
	setBoardToGame(generator, alph, VsMatt)
	generator.board.updateAllAnchors()
	generator.genAllCrossSets()
	generator.GenAll("A")
	if len(generator.plays) != 24 {
		t.Errorf("Expected %v, got %v (%v) plays", 24, generator.plays,
			len(generator.plays))
	}
}

func TestGenAllMovesFullRack(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenerator(gd)
	setBoardToGame(generator, alph, VsMatt)
	generator.board.updateAllAnchors()
	generator.genAllCrossSets()
	generator.GenAll("AABDELT")
	// There should be 673 unique plays
	if len(generator.plays) != 673 {
		t.Errorf("Expected %v, got %v (%v) plays", 673, generator.plays,
			len(generator.plays))
	}
}

func TestGenAllMovesFullRackAgain(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := newGordonGenerator(gd)
	setBoardToGame(generator, alph, VsEd)
	generator.board.updateAllAnchors()
	generator.genAllCrossSets()
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

	generator := newGordonGenerator(gd)
	setBoardToGame(generator, alph, VsEd)
	generator.board.updateAllAnchors()
	generator.genAllCrossSets()
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

	generator := newGordonGenerator(gd)
	setBoardToGame(generator, alph, VsEd)
	generator.board.updateAllAnchors()
	generator.genAllCrossSets()
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

	generator := newGordonGenerator(gd)
	setBoardToGame(generator, alph, VsJeremy)
	generator.board.updateAllAnchors()
	generator.genAllCrossSets()
	generator.GenAll("DDESW??")
	// If I do DDESW? in quackle i generate 1483 moves. My movegen generates
	// 1586, possibly by the same logic as the above.
	// If I add 103 to the Quackle-generated 8194 moves for both blanks (DDESW??)
	// I get 8297, so there should be 8297 unique plays
	if len(generator.plays) != 8297 {
		t.Errorf("Expected %v, got %v (%v) plays", 8297, generator.plays,
			len(generator.plays))
	}
}

func BenchmarkGenFullRack(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 4.1 ms per operation on my macbook pro!! not bad!!
		generator := newGordonGenerator(gd)
		setBoardToGame(generator, alph, VsMatt)
		generator.board.updateAllAnchors()
		generator.genAllCrossSets()
		generator.GenAll("AABDELT")
	}
}

func BenchmarkGenOneBlank(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 26.1 ms per operation on my macbook pro.
		generator := newGordonGenerator(gd)
		setBoardToGame(generator, alph, VsJeremy)
		generator.board.updateAllAnchors()
		generator.genAllCrossSets()
		generator.GenAll("ADDESW?")
	}
}

func BenchmarkGenBothBlanks(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// ~81-85 ms per operation on my macbook pro.
		generator := newGordonGenerator(gd)
		setBoardToGame(generator, alph, VsJeremy)
		generator.board.updateAllAnchors()
		generator.genAllCrossSets()
		generator.GenAll("DDESW??")
	}
}
