package movegen

import (
	"log"
	"os"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
)

type VsWho uint8

const (
	VsEd VsWho = iota
	VsMatt
	VsJeremy
	VsOxy
)

var LexiconDir = os.Getenv("LEXICON_DIR")

func setBoardRow(board board.GameBoard, rowNum int8, letters string, alph *alphabet.Alphabet) {
	// Set the row in board to the passed in letters array.
	for idx := 0; idx < board.Dim(); idx++ {
		board.SetLetter(int(rowNum), idx, alphabet.EmptySquareMarker)
	}
	for idx, r := range letters {
		if r != ' ' {
			letter, err := alph.Val(r)
			if err != nil {
				log.Fatalf(err.Error())
			}
			board.SetLetter(int(rowNum), idx, letter)

		}
	}
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

	generator := NewGordonGenerator(gd)
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
		generator := NewGordonGenerator(gd)
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

	generator := NewGordonGenerator(gd)
	generator.curAnchorCol = 9

	setBoardRow(generator.board, 4, "   THERMOS  A", gd.GetAlphabet())
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
		generator.board.SetFromPlaintext(`
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
		generator.board.SetFromPlaintext(`
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
		generator.board.SetFromPlaintext(`
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
	} else if game == VsOxy {
		// lol
		generator.board.SetFromPlaintext(`
cesar: Turn 11
   A B C D E F G H I J K L M N O      rubin                    ADDELOR   345
   ------------------------------  -> cesar                    OXPBAZE   129
 1|= P A C I F Y I N G   '     =| --Tracking-----------------------------------
 2|  I S     "       "       -  | ADDELORRRTVV  12
 3|Y E -       '   '       -    |
 4|' R E Q U A L I F I E D     '|
 5|H   L   -           -        |
 6|E D S     "       "       "  |
 7|N O '     T '   '       '    |
 8|= R A I N W A S H I N G     =|
 9|U M '     O '   '       '    |
10|T "   E   O       "       "  |
11|  W A K E n E R S   -        |
12|' O n E T I M E       -     '|
13|O O T     E ' B '       -    |
14|N -       "   U   "       -  |
15|= J A C U L A T I N G '     =|
   ------------------------------
`, alph)
	}
}

func TestUpdateAnchors(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsEd)

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
	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsEd)
	generator.board.UpdateAllAnchors()

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
	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsEd)
	generator.board.UpdateAllAnchors()

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
	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsMatt)
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
	if m.word.UserVisible(gd.GetAlphabet()) != "A......" {
		t.Errorf("Expected proper play-through markers (A......), got %v",
			m.word.UserVisible(gd.GetAlphabet()))
	}
}

type crossSetTestCase struct {
	row      int
	col      int
	crossSet board.CrossSet
	dir      board.BoardDirection
	score    int
}

func TestGenCrossSetLoadedGame(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsMatt)
	alph := gd.GetAlphabet()
	// All horizontal for now.
	var testCases = []crossSetTestCase{
		{10, 10, board.CrossSetFromString("E", alph), board.HorizontalDirection, 11},
		{2, 4, board.CrossSetFromString("DKHLRSV", alph), board.HorizontalDirection, 9},
		{8, 7, board.CrossSetFromString("S", alph), board.HorizontalDirection, 11},
		// suffix - no hooks:
		{12, 8, board.CrossSet(0), board.HorizontalDirection, 11},
		// prefix - no hooks:
		{3, 1, board.CrossSet(0), board.HorizontalDirection, 10},
		// prefix and suffix, no path
		{6, 8, board.CrossSet(0), board.HorizontalDirection, 5},
		// More in-between
		{2, 10, board.CrossSetFromString("M", alph), board.HorizontalDirection, 2},
	}

	for _, tc := range testCases {
		generator.board.GenCrossSet(tc.row, tc.col, tc.dir, generator.gaddag,
			generator.bag)
		if generator.board.GetCrossSet(tc.row, tc.col, board.HorizontalDirection) != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				tc.row, tc.col, tc.crossSet,
				generator.board.GetCrossSet(tc.row, tc.col, board.HorizontalDirection))
		}
		if generator.board.GetCrossScore(tc.row, tc.col, board.HorizontalDirection) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				generator.board.GetCrossScore(tc.row, tc.col, board.HorizontalDirection))
		}
	}
}

type crossSetEdgeTestCase struct {
	col         int
	rowContents string
	crossSet    board.CrossSet
	score       int
}

func TestGenCrossSetEdges(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := NewGordonGenerator(gd)
	alph := gd.GetAlphabet()
	var testCases = []crossSetEdgeTestCase{
		{0, " A", board.CrossSetFromString("ABDFHKLMNPTYZ", alph), 1},
		{1, "A", board.CrossSetFromString("ABDEGHILMNRSTWXY", alph), 1},
		{13, "              F", board.CrossSetFromString("EIO", alph), 4},
		{14, "             F ", board.CrossSetFromString("AE", alph), 4},
		{14, "          WECH ", board.CrossSetFromString("T", alph), 12}, // phony!
		{14, "           ZZZ ", board.CrossSet(0), 30},
		{14, "       ZYZZYVA ", board.CrossSetFromString("S", alph), 43},
		{14, "        ZYZZYV ", board.CrossSetFromString("A", alph), 42}, // phony!
		{14, "       Z Z Y A ", board.CrossSetFromString("ABDEGHILMNRSTWXY", alph), 1},
		{12, "       z z Y A ", board.CrossSetFromString("E", alph), 5},
		{14, "OxYpHeNbUTAzON ", board.CrossSetFromString("E", alph), 15},
		{6, "OXYPHE BUTAZONE", board.CrossSetFromString("N", alph), 40},
		// Should still calculate score correctly despite no gaddag path.
		{0, " YHJKTKHKTLV", board.CrossSet(0), 42},
		{14, "   YHJKTKHKTLV ", board.CrossSet(0), 42},
		{6, "YHJKTK HKTLV", board.CrossSet(0), 42},
	}
	row := 4
	for _, tc := range testCases {
		setBoardRow(generator.board, int8(row), tc.rowContents, alph)
		generator.board.GenCrossSet(row, tc.col, board.HorizontalDirection,
			generator.gaddag, generator.bag)
		if generator.board.GetCrossSet(row, tc.col, board.HorizontalDirection) != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				row, tc.col, tc.crossSet,
				generator.board.GetCrossSet(row, tc.col, board.HorizontalDirection))
		}
		if generator.board.GetCrossScore(row, tc.col, board.HorizontalDirection) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				row, tc.col, tc.score,
				generator.board.GetCrossScore(row, tc.col, board.HorizontalDirection))
		}
	}
}

func TestGenAllCrossSets(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsEd)
	generator.board.GenAllCrossSets(gd, generator.bag)
	var testCases = []crossSetTestCase{
		{8, 8, board.CrossSetFromString("OS", alph), board.HorizontalDirection, 8},
		{8, 8, board.CrossSetFromString("S", alph), board.VerticalDirection, 9},
		{5, 11, board.CrossSetFromString("S", alph), board.HorizontalDirection, 5},
		{5, 11, board.CrossSetFromString("AO", alph), board.VerticalDirection, 2},
		{8, 13, board.CrossSetFromString("AEOU", alph), board.HorizontalDirection, 1},
		{8, 13, board.CrossSetFromString("AEIMOUY", alph), board.VerticalDirection, 3},
		{9, 13, board.CrossSetFromString("HMNPST", alph), board.HorizontalDirection, 1},
		{9, 13, board.TrivialCrossSet, board.VerticalDirection, 0},
		{14, 14, board.TrivialCrossSet, board.HorizontalDirection, 0},
		{14, 14, board.TrivialCrossSet, board.VerticalDirection, 0},
		{12, 12, board.CrossSet(0), board.HorizontalDirection, 0},
		{12, 12, board.CrossSet(0), board.VerticalDirection, 0},
	}

	for idx, tc := range testCases {
		// Compare values
		if generator.board.GetCrossSet(tc.row, tc.col, tc.dir) != tc.crossSet {
			t.Errorf("Test=%v For row=%v col=%v, Expected cross-set to be %v, got %v",
				idx, tc.row, tc.col, tc.crossSet,
				generator.board.GetCrossSet(tc.row, tc.col, tc.dir))
		}
		if generator.board.GetCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				generator.board.GetCrossScore(tc.row, tc.col, tc.dir))
		}
	}
	// This one has more nondeterministic (in-between LR) crosssets
	setBoardToGame(generator, gd.GetAlphabet(), VsMatt)
	generator.board.GenAllCrossSets(gd, generator.bag)
	testCases = []crossSetTestCase{
		{8, 7, board.CrossSetFromString("S", alph), board.HorizontalDirection, 11},
		{8, 7, board.CrossSet(0), board.VerticalDirection, 12},
		{5, 11, board.CrossSetFromString("BGOPRTWX", alph), board.HorizontalDirection, 2},
		{5, 11, board.CrossSet(0), board.VerticalDirection, 15},
		{8, 13, board.TrivialCrossSet, board.HorizontalDirection, 0},
		{8, 13, board.TrivialCrossSet, board.VerticalDirection, 0},
		{11, 4, board.CrossSetFromString("DRS", alph), board.HorizontalDirection, 6},
		{11, 4, board.CrossSetFromString("CGM", alph), board.VerticalDirection, 1},
		{2, 2, board.TrivialCrossSet, board.HorizontalDirection, 0},
		{2, 2, board.CrossSetFromString("AEI", alph), board.VerticalDirection, 2},
		{7, 12, board.CrossSetFromString("AEIOY", alph), board.HorizontalDirection, 0}, // it's a blank
		{7, 12, board.TrivialCrossSet, board.VerticalDirection, 0},
		{11, 8, board.CrossSet(0), board.HorizontalDirection, 4},
		{11, 8, board.CrossSetFromString("AEOU", alph), board.VerticalDirection, 1},
		{1, 8, board.CrossSetFromString("AEO", alph), board.HorizontalDirection, 1},
		{1, 8, board.CrossSetFromString("DFHLMNRSTX", alph), board.VerticalDirection, 1},
	}
	for idx, tc := range testCases {
		// Compare values
		if generator.board.GetCrossSet(tc.row, tc.col, tc.dir) != tc.crossSet {
			t.Errorf("Test=%v For row=%v col=%v, Expected cross-set to be %v, got %v",
				idx, tc.row, tc.col, tc.crossSet,
				generator.board.GetCrossSet(tc.row, tc.col, tc.dir))
		}
		if generator.board.GetCrossScore(tc.row, tc.col, tc.dir) != tc.score {
			t.Errorf("For row=%v col=%v, Expected cross-score to be %v, got %v",
				tc.row, tc.col, tc.score,
				generator.board.GetCrossScore(tc.row, tc.col, tc.dir))
		}
	}
}

func TestGenMoveJustOnce(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, alph, VsMatt)
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

	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, alph, VsMatt)
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

	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, alph, VsMatt)
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
		if generator.plays[idx].score != score {
			t.Errorf("Expected %v, got %v", score, generator.plays[idx].score)
		}
	}
}

func TestGenAllMovesFullRackAgain(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, alph, VsEd)
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

	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, alph, VsEd)
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

	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, alph, VsEd)
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

	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, alph, VsJeremy)
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
	if generator.plays[0].score != 106 { // hEaDW(OR)DS!
		t.Errorf("Expected %v, got %v", 106, generator.plays[0].score)
	}
}

func TestGiantTwentySevenTimer(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()

	generator := NewGordonGenerator(gd)
	setBoardToGame(generator, alph, VsOxy)
	generator.board.UpdateAllAnchors()
	generator.board.GenAllCrossSets(gd, generator.bag)
	generator.GenAll("ABEOPXZ")
	if len(generator.plays) != 519 {
		t.Errorf("Expected %v, got %v (%v) plays", 519, generator.plays,
			len(generator.plays))
	}
	if generator.plays[0].score != 1780 { // oxyphenbutazone
		t.Errorf("Expected %v, got %v", 1780, generator.plays[0].score)
	}
}

func TestGenerateEmptyBoard(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := NewGordonGenerator(gd)
	generator.board.UpdateAllAnchors()
	generator.GenAll("DEGORV?")
	if len(generator.plays) != 3313 {
		t.Errorf("Expected %v, got %v (%v) plays", 3313, generator.plays,
			len(generator.plays))
	}
	if generator.plays[0].score != 80 { // overdog
		t.Errorf("Expected %v, got %v", 80, generator.plays[0].score)
	}
}

func BenchmarkGenEmptyBoard(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 1.67ms per operation
		generator := NewGordonGenerator(gd)
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
		generator := NewGordonGenerator(gd)
		setBoardToGame(generator, alph, VsMatt)
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
		generator := NewGordonGenerator(gd)
		setBoardToGame(generator, alph, VsJeremy)
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
		generator := NewGordonGenerator(gd)
		setBoardToGame(generator, alph, VsJeremy)
		generator.board.UpdateAllAnchors()
		generator.board.GenAllCrossSets(gd, generator.bag)
		generator.GenAll("DDESW??")
	}
}
