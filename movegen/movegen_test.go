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
)

var LexiconDir = os.Getenv("LEXICON_DIR")

func setBoardRow(board GameBoard, rowNum int8, letters string, alph *alphabet.Alphabet) {
	// Set the row in board to the passed in letters array.
	var err error
	for idx := 0; idx < board.dim(); idx++ {
		board.squares[rowNum][idx].letter = EmptySquareMarker
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
	generator.curRow = generator.board.squares[4]
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
		{"ABEHITT", 8, 4, "  THERMOS A   ", 1},   // thermostat
		{"TT", 10, 4, "  THERMOS A   ", 3},       // thermostat, at, att
	}
	for _, tc := range cases {
		generator := newGordonGenerator(gd)
		generator.curAnchorCol = tc.curAnchorCol
		rack.Initialize(tc.rack, gd.GetAlphabet())
		setBoardRow(generator.board, tc.row, tc.rowString, gd.GetAlphabet())
		generator.curRow = generator.board.squares[tc.row]
		generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
			gd.GetRootNodeIndex())
		if len(generator.plays) != tc.expectedPlays {
			t.Errorf("Generated %v plays, expected %v", generator.plays, tc.expectedPlays)
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
	generator.curRow = generator.board.squares[4]
	ml, _ := gd.GetAlphabet().Val('I')
	generator.curRow[2].vcrossSet.clear()
	generator.curRow[2].vcrossSet.set(ml)
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate HITHERMOST only
	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
	if generator.plays[0].word.UserVisible(gd.GetAlphabet()) != "HITHERMOST" {
		t.Errorf("Got the wrong word.")
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
 7|    '       O R ' E T A '    |
 8|=     '     J A B S   b     =|
 9|    '     Q I   '     A '    | 1
10|  "       I   N   "   N   "  | 1
11|      R e S P O N D - D      | 11
12|' H O E       V       O     '| 1
13|  E N C O M I A '     N -    |
14|  -       "   T   "       -  |
15|=     V E N G E D     '     =|
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
	generator.curRow = generator.board.squares[4]
	generator.curAnchorCol = 8
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// Should generate AIRGLOWS and REGLOWS only
	if len(generator.plays) != 2 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 2)
	}
}

type crossSetTestCase struct {
	row      int
	col      int
	crossSet CrossSet
	dir      BoardDirection
}

func TestGenCrossSetLoadedGame(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsMatt)
	alph := gd.GetAlphabet()
	// All vertical for now.
	var testCases = []crossSetTestCase{
		{10, 10, crossSetFromString("E", alph), VerticalDirection},
		{2, 4, crossSetFromString("DKHLRSV", alph), VerticalDirection},
		{8, 7, crossSetFromString("S", alph), VerticalDirection},
		// suffix - no hooks:
		{12, 8, crossSetFromString("", alph), VerticalDirection},
		// prefix - no hooks:
		{3, 1, crossSetFromString("", alph), VerticalDirection},
		// prefix and suffix, no path
		{6, 8, crossSetFromString("", alph), VerticalDirection},
		// More in-between
		{2, 10, crossSetFromString("M", alph), VerticalDirection},
	}

	for _, tc := range testCases {
		generator.genCrossSet(tc.row, tc.col, tc.dir)
		if generator.board.squares[tc.row][tc.col].vcrossSet != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				tc.row, tc.col, tc.crossSet,
				generator.board.squares[tc.row][tc.col].vcrossSet)
		}
	}
}

type crossSetEdgeTestCase struct {
	col         int
	rowContents string
	crossSet    CrossSet
}

func TestGenCrossSetEdges(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenerator(gd)
	alph := gd.GetAlphabet()
	var testCases = []crossSetEdgeTestCase{
		{0, " A", crossSetFromString("ABDFHKLMNPTYZ", alph)},
		{1, "A", crossSetFromString("ABDEGHILMNRSTWXY", alph)},
		{13, "              F", crossSetFromString("EIO", alph)},
		{14, "             F ", crossSetFromString("AE", alph)},
		{14, "          WECH ", crossSetFromString("T", alph)}, // phony!
		{14, "           ZZZ ", crossSetFromString("", alph)},
		{14, "       ZYZZYVA ", crossSetFromString("S", alph)},
		{14, "        ZYZZYV ", crossSetFromString("A", alph)}, // phony!
		{14, "       Z Z Y A ", crossSetFromString("ABDEGHILMNRSTWXY", alph)},
		{12, "       z z Y A ", crossSetFromString("E", alph)},
		{14, "OxYpHeNbUTAzON ", crossSetFromString("E", alph)},
		{6, "OXYPHE BUTAZONE", crossSetFromString("N", alph)},
	}
	row := 4
	for _, tc := range testCases {
		setBoardRow(generator.board, int8(row), tc.rowContents, alph)
		generator.genCrossSet(row, tc.col, VerticalDirection)
		if generator.board.squares[row][tc.col].vcrossSet != tc.crossSet {
			t.Errorf("For row=%v col=%v, Expected cross-set to be %v, got %v",
				row, tc.col, tc.crossSet,
				generator.board.squares[row][tc.col].vcrossSet)
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
		{8, 8, crossSetFromString("OS", alph), VerticalDirection},
		{8, 8, crossSetFromString("S", alph), HorizontalDirection},
		{5, 11, crossSetFromString("S", alph), VerticalDirection},
		{5, 11, crossSetFromString("AO", alph), HorizontalDirection},
		{8, 13, crossSetFromString("AEOU", alph), VerticalDirection},
		{8, 13, crossSetFromString("AEIMOUY", alph), HorizontalDirection},
		{9, 13, crossSetFromString("HMNPST", alph), VerticalDirection},
		{9, 13, TrivialCrossSet, HorizontalDirection},
		{14, 14, TrivialCrossSet, VerticalDirection},
		{14, 14, TrivialCrossSet, HorizontalDirection},
		{12, 12, CrossSet(0), VerticalDirection},
		{12, 12, CrossSet(0), HorizontalDirection},
	}

	for idx, tc := range testCases {
		// Compare values
		if generator.board.getCrossSet(tc.row, tc.col, tc.dir) != tc.crossSet {
			t.Errorf("Test=%v For row=%v col=%v, Expected cross-set to be %v, got %v",
				idx, tc.row, tc.col, tc.crossSet,
				generator.board.getCrossSet(tc.row, tc.col, tc.dir))
		}
	}
	// This one has more nondeterministic (in-between LR) crosssets
	setBoardToGame(generator, gd.GetAlphabet(), VsMatt)
	generator.genAllCrossSets()
	testCases = []crossSetTestCase{
		{8, 7, crossSetFromString("S", alph), VerticalDirection},
		{8, 7, crossSetFromString("", alph), HorizontalDirection},
		{5, 11, crossSetFromString("BGOPRTWX", alph), VerticalDirection},
		{5, 11, crossSetFromString("", alph), HorizontalDirection},
		{8, 13, TrivialCrossSet, VerticalDirection},
		{8, 13, TrivialCrossSet, HorizontalDirection},
		{11, 4, crossSetFromString("DRS", alph), VerticalDirection},
		{11, 4, crossSetFromString("CGM", alph), HorizontalDirection},
		{2, 2, TrivialCrossSet, VerticalDirection},
		{2, 2, crossSetFromString("AEI", alph), HorizontalDirection},
	}
	for idx, tc := range testCases {
		// Compare values
		if generator.board.getCrossSet(tc.row, tc.col, tc.dir) != tc.crossSet {
			t.Errorf("Test=%v For row=%v col=%v, Expected cross-set to be %v, got %v",
				idx, tc.row, tc.col, tc.crossSet,
				generator.board.getCrossSet(tc.row, tc.col, tc.dir))
		}
	}
}
