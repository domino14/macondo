package movegen

import (
	"fmt"
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

func TestGenFrontHook(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("P", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 11

	setBoardRow(generator.board, 2, "     REGNANT", gd.GetAlphabet())
	generator.curRow = generator.board.squares[2]
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate PREGNANT

	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays, expected len=%v", generator.plays, 1)
	}
}

func TestGenBackHook(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("O", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 9

	setBoardRow(generator.board, 2, "  PORTOLAN", gd.GetAlphabet())
	generator.curRow = generator.board.squares[2]
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate PORTOLANO
	// NALOTROP^O
	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays, expected len=%v", generator.plays, 1)
	}
}

func TestGenBackHook2(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("S", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 9

	setBoardRow(generator.board, 2, "  PORTOLAN", gd.GetAlphabet())
	generator.curRow = generator.board.squares[2]
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate PORTOLANS
	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays, expected len=%v", generator.plays, 1)
	}
}

func TestGenBackHookBlank(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("?", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 9

	setBoardRow(generator.board, 2, "  PORTOLAN", gd.GetAlphabet())
	generator.curRow = generator.board.squares[2]
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate PORTOLANO and PORTOLANS
	if len(generator.plays) != 2 {
		t.Errorf("Generated %v plays, expected len=%v", generator.plays, 2)
	}
}

func TestGenThroughSmall(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("TY", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 7

	setBoardRow(generator.board, 2, "  SOVRAN", gd.GetAlphabet())
	generator.curRow = generator.board.squares[2]
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate SOVRANTY
	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays, expected len=%v", generator.plays, 2)
	}
}

func TestGenThroughLarger(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("ING", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 6

	setBoardRow(generator.board, 2, "  LAUGH", gd.GetAlphabet())
	generator.curRow = generator.board.squares[2]
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate LAUGHING
	if len(generator.plays) != 1 {
		t.Errorf("Generated %v (%v) plays, expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
}

func TestGenThroughSimple(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("ZA", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	// Anchors on the left are ON the leftmost letter.
	generator.curAnchorCol = 3

	setBoardRow(generator.board, 4, "  BE", gd.GetAlphabet())
	generator.curRow = generator.board.squares[4]
	log.Printf("Current row: %v", generator.curRow)
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate nothing

	if len(generator.plays) != 0 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 0)
	}
}

func TestGenThrough(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("AENPPSW", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 14

	setBoardRow(generator.board, 4, "        CHAWING", gd.GetAlphabet())
	generator.curRow = generator.board.squares[4]
	log.Printf("Current row: %v", generator.curRow)
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate WAPPENSCHAWING

	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
}

func TestGenThroughBothWays(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("ABEHINT", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 9

	setBoardRow(generator.board, 4, "   THERMOS  A", gd.GetAlphabet())
	generator.curRow = generator.board.squares[4]
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate NETHERMOST and HITHERMOST

	if len(generator.plays) != 2 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
}

func TestGenThroughInBetween(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("ABEHITT", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 8 // anchor on the letter S

	setBoardRow(generator.board, 4, "  THERMOS A   ", gd.GetAlphabet())
	generator.curRow = generator.board.squares[4]
	log.Printf("Current row: %v", generator.curRow)
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate THERMOSTAT

	if len(generator.plays) != 1 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
	}
}
func TestGenThroughInBetweenRightAnchor(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("TT", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 10 // anchor on the letter A

	setBoardRow(generator.board, 4, "  THERMOS A   ", gd.GetAlphabet())
	generator.curRow = generator.board.squares[4]
	generator.Gen(generator.curAnchorCol, alphabet.MachineWord(""), rack,
		gd.GetRootNodeIndex())
	// it should generate THERMOSTAT, AT, and ATT from the A

	if len(generator.plays) != 3 {
		t.Errorf("Generated %v plays (%v), expected len=%v", generator.plays,
			len(generator.plays), 1)
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
 9|    '     Q I   '     A '    |
10|  "       I   N   "   N   "  |
11|      R e S P O N D - D      |
12|' H O E       V       O     '|
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

func TestMatt(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	generator := newGordonGenerator(gd)
	setBoardToGame(generator, gd.GetAlphabet(), VsMatt)
	generator.board.updateAllAnchors()
	fmt.Println(generator.board.toDisplayText(gd.GetAlphabet()))
	t.Errorf("bye")
}
