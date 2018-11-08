package movegen

import (
	"log"
	"os"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
)

var LexiconDir = os.Getenv("GADDAG_DIR")

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

// This is going to be a big file; it tests the main move generation
// recursive algorithm

func TestGenBase(t *testing.T) {
	// Sanity check. A board with no cross checks should generate nothing.
	rack := &Rack{}
	gd := gaddag.LoadGaddag(LexiconDir + "America.gaddag")
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
	gd := gaddag.LoadGaddag(LexiconDir + "America.gaddag")
	rack.Initialize("P", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 4

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
	gd := gaddag.LoadGaddag(LexiconDir + "America.gaddag")
	rack.Initialize("O", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 10

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
	gd := gaddag.LoadGaddag(LexiconDir + "America.gaddag")
	rack.Initialize("S", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 10

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
	gd := gaddag.LoadGaddag(LexiconDir + "America.gaddag")
	rack.Initialize("?", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 10

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
	gd := gaddag.LoadGaddag(LexiconDir + "America.gaddag")
	rack.Initialize("TY", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 8

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
	gd := gaddag.LoadGaddag(LexiconDir + "America.gaddag")
	rack.Initialize("ING", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 7

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
	gd := gaddag.LoadGaddag(LexiconDir + "America.gaddag")
	rack.Initialize("ZA", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	// Anchors on the left are ON the leftmost letter.
	generator.curAnchorCol = 2

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
	gd := gaddag.LoadGaddag(LexiconDir + "America.gaddag")
	rack.Initialize("AENPPSW", gd.GetAlphabet())

	generator := newGordonGenerator(gd)
	generator.curAnchorCol = 7

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
