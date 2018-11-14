// Package movegen contains all the move-generating functions. It makes
// heavy use of the GADDAG.
// Implementation notes:
// - Is the specification in the paper a bit buggy? Basically, if I assume
// an anchor is the leftmost tile of a word, the way the algorithm works,
// it will create words blindly. For example, if I have a word FIRE on the
// board, and I have the letter E on my rack, and I specify F as the anchor,
// it will create the word EF! (Ignoring the fact that IRE is on the board)
// You can see this by just stepping through the algorithm.
// It seems that anchors can only be on the rightmost tile of a word

package movegen

import (
	"fmt"
	"log"
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
)

type MoveType uint8

const (
	MoveTypePlay MoveType = iota
	MoveTypeExchange
	MoveTypePass
	MoveTypePhonyTilesReturned

	MoveTypeEndgameTiles
	MoveTypeLostTileScore
)

// LettersRemain returns true if there is at least one letter in the
// rack, 0 otherwise.
func LettersRemain(rack []uint8) bool {
	for i := 0; i < alphabet.MaxAlphabetSize; i++ {
		if rack[i] > 0 {
			return true
		}
	}
	return false
}

type Move struct {
	action      MoveType
	score       int8
	desc        string
	word        alphabet.MachineWord
	rowStart    uint8
	colStart    uint8
	vertical    bool
	bingo       bool
	tilesPlayed uint8
	alph        *alphabet.Alphabet
}

func (m Move) String() string {
	return fmt.Sprintf("<action: %v col: %v word: %v bingo: %v tp: %v>",
		m.action, m.colStart, m.word.UserVisible(m.alph), m.bingo,
		m.tilesPlayed)
}

type GordonGenerator struct {
	gaddag gaddag.SimpleGaddag
	board  GameBoard
	// curRow is the current row for which we are generating moves. Note
	// that we are always thinking in terms of rows, and columns are the
	// current anchor column. In order to generate vertical moves, we just
	// transpose the `board`.
	curRow       []*Square
	curAnchorCol int8

	vertical bool // Are we generating moves vertically or not?

	tilesPlayed uint8
	plays       []Move
}

func newGordonGenerator(gd gaddag.SimpleGaddag) *GordonGenerator {
	gen := &GordonGenerator{
		gaddag: gd,
		board:  strToBoard(CrosswordGameBoard),
		plays:  []Move{},
	}
	gen.board.setAllCrosses()
	return gen
}

func (gen *GordonGenerator) GenAll(rack []string, board GameBoard) {
	gen.board = board
	// gen.curAnchorRow = 7
	// gen.curAnchorCol = 7
}

// Gen is an implementation of the Gordon Gen function.
func (gen *GordonGenerator) Gen(col int8, word alphabet.MachineWord, rack *Rack,
	nodeIdx uint32, recCtrs ...int) {
	var recCtr int
	if len(recCtrs) > 0 {
		recCtr = recCtrs[0]
	} else {
		recCtr = 0
	}
	spaces := strings.Repeat(" ", recCtr)

	log.Printf("[DEBUG]%v Entered Gen, col=%v, rack map=%v", spaces, col, rack.uniqueLetters)

	var crossSet CrossSet

	// If a letter L is already on this square, then GoOn...
	curSquare := gen.curRow[col]
	curLetter := curSquare.letter

	if gen.vertical {
		crossSet = curSquare.hcrossSet
	} else {
		crossSet = curSquare.vcrossSet
	}

	if curLetter != EmptySquareMarker {
		nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, curLetter)
		log.Printf("[DEBUG]%v Tryna find letter, Calling GoOn with col=%v, letter=%v, nodeIdx=%v",
			spaces, col, curLetter.UserVisible(gen.gaddag.GetAlphabet()), nnIdx)

		gen.GoOn(col, curLetter, word, rack, nnIdx, nodeIdx, recCtr+1)

	} else if !rack.empty {
		uniqLetters := []alphabet.MachineLetter{}
		for ml := range rack.uniqueLetters {
			uniqLetters = append(uniqLetters, ml)
		}

		for _, ml := range uniqLetters {
			if ml != BlankPos {
				if crossSet.allowed(ml) {
					nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, ml)
					log.Printf("[DEBUG]%v Taking ml %v", spaces, ml.UserVisible(gen.gaddag.GetAlphabet()))

					rack.take(ml)
					gen.tilesPlayed++

					log.Printf("[DEBUG]%v Calling GoOn with col=%v letter=%v nnIdx=%v nodeIdx=%v",
						spaces, col, ml.UserVisible(gen.gaddag.GetAlphabet()), nnIdx, nodeIdx)
					gen.GoOn(col, ml, word, rack, nnIdx, nodeIdx, recCtr+1)
					rack.add(ml)
					log.Printf("[DEBUG]%v Put ml %v back", spaces, ml.UserVisible(gen.gaddag.GetAlphabet()))

					gen.tilesPlayed--
				}
			} else {
				// It's a blank. Loop only through letters in the cross-set.
				for i := uint8(0); i < gen.gaddag.GetAlphabet().NumLetters(); i++ {
					if crossSet.allowed(alphabet.MachineLetter(i)) {
						nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, alphabet.MachineLetter(i))
						rack.take(BlankPos)
						log.Printf("[DEBUG]%v Take the blank", spaces)

						gen.tilesPlayed++
						gen.GoOn(col, alphabet.MachineLetter(i).Blank(), word, rack, nnIdx, nodeIdx, recCtr+1)
						rack.add(BlankPos)
						log.Printf("[DEBUG]%v Put the blank back", spaces)

						gen.tilesPlayed--
					}
				}
			}
		}
	}
	log.Printf("[DEBUG]%v Leaving Gen function", spaces)

}

// GoOn is an implementation of the Gordon GoOn function.
func (gen *GordonGenerator) GoOn(curCol int8, L alphabet.MachineLetter, word alphabet.MachineWord,
	rack *Rack, newNodeIdx uint32, oldNodeIdx uint32, recCtr int) {

	spaces := strings.Repeat(" ", recCtr)

	if curCol <= gen.curAnchorCol {
		word = alphabet.MachineWord(L) + word
		// if L on OldArc and no letter directly left, then record play.
		// roomToLeft is true unless we are right at the edge of the board.
		//roomToLeft := true
		noLetterDirectlyLeft := (curCol == 0 ||
			gen.curRow[curCol-1].letter == EmptySquareMarker)

		// Check to see if there is a letter directly to the left.

		log.Printf("[DEBUG]%v leftCol=%v, noLetterDirectlyLeft=%v",
			spaces, curCol-1, noLetterDirectlyLeft)
		if noLetterDirectlyLeft {
			log.Printf("[DEBUG]%v Checking if %v is in letterset, word so far %v",
				spaces, L.UserVisible(gen.gaddag.GetAlphabet()), word.UserVisible(
					gen.gaddag.GetAlphabet()))
		}
		if gen.gaddag.InLetterSet(L, oldNodeIdx) && noLetterDirectlyLeft && gen.tilesPlayed > 0 {
			gen.RecordPlay(word, curCol)
		}
		if newNodeIdx == 0 {
			log.Printf("[DEBUG]%v newNodeIdx=0, returning from GoOn", spaces)
			return
		}
		// Keep generating prefixes if there is room to the left

		if curCol > 0 {
			log.Printf("[DEBUG]%v Room to left, Calling Gen with col=%v, word=%v, nodeIdx=%v",
				spaces, curCol-1, word.UserVisible(gen.gaddag.GetAlphabet()), newNodeIdx)

			gen.Gen(curCol-1, word, rack, newNodeIdx, recCtr)
		}
		// Then shift direction.
		// Get the index of the SeparationToken
		separationNodeIdx := gen.gaddag.NextNodeIdx(newNodeIdx, alphabet.SeparationMachineLetter)
		// Check for no letter directly left AND room to the right (of the anchor
		// square)
		if separationNodeIdx != 0 && noLetterDirectlyLeft && gen.curAnchorCol < int8(gen.board.dim()-1) {
			log.Printf("[DEBUG]%v Shift direction, calling Gen with col=%v, word=%v nodeIdx=%v",
				spaces, gen.curAnchorCol+1, word.UserVisible(gen.gaddag.GetAlphabet()), separationNodeIdx)
			gen.Gen(gen.curAnchorCol+1, word, rack, separationNodeIdx, recCtr)
		}

	} else {
		word = word + alphabet.MachineWord(L)
		noLetterDirectlyRight := (curCol == int8(gen.board.dim()-1) ||
			gen.curRow[curCol+1].letter == EmptySquareMarker)
		log.Printf("[DEBUG]%v No letter directly right: %v", spaces, noLetterDirectlyRight)
		if noLetterDirectlyRight {
			log.Printf("[DEBUG]%v Checking if %v is in letterset, word so far %v",
				spaces, L.UserVisible(gen.gaddag.GetAlphabet()), word.UserVisible(
					gen.gaddag.GetAlphabet()))
		}
		if gen.gaddag.InLetterSet(L, oldNodeIdx) && noLetterDirectlyRight && gen.tilesPlayed > 0 {
			gen.RecordPlay(word, curCol-int8(len(word))+1)
		}
		if newNodeIdx != 0 && curCol < int8(gen.board.dim()-1) {
			// There is room to the right
			log.Printf("[DEBUG]%v Room to right, calling Gen with col=%v, word=%v newNodeIdx=%v",
				spaces, curCol+1, word.UserVisible(gen.gaddag.GetAlphabet()), newNodeIdx)
			gen.Gen(curCol+1, word, rack, newNodeIdx, recCtr)
		}
	}
}

// RecordPlay records a play.
func (gen *GordonGenerator) RecordPlay(word alphabet.MachineWord, startCol int8) {
	play := Move{
		action:      MoveTypePlay,
		score:       17,
		desc:        "foo",
		word:        word,
		vertical:    gen.vertical,
		bingo:       gen.tilesPlayed == 7,
		tilesPlayed: gen.tilesPlayed,
		alph:        gen.gaddag.GetAlphabet(),
		colStart:    uint8(startCol),
	}
	gen.plays = append(gen.plays, play)
	log.Printf("[DEBUG] Recorded play %v, startcol %v",
		word.UserVisible(gen.gaddag.GetAlphabet()), startCol)
}

// For future?: The Gordon GADDAG algorithm is somewhat inefficient because
// it goes through all letters on the rack. Then for every letter, it has to
// call the NextNodeIdx or similar function above, which has for loops that
// search for the next child.
// Instead, we need a data structure where the nodes have pointers to their
// "children" or "siblings" on the arcs; we then iterate through all the
// "siblings" and see if their letters are on the rack. This should be
// significantly faster if the data structure is fast.
