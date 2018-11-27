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
	"sort"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
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

type GordonGenerator struct {
	gaddag gaddag.SimpleGaddag
	board  GameBoard
	// curRow is the current row for which we are generating moves. Note
	// that we are always thinking in terms of rows, and columns are the
	// current anchor column. In order to generate vertical moves, we just
	// transpose the `board`.
	curRowIdx     int8
	curAnchorCol  int8
	lastAnchorCol int8

	vertical bool // Are we generating moves vertically or not?

	tilesPlayed uint8
	plays       map[string]Move
	sortedPlays []Move
	bag         *Bag
}

func newGordonGenerator(gd gaddag.SimpleGaddag) *GordonGenerator {
	gen := &GordonGenerator{
		gaddag: gd,
		board:  strToBoard(CrosswordGameBoard),
		plays:  make(map[string]Move),
		bag:    new(Bag),
	}
	gen.bag.Init()
	gen.board.setAllCrosses()
	return gen
}

// GenAll generates all moves on the board. It assumes anchors have already
// been updated, as well as cross-sets / cross-scores.
func (gen *GordonGenerator) GenAll(letters string) {
	// gen.board.updateAllAnchors()
	rack := &Rack{}
	rack.Initialize(letters, gen.gaddag.GetAlphabet())
	dim := int8(gen.board.dim())
	gen.plays = make(map[string]Move)
	orientations := []BoardDirection{HorizontalDirection, VerticalDirection}
	// Once for each orientation
	for idx, dir := range orientations {
		gen.vertical = idx%2 != 0
		for row := int8(0); row < dim; row++ {
			gen.curRowIdx = row
			// A bit of a hack. Set this to a large number at the beginning of
			// every loop
			gen.lastAnchorCol = 100
			for col := int8(0); col < dim; col++ {
				if gen.board.squares[row][col].anchor(dir) {
					gen.curAnchorCol = col
					gen.Gen(col, alphabet.MachineWord(""), rack,
						gen.gaddag.GetRootNodeIndex())
					gen.lastAnchorCol = col
				}
			}
		}
		gen.board.transpose()
	}
	// Sort all plays by score XXX: later equity/win%/whatever
	gen.sortedPlays = []Move{}
	for _, m := range gen.plays {
		gen.sortedPlays = append(gen.sortedPlays, m)
	}
	sort.Sort(ByScore(gen.sortedPlays))
}

// Gen is an implementation of the Gordon Gen function.
func (gen *GordonGenerator) Gen(col int8, word alphabet.MachineWord, rack *Rack,
	nodeIdx uint32) {

	var crossSet CrossSet

	// If a letter L is already on this square, then GoOn...
	curSquare := gen.board.squares[gen.curRowIdx][col]
	curLetter := curSquare.letter

	if gen.vertical {
		crossSet = curSquare.hcrossSet
	} else {
		crossSet = curSquare.vcrossSet
	}

	if !curSquare.isEmpty() {
		nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, curLetter.Unblank())
		gen.GoOn(col, curLetter, word, rack, nnIdx, nodeIdx)

	} else if !rack.empty {
		// We copy the array here to avoid modifying the rack.uniqueLetters
		// in place (with the rack.take / rack.add)
		uniqLetters := []alphabet.MachineLetter{}
		for ml := range rack.uniqueLetters {
			uniqLetters = append(uniqLetters, ml)
		}

		for _, ml := range uniqLetters {
			if ml != BlankPos {
				if crossSet.allowed(ml) {
					nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, ml)
					rack.take(ml)
					gen.tilesPlayed++
					gen.GoOn(col, ml, word, rack, nnIdx, nodeIdx)
					rack.add(ml)
					gen.tilesPlayed--
				}
			} else {
				// It's a blank. Loop only through letters in the cross-set.
				for i := uint8(0); i < gen.gaddag.GetAlphabet().NumLetters(); i++ {
					if crossSet.allowed(alphabet.MachineLetter(i)) {
						nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, alphabet.MachineLetter(i))
						rack.take(BlankPos)
						gen.tilesPlayed++
						gen.GoOn(col, alphabet.MachineLetter(i).Blank(), word, rack, nnIdx, nodeIdx)
						rack.add(BlankPos)
						gen.tilesPlayed--
					}
				}
			}
		}
	}
}

// GoOn is an implementation of the Gordon GoOn function.
func (gen *GordonGenerator) GoOn(curCol int8, L alphabet.MachineLetter, word alphabet.MachineWord,
	rack *Rack, newNodeIdx uint32, oldNodeIdx uint32) {

	if curCol <= gen.curAnchorCol {
		if !gen.board.squares[gen.curRowIdx][curCol].isEmpty() {
			word = alphabet.MachineWord(alphabet.PlayedThroughMarker) + word
		} else {
			word = alphabet.MachineWord(L) + word
		}
		// if L on OldArc and no letter directly left, then record play.
		// roomToLeft is true unless we are right at the edge of the board.
		//roomToLeft := true
		noLetterDirectlyLeft := curCol == 0 || gen.board.squares[gen.curRowIdx][curCol-1].isEmpty()

		// Check to see if there is a letter directly to the left.
		if gen.gaddag.InLetterSet(L, oldNodeIdx) && noLetterDirectlyLeft && gen.tilesPlayed > 0 {
			gen.RecordPlay(word, curCol)
		}
		if newNodeIdx == 0 {
			return
		}
		// Keep generating prefixes if there is room to the left, and don't
		// revisit an anchor we just saw.
		// This seems to work because we always shift direction afterwards, so we're
		// only looking at the first of a consecutive set of anchors going backwards,
		// and then always looking forward from then on.
		if curCol > 0 && gen.lastAnchorCol != gen.curAnchorCol-1 {
			gen.Gen(curCol-1, word, rack, newNodeIdx)
		}
		// Then shift direction.
		// Get the index of the SeparationToken
		separationNodeIdx := gen.gaddag.NextNodeIdx(newNodeIdx, alphabet.SeparationMachineLetter)
		// Check for no letter directly left AND room to the right (of the anchor
		// square)
		if separationNodeIdx != 0 && noLetterDirectlyLeft && gen.curAnchorCol < int8(gen.board.dim()-1) {
			gen.Gen(gen.curAnchorCol+1, word, rack, separationNodeIdx)
		}

	} else {
		if !gen.board.squares[gen.curRowIdx][curCol].isEmpty() {
			word += alphabet.MachineWord(alphabet.PlayedThroughMarker)
		} else {
			word += alphabet.MachineWord(L)
		}

		noLetterDirectlyRight := curCol == int8(gen.board.dim()-1) ||
			gen.board.squares[gen.curRowIdx][curCol+1].isEmpty()
		if gen.gaddag.InLetterSet(L, oldNodeIdx) && noLetterDirectlyRight && gen.tilesPlayed > 0 {
			gen.RecordPlay(word, curCol-int8(len(word))+1)
		}
		if newNodeIdx != 0 && curCol < int8(gen.board.dim()-1) {
			// There is room to the right
			gen.Gen(curCol+1, word, rack, newNodeIdx)
		}
	}
}

// RecordPlay records a play.
func (gen *GordonGenerator) RecordPlay(word alphabet.MachineWord, startCol int8) {
	row := uint8(gen.curRowIdx)
	col := uint8(startCol)
	if gen.vertical {
		// We flip it here because we only generate vertical moves when we transpose
		// the board, so the row and col are actually transposed.
		row, col = col, row
	}
	coords := toBoardGameCoords(row, col, gen.vertical)

	play := Move{
		action:      MoveTypePlay,
		score:       gen.scoreMove(word, startCol),
		desc:        "foo",
		word:        word,
		vertical:    gen.vertical,
		bingo:       gen.tilesPlayed == 7,
		tilesPlayed: gen.tilesPlayed,
		alph:        gen.gaddag.GetAlphabet(),
		colStart:    uint8(startCol),
		rowStart:    uint8(gen.curRowIdx),
		coords:      coords,
	}
	gen.plays[play.uniqueKey()] = play
}

func (gen *GordonGenerator) scoreMove(word alphabet.MachineWord, col int8) int {

	var dir BoardDirection
	var ls int
	if gen.vertical {
		dir = HorizontalDirection
	} else {
		dir = VerticalDirection
	}
	mainWordScore := 0
	crossScores := 0
	bingoBonus := 0
	if gen.tilesPlayed == 7 {
		bingoBonus = 50
	}
	wordMultiplier := 1

	for idx, rn := range word {
		ml := alphabet.MachineLetter(rn)
		bonusSq := gen.board.getBonus(int(gen.curRowIdx), int(col)+idx)
		letterMultiplier := 1
		thisWordMultiplier := 1
		freshTile := false
		if ml == alphabet.PlayedThroughMarker {
			ml = gen.board.getLetter(int(gen.curRowIdx), int(col)+idx)
		} else {
			freshTile = true
			// Only count bonus if we are putting a fresh tile on it.
			switch bonusSq {
			case Bonus3WS:
				wordMultiplier *= 3
				thisWordMultiplier = 3
			case Bonus2WS:
				wordMultiplier *= 2
				thisWordMultiplier = 2
			case Bonus2LS:
				letterMultiplier = 2
			case Bonus3LS:
				letterMultiplier = 3
			}
			// else all the multipliers are 1.
		}
		cs := gen.board.squares[gen.curRowIdx][int(col)+idx].getCrossScore(dir)

		if ml >= alphabet.BlankOffset {
			// letter score is 0
			ls = 0
		} else {
			ls = gen.bag.scores[ml]
		}

		mainWordScore += ls * letterMultiplier
		// We only add cross scores if the cross set of this square is non-trivial
		// (i.e. we have to be making an across word). Note that it's not enough
		// to check that the cross-score is 0 because we could have a blank.
		if freshTile && gen.board.getCrossSet(int(gen.curRowIdx),
			int(col)+idx, dir) != TrivialCrossSet {

			crossScores += ls*letterMultiplier*thisWordMultiplier + cs*thisWordMultiplier
		}
	}
	return mainWordScore*wordMultiplier + crossScores + bingoBonus
}

func (gen *GordonGenerator) traverseBackwardsForScore(row int, col int) int {
	score := 0
	for gen.board.posExists(row, col) {
		ml := gen.board.squares[row][col].letter
		if ml == alphabet.EmptySquareMarker {
			break
		}
		score += gen.bag.score(ml)
		col--
	}
	return score
}

func (gen *GordonGenerator) traverseBackwards(row int, col int, nodeIdx uint32,
	checkLetterSet bool, leftMostCol int) (uint32, bool) {
	// Traverse the letters on the board backwards (left). Return the index
	// of the node in the gaddag for the left-most letter, and a boolean
	// indicating if the gaddag path was valid.
	// If checkLetterSet is true, then we traverse until leftMostCol+1 and
	// check the letter set of this node to see if it includes the letter
	// at leftMostCol
	for gen.board.posExists(row, col) {
		ml := gen.board.squares[row][col].letter
		if ml == alphabet.EmptySquareMarker {
			break
		}

		if checkLetterSet && col == leftMostCol {
			if gen.gaddag.InLetterSet(ml, nodeIdx) {
				return nodeIdx, true
			}
			// Give up early; if we're checking letter sets we only care about
			// this column.
			return nodeIdx, false
		}

		nodeIdx = gen.gaddag.NextNodeIdx(nodeIdx, ml.Unblank())
		if nodeIdx == 0 {
			// There is no path in the gaddag for this word part; this
			// can occur if a phony was played and stayed on the board
			// and the phony has no extensions for example, or if it's
			// a real word with no further extensions.
			return nodeIdx, false
		}

		col--
	}

	return nodeIdx, true
}

func (gen *GordonGenerator) genAllCrossSets() {
	// Generate all cross-sets. Basically go through the entire board;
	// our anchor algorithm doesn't quite match the one in the Gordon
	// paper.

	// We should do this for both transpositions of the board.

	n := gen.board.dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !gen.board.squares[i][j].isEmpty() {
				gen.board.squares[i][j].setCrossSet(CrossSet(0), HorizontalDirection)
				gen.board.squares[i][j].setCrossScore(0, HorizontalDirection)
			} else {
				gen.genCrossSet(i, j, HorizontalDirection)
			}
		}
	}
	gen.board.transpose()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !gen.board.squares[i][j].isEmpty() {
				gen.board.squares[i][j].setCrossSet(CrossSet(0), VerticalDirection)
				gen.board.squares[i][j].setCrossScore(0, VerticalDirection)
			} else {
				gen.genCrossSet(i, j, VerticalDirection)
			}
		}
	}
	// And transpose back to the original orientation.
	gen.board.transpose()
}

func (gen *GordonGenerator) genCrossSet(row int, col int, dir BoardDirection) {
	// This function is always called for empty squares.
	// If there's no tile adjacent to this square in any direction,
	// every letter is allowed.
	if gen.board.leftAndRightEmpty(row, col) {
		gen.board.squares[row][col].setCrossSet(TrivialCrossSet, dir)
		gen.board.squares[row][col].setCrossScore(0, dir)
		return
	}
	// If we are here, there is a letter to the left, to the right, or both.
	// start from the right and go backwards.
	rightCol := gen.board.wordEdge(row, col+1, RightDirection)
	if rightCol == col {
		// This means the right was always empty; we only want to go left.
		lNodeIdx, lPathValid := gen.traverseBackwards(row, col-1,
			gen.gaddag.GetRootNodeIndex(), false, 0)
		score := gen.traverseBackwardsForScore(row, col-1)
		gen.board.squares[row][col].setCrossScore(score, dir)

		if !lPathValid {
			// There are no further extensions to the word on the board,
			// which may also be a phony.
			gen.board.squares[row][col].setCrossSet(CrossSet(0), dir)
			return
		}
		// Otherwise, we have a left node index.
		sIdx := gen.gaddag.NextNodeIdx(lNodeIdx, alphabet.SeparationMachineLetter)
		// Take the letter set of this sIdx as the cross-set.
		letterSet := gen.gaddag.GetLetterSet(sIdx)
		// Miraculously, letter sets and cross sets are compatible.
		gen.board.squares[row][col].setCrossSet(CrossSet(letterSet), dir)
	} else {
		// Otherwise, the right is not empty. Check if the left is empty,
		// if so we just traverse right, otherwise, we try every letter.
		leftCol := gen.board.wordEdge(row, col-1, LeftDirection)
		// Start at the right col and work back to this square.
		lNodeIdx, lPathValid := gen.traverseBackwards(row, rightCol,
			gen.gaddag.GetRootNodeIndex(), false, 0)
		scoreR := gen.traverseBackwardsForScore(row, rightCol)
		scoreL := gen.traverseBackwardsForScore(row, col-1)
		gen.board.squares[row][col].setCrossScore(scoreR+scoreL, dir)
		if !lPathValid {
			gen.board.squares[row][col].setCrossSet(CrossSet(0), dir)
			return
		}
		if leftCol == col {
			// The left is empty, but the right isn't.
			// The cross-set is just the letter set of the letter directly
			// to our right.
			letterSet := gen.gaddag.GetLetterSet(lNodeIdx)
			gen.board.squares[row][col].setCrossSet(CrossSet(letterSet), dir)
		} else {
			// Both the left and the right have a tile. Go through the
			// siblings, from the right, to see what nodes lead to the left.
			numArcs := gen.gaddag.NumArcs(lNodeIdx)
			crossSet := gen.board.squares[row][col].getCrossSet(dir)
			*crossSet = CrossSet(0)
			for i := lNodeIdx + 1; i <= uint32(numArcs)+lNodeIdx; i++ {
				ml := alphabet.MachineLetter(gen.gaddag.Nodes[i] >>
					gaddagmaker.LetterBitLoc)
				if ml == alphabet.SeparationMachineLetter {
					continue
				}
				nnIdx := gen.gaddag.Nodes[i] & gaddagmaker.NodeIdxBitMask
				_, success := gen.traverseBackwards(row, col-1, nnIdx, true,
					leftCol)
				if success {
					crossSet.set(ml)
				}
			}
		}
	}
}

// For future?: The Gordon GADDAG algorithm is somewhat inefficient because
// it goes through all letters on the rack. Then for every letter, it has to
// call the NextNodeIdx or similar function above, which has for loops that
// search for the next child.
// Instead, we need a data structure where the nodes have pointers to their
// "children" or "siblings" on the arcs; we then iterate through all the
// "siblings" and see if their letters are on the rack. This should be
// significantly faster if the data structure is fast.
