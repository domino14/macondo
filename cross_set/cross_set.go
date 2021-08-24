package cross_set

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/crosses"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/move"
)

const (
	// TrivialCrossSet allows every possible letter. It is the default
	// state of a square.
	TrivialCrossSet = (1 << alphabet.MaxAlphabetSize) - 1
)

// A CrossSet is a bit mask of letters that are allowed on a square. It is
// inherently directional, as it depends on which direction we are generating
// moves in. If we are generating moves HORIZONTALLY, we check in the
// VERTICAL cross set to make sure we can play a letter there.
// Therefore, a VERTICAL cross set is created by looking at the tile(s)
// above and/or below the relevant square and seeing what letters lead to
// valid words.
type CrossSet uint64

func (c CrossSet) Allowed(letter alphabet.MachineLetter) bool {
	return c&(1<<uint8(letter)) != 0
}

func (c *CrossSet) Set(letter alphabet.MachineLetter) {
	*c = *c | (1 << letter)
}

func CrossSetFromString(letters string, alph *alphabet.Alphabet) CrossSet {
	c := CrossSet(0)
	for _, l := range letters {
		v, err := alph.Val(l)
		if err != nil {
			panic("Letter error: " + string(l))
		}
		c.Set(v)
	}
	return c
}

func (c *CrossSet) SetAll() {
	*c = TrivialCrossSet
}

func (c *CrossSet) Clear() {
	*c = 0
}

// BoardCrossSets stores the cross-sets for a game board (see board module).
// We don't store these directly in the game board structure as we want to
// keep cross-sets and move generation separate from the crossword game logic.
type BoardCrossSets struct {
	hcrossSets []CrossSet
	vcrossSets []CrossSet

	board *board.GameBoard
}

func (bcs *BoardCrossSets) Set(row int, col int, cs int64, dir board.BoardDirection) {
	pos := row*bcs.board.Dim() + col
	if bcs.board.IsTransposed() {
		pos = col*bcs.board.Dim() + row
	}
	if dir == board.HorizontalDirection {
		bcs.hcrossSets[pos] = CrossSet(cs)
		return
	}
	bcs.vcrossSets[pos] = CrossSet(cs)
}

func (bcs *BoardCrossSets) Add(row, col int, ml alphabet.MachineLetter, dir board.BoardDirection) {
	c := bcs.Get(row, col, dir)
	c = c | (1 << ml)

	bcs.Set(row, col, int64(c), dir)
}

func (bcs *BoardCrossSets) Get(row, col int, dir board.BoardDirection) CrossSet {

	pos := row*bcs.board.Dim() + col
	if bcs.board.IsTransposed() {
		pos = col*bcs.board.Dim() + row
	}
	if dir == board.HorizontalDirection {
		return bcs.hcrossSets[pos]
	}
	return bcs.vcrossSets[pos]
}

func (bcs *BoardCrossSets) Clear(row, col int, dir board.BoardDirection) {
	pos := row*bcs.board.Dim() + col
	if bcs.board.IsTransposed() {
		pos = col*bcs.board.Dim() + row
	}
	if dir == board.HorizontalDirection {
		bcs.hcrossSets[pos] = 0
		return
	}
	bcs.vcrossSets[pos] = 0
}

func (bcs *BoardCrossSets) ClearAll() {
	for i := 0; i < len(bcs.hcrossSets); i++ {
		bcs.hcrossSets[i] = 0
		bcs.vcrossSets[i] = 0
	}
}

func (bcs *BoardCrossSets) SetAll() {
	for i := 0; i < len(bcs.hcrossSets); i++ {
		bcs.hcrossSets[i] = TrivialCrossSet
		bcs.vcrossSets[i] = TrivialCrossSet
	}
}

func (bcs *BoardCrossSets) Equals(other *BoardCrossSets) bool {
	if len(bcs.hcrossSets) != len(other.hcrossSets) {
		return false
	}
	if len(bcs.vcrossSets) != len(other.vcrossSets) {
		return false
	}
	for i, h := range bcs.hcrossSets {
		if h != other.hcrossSets[i] {
			return false
		}
	}
	for i, v := range bcs.vcrossSets {
		if v != other.vcrossSets[i] {
			return false
		}
	}
	return true
}

func MakeBoardCrossSets(board *board.GameBoard) *BoardCrossSets {
	n := board.Dim() * board.Dim()
	return &BoardCrossSets{
		hcrossSets: make([]CrossSet, n),
		vcrossSets: make([]CrossSet, n),
		board:      board,
	}
}

// ----------------------------------------------------------------------
// Implementation for CrossScoreOnlyGenerator

// ----------------------------------------------------------------------
// GaddagCrossSetGenerator generates cross sets via a gaddag

type GaddagCrossSetGenerator struct {
	Dist   *alphabet.LetterDistribution
	Gaddag gaddag.GenericDawg
}

func (g GaddagCrossSetGenerator) Generate(b *board.GameBoard, cs crosses.Crosser, row int, col int, dir board.BoardDirection) {
	GenCrossSet(b, cs, row, col, dir, g.Gaddag, g.Dist)
}

func (g GaddagCrossSetGenerator) GenerateAll(b *board.GameBoard, cs crosses.Crosser) {
	crosses.GenerateAll(g, b, cs)
}

func (g GaddagCrossSetGenerator) UpdateForMove(b *board.GameBoard, cs crosses.Crosser, m *move.Move) {
	crosses.UpdateForMove(g, b, cs, m)
}

// Wrapper functions to save rewriting all the tests

func GenAllCrossSets(b *board.GameBoard, cs *BoardCrossSets, gd gaddag.GenericDawg, ld *alphabet.LetterDistribution) {
	// Shortcut if board has nothing on it.
	if b.TilesPlayed() == 0 {
		cs.SetAll()
		return
	}

	gen := GaddagCrossSetGenerator{Dist: ld, Gaddag: gd}
	gen.GenerateAll(b, cs)
}

func UpdateCrossSetsForMove(b *board.GameBoard, cs *BoardCrossSets, m *move.Move,
	gd gaddag.GenericDawg, ld *alphabet.LetterDistribution) {
	gen := GaddagCrossSetGenerator{Dist: ld, Gaddag: gd}
	gen.UpdateForMove(b, cs, m)
}

// ----------------------------------------------------------------------
// Implementation for GaddagCrossSetGenerator

func traverseBackwards(b *board.GameBoard, row int, col int,
	nodeIdx uint32, checkLetterSet bool, leftMostCol int,
	gaddag gaddag.GenericDawg) (uint32, bool) {
	// Traverse the letters on the board backwards (left). Return the index
	// of the node in the gaddag for the left-most letter, and a boolean
	// indicating if the gaddag path was valid.
	// If checkLetterSet is true, then we traverse until leftMostCol+1 and
	// check the letter set of this node to see if it includes the letter
	// at leftMostCol
	for b.PosExists(row, col) {
		ml := b.GetLetter(row, col)
		if ml == alphabet.EmptySquareMarker {
			break
		}

		if checkLetterSet && col == leftMostCol {
			if gaddag.InLetterSet(ml, nodeIdx) {
				return nodeIdx, true
			}
			// Give up early; if we're checking letter sets we only care about
			// this column.
			return nodeIdx, false
		}

		nodeIdx = gaddag.NextNodeIdx(nodeIdx, ml.Unblank())
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

// GenCrossSet generates a cross-set for each individual square.
func GenCrossSet(b *board.GameBoard, cs crosses.Crosser, row int, col int, dir board.BoardDirection,
	gaddag gaddag.GenericDawg, ld *alphabet.LetterDistribution) {

	if row < 0 || row >= b.Dim() || col < 0 || col >= b.Dim() {
		return
	}
	// If the square has a letter in it, its cross set and cross score
	// should both be 0
	if b.HasLetter(row, col) {
		b.SetCrossScore(row, col, 0, dir)
		cs.Set(row, col, 0, dir)
		return
	}
	// If there's no tile adjacent to this square in any direction,
	// every letter is allowed.
	if b.LeftAndRightEmpty(row, col) {
		b.SetCrossScore(row, col, 0, dir)
		cs.Set(row, col, TrivialCrossSet, dir)
		return
	}
	// If we are here, there is a letter to the left, to the right, or both.
	// start from the right and go backwards.
	rightCol := b.WordEdge(row, col+1, board.RightDirection)
	if rightCol == col {
		// This means the right was always empty; we only want to go left.
		lNodeIdx, lPathValid := traverseBackwards(b, row, col-1,
			gaddag.GetRootNodeIndex(), false, 0, gaddag)
		score := b.TraverseBackwardsForScore(row, col-1, ld)
		b.SetCrossScore(row, col, score, dir)

		if !lPathValid {
			// There are no further extensions to the word on the board,
			// which may also be a phony.
			cs.Set(row, col, 0, dir)
			return
		}
		// Otherwise, we have a left node index.
		sIdx := gaddag.NextNodeIdx(lNodeIdx, alphabet.SeparationMachineLetter)
		// Take the letter set of this sIdx as the cross-set.
		letterSet := gaddag.GetLetterSet(sIdx)
		// Miraculously, letter sets and cross sets are compatible.
		cs.Set(row, col, int64(letterSet), dir)
	} else {

		// Otherwise, the right is not empty. Check if the left is empty,
		// if so we just traverse right, otherwise, we try every letter.
		leftCol := b.WordEdge(row, col-1, board.LeftDirection)
		// Start at the right col and work back to this square.
		lNodeIdx, lPathValid := traverseBackwards(b, row, rightCol,
			gaddag.GetRootNodeIndex(), false, 0, gaddag)
		scoreR := b.TraverseBackwardsForScore(row, rightCol, ld)
		scoreL := b.TraverseBackwardsForScore(row, col-1, ld)
		b.SetCrossScore(row, col, scoreR+scoreL, dir)
		if !lPathValid {
			cs.Set(row, col, 0, dir)
			return
		}
		if leftCol == col {
			// The left is empty, but the right isn't.
			// The cross-set is just the letter set of the letter directly
			// to our right.

			letterSet := gaddag.GetLetterSet(lNodeIdx)
			cs.Set(row, col, int64(letterSet), dir)
		} else {
			// Both the left and the right have a tile. Go through the
			// siblings, from the right, to see what nodes lead to the left.

			numArcs := gaddag.NumArcs(lNodeIdx)
			cs.Set(row, col, 0, dir)
			for i := lNodeIdx + 1; i <= uint32(numArcs)+lNodeIdx; i++ {

				ml := alphabet.MachineLetter(gaddag.Nodes()[i] >>
					gaddagmaker.LetterBitLoc)

				if ml == alphabet.SeparationMachineLetter {
					continue
				}
				nnIdx := gaddag.Nodes()[i] & gaddagmaker.NodeIdxBitMask
				_, success := traverseBackwards(b, row, col-1, nnIdx, true,
					leftCol, gaddag)
				if success {
					cs.Add(row, col, ml, dir)
				}
			}
		}
	}

}
