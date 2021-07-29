package cross_set

import (
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/cgboard"
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

// BoardCrossSets stores the cross-sets for a game board (see cgboard module).
// We don't store these directly in the game board structure as we want to
// keep cross-sets and move generation separate from the crossword game logic.
type BoardCrossSets struct {
	hcrossSets []CrossSet
	vcrossSets []CrossSet

	board *Board
}

func (bcs *BoardCrossSets) SetCrossSet(row int, col int, cs CrossSet, dir cgboard.BoardDirection) {
	pos := row*bcs.board.Dim() + col
	if bcs.board.IsTransposed() {
		pos = col*bcs.board.Dim() + row
	}
	if dir == Horizontal {
		bcs.hcrossSets[pos] = cs
		return
	}
	bcs.vcrossSets[pos] = cs
}

func (bcs *BoardCrossSets) AddCrossSet(row, col int, ml alphabet.MachineLetter, dir cgboard.BoardDirection) {
	c := bcs.GetCrossSet(row, col, dir)
	c = c | (1 << ml)

	bcs.SetCrossSet(row, col, c, dir)
}

func (bcs *BoardCrossSets) GetCrossSet(row, col int, dir cgboard.BoardDirection) CrossSet {

	pos := row*bcs.board.Dim() + col
	if bcs.board.IsTransposed() {
		pos = col*bcs.board.Dim() + row
	}
	if dir == Horizontal {
		return bcs.hcrossSets[pos]
	}
	return bcs.vcrossSets[pos]
}

func (bcs *BoardCrossSets) ClearCrossSet(row, col int, dir cgboard.BoardDirection) {
	pos := row*bcs.board.Dim() + col
	if bcs.board.IsTransposed() {
		pos = col*bcs.board.Dim() + row
	}
	if dir == Horizontal {
		bcs.hcrossSets[pos] = 0
		return
	}
	bcs.vcrossSets[pos] = 0
}

func (bcs *BoardCrossSets) ClearAllCrosses() {
	for i := 0; i < bcs.board.Dim(); i++ {
		for j := 0; j < bcs.board.Dim(); j++ {
			bcs.ClearCrossSet(i, j, Horizontal)
			bcs.ClearCrossSet(i, j, Vertical)
		}
	}
}

func (bcs *BoardCrossSets) SetAllCrosses() {
	for i := 0; i < bcs.board.Dim(); i++ {
		for j := 0; j < bcs.board.Dim(); j++ {
			bcs.SetCrossSet(i, j, TrivialCrossSet, Horizontal)
			bcs.SetCrossSet(i, j, TrivialCrossSet, Vertical)
		}
	}
}

func MakeBoardCrossSets(board *Board) *BoardCrossSets {
	n := board.Dim() * board.Dim()
	return &BoardCrossSets{
		hcrossSets: make([]CrossSet, n),
		vcrossSets: make([]CrossSet, n),
		board:      board,
	}
}

type Board = cgboard.GameBoard

const (
	Left       = cgboard.LeftDirection
	Right      = cgboard.RightDirection
	Horizontal = cgboard.HorizontalDirection
	Vertical   = cgboard.VerticalDirection
)

// Public cross_set.Generator Interface
// There are two concrete implementations below,
// - CrossScoreOnlyGenerator{Dist}
// - GaddagCrossSetGenerator{Dist, Gaddag}

type Generator interface {
	Generate(b *Board, cs *BoardCrossSets, row int, col int, dir cgboard.BoardDirection)
	GenerateAll(b *Board, cs *BoardCrossSets)
	UpdateForMove(b *Board, cs *BoardCrossSets, m *move.Move)
}

// We have to go through this dance since go will not let us simply provide
// Generator with default implementations of GenerateAll and UpdateForMove that
// call a given implementation of Generate.

type iGenerator interface {
	Generate(b *Board, cs *BoardCrossSets, row int, col int, dir cgboard.BoardDirection)
}

// generateAll generates all cross-sets. It goes through the entire
// board; our anchor algorithm doesn't quite match the one in the Gordon
// paper.
// We do this for both transpositions of the board.
func generateAll(g iGenerator, b *Board, cs *BoardCrossSets) {
	n := b.Dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.Generate(b, cs, i, j, Horizontal)
		}
	}
	b.Transpose()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.Generate(b, cs, i, j, Vertical)
		}
	}
	// And transpose back to the original orientation.
	b.Transpose()
}

func updateForMove(g iGenerator, b *Board, cs *BoardCrossSets, m *move.Move) {

	log.Debug().Msgf("Updating for move: %s", m.ShortDescription())
	row, col, vertical := m.CoordsAndVertical()
	// Every tile placed by this new move creates new "across" words, and we need
	// to update the cross sets on both sides of these across words, as well
	// as the cross sets for THIS word.

	// Assumes all across words are HORIZONTAL.
	calcForAcross := func(rowStart int, colStart int, csd cgboard.BoardDirection) {
		for row := rowStart; row < len(m.Tiles())+rowStart; row++ {
			if m.Tiles()[row-rowStart] == alphabet.PlayedThroughMarker {
				// No new "across word" was generated by this tile, so no need
				// to update cross set.
				continue
			}
			// Otherwise, look along this row. Note, the edge is still part
			// of the word.
			rightCol := b.WordEdge(int(row), int(colStart), Right)
			leftCol := b.WordEdge(int(row), int(colStart), Left)
			g.Generate(b, cs, int(row), int(rightCol)+1, csd)
			g.Generate(b, cs, int(row), int(leftCol)-1, csd)
			// This should clear the cross set on the just played tile.
			g.Generate(b, cs, int(row), int(colStart), csd)
		}
	}

	// assumes self is HORIZONTAL
	calcForSelf := func(rowStart int, colStart int, csd cgboard.BoardDirection) {
		// Generate cross-sets on either side of the word.
		for col := int(colStart) - 1; col <= int(colStart)+len(m.Tiles()); col++ {
			g.Generate(b, cs, int(rowStart), col, csd)
		}
	}

	if vertical {
		calcForAcross(row, col, Horizontal)
		b.Transpose()
		row, col = col, row
		calcForSelf(row, col, Vertical)
		b.Transpose()
	} else {
		calcForSelf(row, col, Horizontal)
		b.Transpose()
		row, col = col, row
		calcForAcross(row, col, Vertical)
		b.Transpose()
	}
}

// ----------------------------------------------------------------------
// Use a CrossScoreOnlyGenerator when you don't need cross sets

type CrossScoreOnlyGenerator struct {
	Dist *alphabet.LetterDistribution
}

func (g CrossScoreOnlyGenerator) Generate(b *Board, cs *BoardCrossSets, row int, col int, dir cgboard.BoardDirection) {
	genCrossScore(b, row, col, dir, g.Dist)
}

func (g CrossScoreOnlyGenerator) GenerateAll(b *Board, cs *BoardCrossSets) {
	generateAll(g, b, cs)
}

func (g CrossScoreOnlyGenerator) UpdateForMove(b *Board, cs *BoardCrossSets, m *move.Move) {
	updateForMove(g, b, cs, m)
}

// Wrapper functions to save rewriting all the tests

func GenAllCrossScores(b *Board, cs *BoardCrossSets, ld *alphabet.LetterDistribution) {
	gen := CrossScoreOnlyGenerator{Dist: ld}
	gen.GenerateAll(b, cs)
}

// ----------------------------------------------------------------------
// Implementation for CrossScoreOnlyGenerator

func genCrossScore(b *Board, row int, col int, dir cgboard.BoardDirection,
	ld *alphabet.LetterDistribution) {
	if row < 0 || row >= b.Dim() || col < 0 || col >= b.Dim() {
		return
	}
	// If the square has a letter in it, its cross set and cross score
	// should both be 0
	if b.HasLetter(row, col) {
		b.SetCrossScore(row, col, 0, dir)
		return
	}
	// If there's no tile adjacent to this square in any direction,
	// every letter is allowed.
	if b.LeftAndRightEmpty(row, col) {
		b.SetCrossScore(row, col, 0, dir)
		return
	}
	// If we are here, there is a letter to the left, to the right, or both.
	// start from the right and go backwards.
	rightCol := b.WordEdge(row, col+1, Right)
	if rightCol == col {
		score := b.TraverseBackwardsForScore(row, col-1, ld)
		b.SetCrossScore(row, col, score, dir)
	} else {
		// Otherwise, the right is not empty. Check if the left is empty,
		// if so we just traverse right, otherwise, we try every letter.
		scoreR := b.TraverseBackwardsForScore(row, rightCol, ld)
		scoreL := b.TraverseBackwardsForScore(row, col-1, ld)
		b.SetCrossScore(row, col, scoreR+scoreL, dir)
	}
}

// ----------------------------------------------------------------------
// GaddagCrossSetGenerator generates cross sets via a gaddag

type GaddagCrossSetGenerator struct {
	Dist   *alphabet.LetterDistribution
	Gaddag gaddag.GenericDawg
}

func (g GaddagCrossSetGenerator) Generate(b *Board, cs *BoardCrossSets, row int, col int, dir cgboard.BoardDirection) {
	GenCrossSet(b, cs, row, col, dir, g.Gaddag, g.Dist)
}

func (g GaddagCrossSetGenerator) GenerateAll(b *Board, cs *BoardCrossSets) {
	generateAll(g, b, cs)
}

func (g GaddagCrossSetGenerator) UpdateForMove(b *Board, cs *BoardCrossSets, m *move.Move) {
	updateForMove(g, b, cs, m)
}

// Wrapper functions to save rewriting all the tests

func GenAllCrossSets(b *Board, cs *BoardCrossSets, gd gaddag.GenericDawg, ld *alphabet.LetterDistribution) {
	gen := GaddagCrossSetGenerator{Dist: ld, Gaddag: gd}
	gen.GenerateAll(b, cs)
}

func UpdateCrossSetsForMove(b *Board, cs *BoardCrossSets, m *move.Move,
	gd gaddag.GenericDawg, ld *alphabet.LetterDistribution) {
	gen := GaddagCrossSetGenerator{Dist: ld, Gaddag: gd}
	gen.UpdateForMove(b, cs, m)
}

// ----------------------------------------------------------------------
// Implementation for GaddagCrossSetGenerator

func traverseBackwards(b *Board, cs *BoardCrossSets, row int, col int,
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
func GenCrossSet(b *Board, cs *BoardCrossSets, row int, col int, dir cgboard.BoardDirection,
	gaddag gaddag.GenericDawg, ld *alphabet.LetterDistribution) {

	log.Debug().Int("row", row).Int("col", col).Int("dim", b.Dim()).Msg("gcs")
	if row < 0 || row >= b.Dim() || col < 0 || col >= b.Dim() {
		return
	}
	// If the square has a letter in it, its cross set and cross score
	// should both be 0
	if b.HasLetter(row, col) {
		b.SetCrossScore(row, col, 0, dir)
		cs.SetCrossSet(row, col, CrossSet(0), dir)
		return
	}
	// If there's no tile adjacent to this square in any direction,
	// every letter is allowed.
	if b.LeftAndRightEmpty(row, col) {
		b.SetCrossScore(row, col, 0, dir)
		cs.SetCrossSet(row, col, TrivialCrossSet, dir)
		return
	}
	// If we are here, there is a letter to the left, to the right, or both.
	// start from the right and go backwards.
	rightCol := b.WordEdge(row, col+1, Right)
	if rightCol == col {
		// This means the right was always empty; we only want to go left.
		lNodeIdx, lPathValid := traverseBackwards(b, cs, row, col-1,
			gaddag.GetRootNodeIndex(), false, 0, gaddag)
		score := b.TraverseBackwardsForScore(row, col-1, ld)
		b.SetCrossScore(row, col, score, dir)

		if !lPathValid {
			// There are no further extensions to the word on the board,
			// which may also be a phony.
			cs.SetCrossSet(row, col, CrossSet(0), dir)
			return
		}
		// Otherwise, we have a left node index.
		sIdx := gaddag.NextNodeIdx(lNodeIdx, alphabet.SeparationMachineLetter)
		// Take the letter set of this sIdx as the cross-set.
		letterSet := gaddag.GetLetterSet(sIdx)
		// Miraculously, letter sets and cross sets are compatible.
		log.Debug().Msgf("setting crossset to %v", letterSet)
		cs.SetCrossSet(row, col, CrossSet(letterSet), dir)
	} else {

		// Otherwise, the right is not empty. Check if the left is empty,
		// if so we just traverse right, otherwise, we try every letter.
		leftCol := b.WordEdge(row, col-1, Left)
		log.Debug().Msgf("leftCol=%d, rightCol=%d", leftCol, rightCol)
		// Start at the right col and work back to this square.
		lNodeIdx, lPathValid := traverseBackwards(b, cs, row, rightCol,
			gaddag.GetRootNodeIndex(), false, 0, gaddag)
		log.Debug().Msgf("lpathvalid %v", lPathValid)
		scoreR := b.TraverseBackwardsForScore(row, rightCol, ld)
		scoreL := b.TraverseBackwardsForScore(row, col-1, ld)
		log.Debug().Msgf("scores %v %v dir %v row col %v %v", scoreR, scoreL, dir, row, col)
		b.SetCrossScore(row, col, scoreR+scoreL, dir)
		if !lPathValid {
			cs.SetCrossSet(row, col, CrossSet(0), dir)
			return
		}
		if leftCol == col {
			// The left is empty, but the right isn't.
			// The cross-set is just the letter set of the letter directly
			// to our right.

			letterSet := gaddag.GetLetterSet(lNodeIdx)
			log.Debug().Msgf("l setting crossset to %v", letterSet)
			cs.SetCrossSet(row, col, CrossSet(letterSet), dir)
		} else {
			// Both the left and the right have a tile. Go through the
			// siblings, from the right, to see what nodes lead to the left.

			numArcs := gaddag.NumArcs(lNodeIdx)
			cs.SetCrossSet(row, col, CrossSet(0), dir)
			log.Debug().Msgf("numArcs %v", numArcs)
			for i := lNodeIdx + 1; i <= uint32(numArcs)+lNodeIdx; i++ {

				ml := alphabet.MachineLetter(gaddag.Nodes()[i] >>
					gaddagmaker.LetterBitLoc)

				if ml == alphabet.SeparationMachineLetter {
					continue
				}
				nnIdx := gaddag.Nodes()[i] & gaddagmaker.NodeIdxBitMask
				_, success := traverseBackwards(b, cs, row, col-1, nnIdx, true,
					leftCol, gaddag)
				if success {
					log.Debug().Msgf("m setting crossset to %v", ml)
					// XXX this needs to be OR
					cs.AddCrossSet(row, col, ml, dir)
				}
			}
		}
	}
	log.Debug().Msg("leaving")

}
