// We move cross scores into its own package as the cwgame repo will
// only need cross score. It shouldn't know anything about cross sets.
// This function will also define some common functions to both cross sets
// and cross scores.
package crosses

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

type Board = board.GameBoard

const (
	Left       = board.LeftDirection
	Right      = board.RightDirection
	Horizontal = board.HorizontalDirection
	Vertical   = board.VerticalDirection
)

// CrossScore implements crosser.
type CrossScore struct {
	board *Board
}

// SetAll is a no-op
func (c *CrossScore) SetAll() {}

// Add is a no-op
func (c *CrossScore) Add(row, col int, ml alphabet.MachineLetter, dir board.BoardDirection) {
}

func (c *CrossScore) Set(row, col int, crossinfo int64, dir board.BoardDirection) {
	c.board.SetCrossScore(row, col, int(crossinfo), dir)
}

// ----------------------------------------------------------------------
// Use a CrossScoreOnlyGenerator when you don't need cross sets

type CrossScoreOnlyGenerator struct {
	Dist *alphabet.LetterDistribution
}

func (g CrossScoreOnlyGenerator) Generate(b *Board, cs Crosser, row int, col int,
	dir board.BoardDirection) {
	// genCrossScore can be implemented with a Crosser, and probably should,
	// but there's no point for now (it would probably be a bit of a performance hit).
	genCrossScore(b, row, col, dir, g.Dist)
}

func (g CrossScoreOnlyGenerator) GenerateAll(b *Board, cs Crosser) {
	GenerateAll(g, b, cs)
}

func (g CrossScoreOnlyGenerator) UpdateForMove(b *Board, cs Crosser, m *move.Move) {
	UpdateForMove(g, b, cs, m)
}

func GenAllCrossScores(b *Board, ld *alphabet.LetterDistribution) {
	gen := CrossScoreOnlyGenerator{Dist: ld}
	// Pass in a nil "Crosser", since this implementation actually looks
	// directly in the board and modifies it. This could probably be moved
	// to a Crosser that does the same thing.
	gen.GenerateAll(b, nil)
}

func genCrossScore(b *Board, row int, col int, dir board.BoardDirection,
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

// generateAll generates all crosses. It goes through the entire
// board; our anchor algorithm doesn't quite match the one in the Gordon
// paper.
// We do this for both transpositions of the board.
func GenerateAll(g iGenerator, b *Board, cs Crosser) {
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

// Public crosses.Generator Interface
// There is one concrete implementations below:
// - CrossScoreOnlyGenerator{Dist}
// There is another implementation in an external package:
// (but we really shouldn't care about this)
// - GaddagCrossSetGenerator{Dist, Gaddag}

type Generator interface {
	Generate(b *Board, cs Crosser, row int, col int, dir board.BoardDirection)
	GenerateAll(b *Board, cs Crosser)
	UpdateForMove(b *Board, cs Crosser, m *move.Move)
}

// We have to go through this dance since go will not let us simply provide
// Generator with default implementations of GenerateAll and UpdateForMove that
// call a given implementation of Generate.

type iGenerator interface {
	Generate(b *Board, cs Crosser, row int, col int, dir board.BoardDirection)
}

func UpdateForMove(g iGenerator, b *Board, cs Crosser, m *move.Move) {

	row, col, vertical := m.CoordsAndVertical()
	// Every tile placed by this new move creates new "across" words, and we need
	// to update the cross sets on both sides of these across words, as well
	// as the cross sets for THIS word.

	// Assumes all across words are HORIZONTAL.
	calcForAcross := func(rowStart int, colStart int, csd board.BoardDirection) {
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
	calcForSelf := func(rowStart int, colStart int, csd board.BoardDirection) {
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