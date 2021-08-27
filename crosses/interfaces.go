package crosses

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

// Crosser is an interface that should be implemented by something that
// wishes to extend this package.
type Crosser interface {
	SetAll()
	Add(row, col int, ml alphabet.MachineLetter, dir board.BoardDirection)
	Set(row, col int, crossinfo int64, dir board.BoardDirection)
}

// Generator: Public crosses.Generator Interface
// There is one concrete implementations below:
// - CrossScoreOnlyGenerator{Dist}
// There is another implementation in an external package:
// (but we really shouldn't care about this)
// - GaddagCrossSetGenerator{Dist, Gaddag}
type Generator interface {
	Generate(b *Board, row int, col int, dir board.BoardDirection)
	GenerateAll(b *Board)
	UpdateForMove(b *Board, m *move.Move)
}

// iGenerator: We have to go through this dance since go will not let us simply provide
// Generator with default implementations of GenerateAll and UpdateForMove that
// call a given implementation of Generate.
type iGenerator interface {
	Generate(b *Board, row int, col int, dir board.BoardDirection)
}
