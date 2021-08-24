package crosses

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
)

// Crosser is an interface that should be implemented by something that
// wishes to extend this package.
type Crosser interface {
	SetAll()
	Add(row, col int, ml alphabet.MachineLetter, dir board.BoardDirection)
	Set(row, col int, crossinfo int64, dir board.BoardDirection)
}
