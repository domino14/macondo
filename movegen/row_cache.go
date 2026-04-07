package movegen

import (
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
)

const maxBoardDim = board.MaxBoardDim

// cachedSquare holds the frequently-accessed board data for one square,
// packed into a single struct for cache-friendly access during move generation.
type cachedSquare struct {
	crossSet   board.CrossSet
	crossScore int
	letterMul  int
	wordMul    int
	letter     tilemapping.MachineLetter
}

// rowCache stores per-square data for the current row being processed.
// Populated once per row (or per anchor in shadow best-first mode),
// replacing scattered reads from separate board arrays.
type rowCache struct {
	squares   [maxBoardDim]cachedSquare
	loadedRow int // -1 if not loaded
	loadedDir board.BoardDirection
}

// loadRow populates the cache from the board for the given row and
// cross-set direction. After this call, all board lookups for this row
// can use the cache instead.
func (rc *rowCache) loadRow(b *board.GameBoard, row int, csDir board.BoardDirection, dim int) {
	for col := 0; col < dim; col++ {
		sqIdx := b.GetSqIdx(row, col)
		rc.squares[col] = cachedSquare{
			crossSet:   b.GetCrossSetIdx(sqIdx, csDir),
			crossScore: b.GetCrossScoreIdx(sqIdx, csDir),
			letterMul:  b.GetLetterMultiplier(sqIdx),
			wordMul:    b.GetWordMultiplier(sqIdx),
			letter:     b.GetLetter(row, col),
		}
	}
	rc.loadedRow = row
	rc.loadedDir = csDir
}
