package movegen

import (
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
)

const maxBoardDim = board.MaxBoardDim

// cachedSquare holds the frequently-accessed board data for one square,
// packed for cache-friendly access during move generation.
//
// Layout (16 bytes, down from 40):
//
//	crossSet   uint64  (8) — bit-set of allowed letters at this square
//	crossScore int32   (4) — perp-word tile sum (well below int16 max
//	                        in practice, int32 chosen for headroom)
//	letter     uint8   (1) — board tile, 0 = empty, high bit = blank
//	letterMul  uint8   (1) — bonus square letter multiplier (1..4)
//	wordMul    uint8   (1) — bonus square word multiplier (1..4)
//	_padding   [1]byte (1) — implicit, brings total to 16
//
// At 16 bytes, an entire 15-board row of squares fits in 240 bytes
// (4 cache lines on Apple M2's 64-byte lines), down from 600 bytes
// (10 cache lines). recursiveGen + shadowPlayRight scan the row
// repeatedly, so the smaller working set lets the L1 hold the
// whole row easily.
type cachedSquare struct {
	crossSet   board.CrossSet
	crossScore int32
	letter     tilemapping.MachineLetter
	letterMul  uint8
	wordMul    uint8
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
			crossScore: int32(b.GetCrossScoreIdx(sqIdx, csDir)),
			letterMul:  uint8(b.GetLetterMultiplier(sqIdx)),
			wordMul:    uint8(b.GetWordMultiplier(sqIdx)),
			letter:     b.GetLetter(row, col),
		}
	}
	rc.loadedRow = row
	rc.loadedDir = csDir
}
