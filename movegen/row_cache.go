package movegen

import (
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
)

const maxBoardDim = board.MaxBoardDim

// cachedSquare holds the frequently-accessed board data for one square,
// packed for cache-friendly access during move generation.
//
// Layout (24 bytes):
//
//	crossSet   uint64  (8) — bit-set of allowed letters at this square
//	leftExtSet uint64  (8) — left-extension set for shadow play
//	crossScore int32   (4) — perp-word tile sum (well below int16 max
//	                        in practice, int32 chosen for headroom)
//	letter     uint8   (1) — board tile, 0 = empty, high bit = blank
//	letterMul  uint8   (1) — bonus square letter multiplier (1..4)
//	wordMul    uint8   (1) — bonus square word multiplier (1..4)
//	_padding   [1]byte (1) — implicit, brings total to 24
//
// At 24 bytes, an entire 15-board row of squares fits in 360 bytes
// (6 cache lines on a 64-byte line machine), still comfortably within
// L1 cache. Both recursiveGen (scoring phase) and the shadow play
// functions scan the row repeatedly; the smaller working set lets the
// L1 hold the whole row across iterations.
type cachedSquare struct {
	crossSet   board.CrossSet
	leftExtSet board.CrossSet
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
//
// The direction-dependent slices are resolved once before the loop so each
// square pays one direct slice index per field instead of a per-call
// direction switch. The bonuses slice is read once per square to derive both
// the letter multiplier and the word multiplier.
func (rc *rowCache) loadRow(b *board.GameBoard, row int, csDir board.BoardDirection, dim int) {
	crossSets := b.CrossSetsForDir(csDir)
	leftExtSets := b.LeftExtSetsForDir(csDir)
	crossScores := b.CrossScoresForDir(csDir)
	squares := b.SquaresSlice()
	bonuses := b.BonusesSlice()
	sqBase := b.GetSqIdx(row, 0) // row*rowMul; col*colMul accumulates below
	colMul := b.GetSqIdx(0, 1)   // == colMul (1 when normal, dim when transposed)
	for col := 0; col < dim; col++ {
		sqIdx := sqBase + col*colMul
		lm, wm := board.BonusMuls(bonuses[sqIdx])
		rc.squares[col] = cachedSquare{
			crossSet:   crossSets[sqIdx],
			leftExtSet: board.CrossSet(leftExtSets[sqIdx]),
			crossScore: int32(crossScores[sqIdx]),
			letterMul:  lm,
			wordMul:    wm,
			letter:     squares[sqIdx],
		}
	}
	rc.loadedRow = row
	rc.loadedDir = csDir
}
