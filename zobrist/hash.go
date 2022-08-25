package zobrist

import "lukechampine.com/frand"

const bignum = 1 << 63

// generate a zobrist hash for a crossword game position.
// https://en.wikipedia.org/wiki/Zobrist_hashing
type Zobrist struct {
	p2ToMove int64
	// note that the endgame only cares about 2 zeros in a row.
	// if we use this hash for non-endgame positions we might
	// want to turn this into an array as well, with each element
	// corresponding to the number of zero positions.
	lastMoveWasZero int64

	posTable  [][]int64
	rackTable [][]int64
}

func (z *Zobrist) Initialize(boardDim int, numtiles int, numblankletters int) {

	z.posTable = make([][]int64, boardDim*boardDim)
	for i := 0; i < boardDim*boardDim; i++ {
		// for example, 27 types of tiles + 26 letters the blank can be
		// notice that in norwegian, the blank can be a tile that's not
		// one of the tiles in the distribution.
		z.posTable[i] = make([]int64, numtiles+numblankletters)
		for j := 0; j < numtiles+numblankletters; j++ {
			z.posTable[i][j] = int64(frand.Uint64n(bignum))
		}
	}
	z.rackTable = make([][]int64, 7 /*this should be a constant somewhere maybe...*/)
	for i := 0; i < 7; i++ {
		z.rackTable[i] = make([]int64, numtiles)
		for j := 0; j < numtiles; j++ {
			z.rackTable[i][j] = int64(frand.Uint64n(bignum))
		}
	}
	z.p2ToMove = int64(frand.Uint64n(bignum))
	z.lastMoveWasZero = int64(frand.Uint64n(bignum))

}
