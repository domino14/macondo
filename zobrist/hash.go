package zobrist

import (
	"lukechampine.com/frand"

	"github.com/domino14/macondo/alphabet"
)

const bignum = 1 << 63 - 2

// generate a zobrist hash for a crossword game position.
// https://en.wikipedia.org/wiki/Zobrist_hashing
type Zobrist struct {
	p2ToMove uint64
	posTable  [][]uint64
	rackTable [][]uint64
}

func (z *Zobrist) Initialize(boardDim int, numtiles int, numblankletters int) {

	z.posTable = make([][]uint64, boardDim*boardDim)
	for i := 0; i < boardDim*boardDim; i++ {
		// for example, 27 types of tiles + 26 letters the blank can be
		// notice that in norwegian, the blank can be a tile that's not
		// one of the tiles in the distribution.
		z.posTable[i] = make([]uint64, 256)
		for j := 0; j < 256; j++ {
			z.posTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}
	z.rackTable = make([][]uint64, 7 /*this should be a constant somewhere maybe...*/)
	for i := 0; i < 7; i++ {
		z.rackTable[i] = make([]uint64, numtiles)
		for j := 0; j < numtiles; j++ {
			z.rackTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}
	z.p2ToMove = frand.Uint64n(bignum) + 1
}

func (z *Zobrist) Hash(squares alphabet.MachineWord, leave alphabet.MachineWord) uint64 {
	key := z.p2ToMove
	for i, letter := range squares {
		key ^= z.posTable[i][letter]
	}
	for i, letter := range leave {
		key ^= z.rackTable[i][letter]
	}
	return key
}
