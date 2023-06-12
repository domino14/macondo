package zobrist

import (
	"lukechampine.com/frand"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
)

const bignum = 1<<63 - 2

// generate a zobrist hash for a crossword game position.
// https://en.wikipedia.org/wiki/Zobrist_hashing
type Zobrist struct {
	theirTurn uint64

	posTable       [][]uint64
	ourRackTable   [][]uint64
	theirRackTable [][]uint64
	scorelessTurns [3]uint64

	boardDim        int
	placeholderRack []tilemapping.MachineLetter
}

const MaxLetters = 35

func (z *Zobrist) Initialize(boardDim int) {
	z.boardDim = boardDim
	z.posTable = make([][]uint64, boardDim*boardDim)
	for i := 0; i < boardDim*boardDim; i++ {

		z.posTable[i] = make([]uint64, MaxLetters*2)
		for j := 0; j < 70; j++ {
			z.posTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}
	z.ourRackTable = make([][]uint64, MaxLetters)
	for i := 0; i < MaxLetters; i++ {
		z.ourRackTable[i] = make([]uint64, game.RackTileLimit)
		for j := 0; j < game.RackTileLimit; j++ {
			z.ourRackTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}
	z.theirRackTable = make([][]uint64, MaxLetters)
	for i := 0; i < MaxLetters; i++ {
		z.theirRackTable[i] = make([]uint64, game.RackTileLimit)
		for j := 0; j < game.RackTileLimit; j++ {
			z.theirRackTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}

	for i := 0; i < 3; i++ {
		z.scorelessTurns[i] = frand.Uint64n(bignum) + 1
	}

	z.theirTurn = frand.Uint64n(bignum) + 1
	z.placeholderRack = make([]tilemapping.MachineLetter, MaxLetters)
}

// https://stackoverflow.com/a/12996028/1737333
func hashUint64(x uint64) uint64 {
	x = (x ^ (x >> 30)) * uint64(0xbf58476d1ce4e5b9)
	x = (x ^ (x >> 27)) * uint64(0x94d049bb133111eb)
	x = x ^ (x >> 31)
	return x
}

func (z *Zobrist) Hash(squares tilemapping.MachineWord,
	ourRack, theirRack *tilemapping.Rack, theirTurn bool, scorelessTurns int,
	ourSpread int) uint64 {

	key := uint64(0)
	for i, letter := range squares {
		if letter == 0 {
			continue
		}
		if letter&0x80 > 0 {
			// it's a blank
			letter = (letter & (0x7F)) + MaxLetters
		}
		key ^= z.posTable[i][letter]
	}
	for i, ct := range ourRack.LetArr {
		key ^= z.ourRackTable[i][ct]
	}
	for i, ct := range theirRack.LetArr {
		key ^= z.theirRackTable[i][ct]
	}
	if theirTurn {
		key ^= z.theirTurn
	}
	key ^= z.scorelessTurns[scorelessTurns]
	key ^= hashUint64(uint64(ourSpread))
	return key
}

func (z *Zobrist) AddMove(key uint64, m move.PlayMaker, wasOurMove bool,
	scorelessTurns, lastScorelessTurns int, ourSpread, lastOurSpread int) uint64 {

	// Adding a move:
	// For every letter in the move (assume it's only a tile placement move
	// or a pass for now):
	// - XOR with its position on the board
	// - XOR with the "position" on the rack hash
	// Then:
	// - XOR with p2ToMove since we always alternate

	rackTable := z.ourRackTable
	if !wasOurMove {
		rackTable = z.theirRackTable
	}
	if m.Type() == move.MoveTypePlay {
		row, col, vertical := m.RowStart(), m.ColStart(), m.Vertical()
		ri, ci := 0, 1
		if vertical {
			ri, ci = 1, 0
		}
		// clear out placeholder rack first:
		for i := 0; i < MaxLetters; i++ {
			z.placeholderRack[i] = 0
		}

		for idx, tile := range m.Tiles() {
			newRow := row + (ri * idx)
			newCol := col + (ci * idx)
			if tile == 0 {
				// 0 is a played-through marker if it's part of a move's tiles
				continue
			}
			boardTile := tile
			if tile&0x80 > 0 {
				// it's a blank
				boardTile = (tile & (0x7F)) + MaxLetters
			}
			key ^= z.posTable[newRow*z.boardDim+newCol][boardTile]
			// build up placeholder rack.
			tileIdx := tile.IntrinsicTileIdx()
			z.placeholderRack[tileIdx]++
		}
		for _, tile := range m.Leave() {
			z.placeholderRack[tile]++
		}
		// now "Play" all the tiles in the rack
		for _, tile := range m.Tiles() {
			if tile == 0 {
				// this is a play-through tile, if it's in a move. Ignore it.
				continue
			}
			tileIdx := tile.IntrinsicTileIdx()
			key ^= rackTable[tileIdx][z.placeholderRack[tileIdx]]
			z.placeholderRack[tileIdx]--
			key ^= rackTable[tileIdx][z.placeholderRack[tileIdx]]
		}
		// don't bother hashing the spread for now; delete soon.
		// key ^= uint64(hashUint64(uint64(lastOurSpread)))
		// key ^= uint64(hashUint64(uint64(ourSpread)))
	}
	if lastScorelessTurns != scorelessTurns {
		key ^= z.scorelessTurns[lastScorelessTurns]
		key ^= z.scorelessTurns[scorelessTurns]
	}

	key ^= z.theirTurn
	return key
}
