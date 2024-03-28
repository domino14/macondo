package zobrist

import (
	"encoding/json"
	"io"

	"github.com/domino14/word-golib/tilemapping"
	"lukechampine.com/frand"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/tinymove"
)

const bignum = 1<<63 - 2

// generate a zobrist hash for a crossword game position.
// https://en.wikipedia.org/wiki/Zobrist_hashing
type Zobrist struct {
	TheirTurn uint64

	PosTable       [][]uint64
	OurRackTable   [][]uint64
	TheirRackTable [][]uint64
	ScorelessTurns [3]uint64

	boardDim int
}

const MaxLetters = 35

func (z *Zobrist) Initialize(boardDim int) {
	z.boardDim = boardDim
	z.PosTable = make([][]uint64, boardDim*boardDim)
	for i := 0; i < boardDim*boardDim; i++ {

		z.PosTable[i] = make([]uint64, MaxLetters*2)
		for j := 0; j < 70; j++ {
			z.PosTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}
	z.OurRackTable = make([][]uint64, MaxLetters)
	for i := 0; i < MaxLetters; i++ {
		z.OurRackTable[i] = make([]uint64, game.RackTileLimit)
		for j := 0; j < game.RackTileLimit; j++ {
			z.OurRackTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}
	z.TheirRackTable = make([][]uint64, MaxLetters)
	for i := 0; i < MaxLetters; i++ {
		z.TheirRackTable[i] = make([]uint64, game.RackTileLimit)
		for j := 0; j < game.RackTileLimit; j++ {
			z.TheirRackTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}

	for i := 0; i < 3; i++ {
		z.ScorelessTurns[i] = frand.Uint64n(bignum) + 1
	}

	z.TheirTurn = frand.Uint64n(bignum) + 1
}

func (z *Zobrist) Dump(w io.Writer) {
	bts, err := json.Marshal(z)
	if err != nil {
		panic(err)
	}
	_, err = w.Write(bts)
	if err != nil {
		panic(err)
	}
}

func (z *Zobrist) Hash(squares tilemapping.MachineWord,
	ourRack, theirRack *tilemapping.Rack, theirTurn bool, scorelessTurns int) uint64 {

	key := uint64(0)
	for i, letter := range squares {
		if letter == 0 {
			continue
		}
		if letter&0x80 > 0 {
			// it's a blank
			letter = (letter & (0x7F)) + MaxLetters
		}
		key ^= z.PosTable[i][letter]
	}
	for i, ct := range ourRack.LetArr {
		key ^= z.OurRackTable[i][ct]
	}
	for i, ct := range theirRack.LetArr {
		key ^= z.TheirRackTable[i][ct]
	}
	if theirTurn {
		key ^= z.TheirTurn
	}
	key ^= z.ScorelessTurns[scorelessTurns]
	return key
}

func (z *Zobrist) AddMove(key uint64, m *tinymove.SmallMove,
	moveRack *tilemapping.Rack, moveTiles *[board.MaxBoardDim]tilemapping.MachineLetter,
	wasOurMove bool, ScorelessTurns, lastScorelessTurns int) uint64 {

	// Adding a move:
	// For every letter in the move (assume it's only a tile placement move
	// or a pass for now):
	// - XOR with its position on the board
	// - XOR with the "position" on the rack hash
	// Then:
	// - XOR with p2ToMove since we always alternate

	rackTable := z.OurRackTable
	if !wasOurMove {
		rackTable = z.TheirRackTable
	}
	if !m.IsPass() {
		row, col, vertical := m.CoordsAndVertical()
		ri, ci := 0, 1
		if vertical {
			ri, ci = 1, 0
		}
		placeholderRack := [MaxLetters]tilemapping.MachineLetter{}

		for idx := 0; idx < m.PlayLength(); idx++ {
			tile := moveTiles[idx]
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
			key ^= z.PosTable[newRow*z.boardDim+newCol][boardTile]
			// build up placeholder rack.
			tileIdx := tile.IntrinsicTileIdx()
			placeholderRack[tileIdx]++
		}
		// moveRack contains the left-over tiles
		for idx := range moveRack.LetArr {
			for j := 0; j < moveRack.LetArr[idx]; j++ {
				placeholderRack[idx]++
			}
		}
		// now "Play" all the tiles in the rack
		for idx := 0; idx < m.PlayLength(); idx++ {
			tile := moveTiles[idx]

			if tile == 0 {
				// this is a play-through tile, if it's in a move. Ignore it.
				continue
			}
			tileIdx := tile.IntrinsicTileIdx()
			key ^= rackTable[tileIdx][placeholderRack[tileIdx]]
			placeholderRack[tileIdx]--
			key ^= rackTable[tileIdx][placeholderRack[tileIdx]]
		}
	}
	if lastScorelessTurns != ScorelessTurns {
		key ^= z.ScorelessTurns[lastScorelessTurns]
		key ^= z.ScorelessTurns[ScorelessTurns]
	}

	key ^= z.TheirTurn
	return key
}

func (z *Zobrist) BoardDim() int {
	return z.boardDim
}
