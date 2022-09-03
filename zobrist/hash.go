package zobrist

import (
	"lukechampine.com/frand"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
)

const bignum = 1<<63 - 2

// generate a zobrist hash for a crossword game position.
// https://en.wikipedia.org/wiki/Zobrist_hashing
type Zobrist struct {
	minimizingPlayerToMove uint64

	posTable  [][]uint64
	rackTable [][]uint64

	boardDim        int
	placeholderRack []alphabet.MachineLetter
}

func (z *Zobrist) Initialize(boardDim int) {
	z.boardDim = boardDim
	z.posTable = make([][]uint64, boardDim*boardDim)
	for i := 0; i < boardDim*boardDim; i++ {

		// 160 is MaxAlphabetSize + BlankOffset + some fudge factor.
		// This is kind of ugly; we should clarify the domain a bit.
		// We don't lose a huge deal by generating this many random numbers,
		// however, even if we're not using all of them.
		z.posTable[i] = make([]uint64, 160)
		for j := 0; j < 160; j++ {
			z.posTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}
	z.rackTable = make([][]uint64, alphabet.MaxAlphabetSize+1)
	for i := 0; i < alphabet.MaxAlphabetSize+1; i++ {
		z.rackTable[i] = make([]uint64, game.RackTileLimit)
		for j := 0; j < game.RackTileLimit; j++ {
			z.rackTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}
	z.minimizingPlayerToMove = frand.Uint64n(bignum) + 1

	z.placeholderRack = make([]alphabet.MachineLetter, alphabet.MaxAlphabetSize+1)
}

func (z *Zobrist) Hash(squares alphabet.MachineWord, onTurnRack *alphabet.Rack,
	minimizingPlayerToMove bool) uint64 {

	key := uint64(0)
	for i, letter := range squares {
		key ^= z.posTable[i][letter]
	}
	for i, ct := range onTurnRack.LetArr {
		key ^= z.rackTable[i][ct]
	}

	if minimizingPlayerToMove {
		key ^= z.minimizingPlayerToMove
	}
	return key
}

func (z *Zobrist) AddMove(key uint64, m *move.Move, unplay bool) uint64 {
	// Adding a move:
	// For every letter in the move (assume it's only a tile placement move
	// or a pass for now):
	// - XOR with its position on the board
	// - XOR with the "position" on the rack hash
	// Then:
	// - XOR with p2ToMove since we always alternate
	// - XOR with lastMoveWasZero if it's a pass (or a zero-score...)
	// XXX: as a side note, could there be an edge condition with the endgame
	// where the best move might be to play a zero-point blank play, AND
	// we are losing the game, so that the opponent may want to pass back
	// and end the game right away?
	if m.Action() == move.MoveTypePlay {
		row, col, vertical := m.CoordsAndVertical()
		ri, ci := 0, 1
		if vertical {
			ri, ci = 1, 0
		}
		// clear out placeholder rack first:
		for i := 0; i < alphabet.MaxAlphabetSize+1; i++ {
			z.placeholderRack[i] = 0
		}

		for idx, tile := range m.Tiles() {
			newRow := row + (ri * idx)
			newCol := col + (ci * idx)
			if tile == alphabet.PlayedThroughMarker {
				continue
			}
			key ^= z.posTable[newRow*z.boardDim+newCol][tile]
			if unplay {
				// If we are unplaying this move, don't add it
				// to the "placeholderRack".
				continue
			}
			// build up placeholder rack.
			tileIdx, isPlayedTile := tile.IntrinsicTileIdx()
			if !isPlayedTile {
				// isPlayedTile should never be false here, since
				// the PlayedThroughMarker case would have been handled
				// above.
				panic("unexpected isPlayedTile")
			}
			z.placeholderRack[tileIdx]++
		}
		for _, tile := range m.Leave() {
			tileIdx, isPlayedTile := tile.IntrinsicTileIdx()
			if !isPlayedTile {
				panic("unexpected isPlayedTile during leave hashing")
			}
			z.placeholderRack[tileIdx]++
		}
		// now "Play" all the tiles in the rack
		for _, tile := range m.Tiles() {
			tileIdx, isPlayedTile := tile.IntrinsicTileIdx()
			if tile == alphabet.PlayedThroughMarker {
				continue
			}
			if !isPlayedTile {
				// isPlayedTile should never be false here, since
				// the PlayedThroughMarker case would have been handled
				// above.
				panic("unexpected isPlayedTile - 2nd go")
			}

			key ^= z.rackTable[tileIdx][z.placeholderRack[tileIdx]]
			if !unplay {
				z.placeholderRack[tileIdx]--
			} else {
				z.placeholderRack[tileIdx]++
			}

			key ^= z.rackTable[tileIdx][z.placeholderRack[tileIdx]]
		}

	} else if m.Action() == move.MoveTypePass {

		for i := 0; i < alphabet.MaxAlphabetSize+1; i++ {
			z.placeholderRack[i] = 0
		}

		for _, tile := range m.Leave() {
			tileIdx, isPlayedTile := tile.IntrinsicTileIdx()
			if !isPlayedTile {
				panic("unexpected isPlayedTile during leave hashing")
			}
			z.placeholderRack[tileIdx]++
		}

		for _, tile := range m.Leave() {
			tileIdx, isPlayedTile := tile.IntrinsicTileIdx()
			if !isPlayedTile {
				// isPlayedTile should never be false here, since
				// this is a leave.
				panic("unexpected isPlayedTile - during pass")
			}

			key ^= z.rackTable[tileIdx][z.placeholderRack[tileIdx]]
		}

	}
	key ^= z.minimizingPlayerToMove
	return key
}
