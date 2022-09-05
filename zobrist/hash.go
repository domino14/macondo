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

	posTable     [][]uint64
	maxRackTable [][]uint64 // rack for the maximizing player
	minRackTable [][]uint64 // rack for the minimizing player

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
	z.maxRackTable = make([][]uint64, alphabet.MaxAlphabetSize+1)
	for i := 0; i < alphabet.MaxAlphabetSize+1; i++ {
		z.maxRackTable[i] = make([]uint64, game.RackTileLimit)
		for j := 0; j < game.RackTileLimit; j++ {
			z.maxRackTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}
	z.minRackTable = make([][]uint64, alphabet.MaxAlphabetSize+1)
	for i := 0; i < alphabet.MaxAlphabetSize+1; i++ {
		z.minRackTable[i] = make([]uint64, game.RackTileLimit)
		for j := 0; j < game.RackTileLimit; j++ {
			z.minRackTable[i][j] = frand.Uint64n(bignum) + 1
		}
	}

	z.minimizingPlayerToMove = frand.Uint64n(bignum) + 1

	z.placeholderRack = make([]alphabet.MachineLetter, alphabet.MaxAlphabetSize+1)
}

func (z *Zobrist) Hash(squares alphabet.MachineWord, maxPlayerRack *alphabet.Rack,
	minPlayerRack *alphabet.Rack, minimizingPlayerToMove bool) uint64 {

	key := uint64(0)
	for i, letter := range squares {
		if letter == alphabet.EmptySquareMarker {
			continue
		}
		key ^= z.posTable[i][letter]
	}
	for i, ct := range maxPlayerRack.LetArr {
		key ^= z.maxRackTable[i][ct]
	}
	for i, ct := range minPlayerRack.LetArr {
		key ^= z.minRackTable[i][ct]
	}
	if minimizingPlayerToMove {
		key ^= z.minimizingPlayerToMove
	}
	return key
}

func (z *Zobrist) AddMove(key uint64, m *move.Move, maxPlayer bool) uint64 {
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

	ourRackTable := z.maxRackTable
	if !maxPlayer {
		ourRackTable = z.minRackTable
	}
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

			key ^= ourRackTable[tileIdx][z.placeholderRack[tileIdx]]
			z.placeholderRack[tileIdx]--
			key ^= ourRackTable[tileIdx][z.placeholderRack[tileIdx]]

		}

	} else if m.Action() == move.MoveTypePass {
		// it's just a pass. nothing else changes, except for the
		// minimizingPlayerToMove
	}
	// for i, ct := range otherPlayerRack.LetArr {
	// 	key ^= theirRackTable[i][ct]
	// }

	key ^= z.minimizingPlayerToMove
	return key
}
