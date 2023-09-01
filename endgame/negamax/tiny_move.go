package negamax

import (
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
)

// TinyMove is a 64-bit representation of a move. We can probably make it
// smaller at the cost of higher decoding. It is made to be as small as possible
// to fit it in a transposition table.
type TinyMove uint64

// Schema:
// 42 bits (7 groups of 6 bits) representing each tile value in the move.
// 7 bit flags, representing whether the associated tile is a blank or not (1 = blank, 0 = no blank)
// 5 bits for row
// 5 bits for column
// 1 bit for horiz/vert (horiz = 0, vert = 1)

// If move is a pass, the entire value is 0.
// 63   59   55   51   47   43   39   35   31   27   23   19   15   11    7    3
//  xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
//    77 7777 6666 6655 5555 4444 4433 3333 2222 2211 1111  BBB BBBB  RRR RRCC CCCV

const ColBitMask = 0b00111110
const RowBitMask = 0b00000111_11000000
const BlanksBitMask = 127 << 12

var TBitMasks = [7]uint64{63 << 20, 63 << 26, 63 << 32, 63 << 38, 63 << 44, 63 << 50, 63 << 56}

func moveToTinyMove(m *move.Move) TinyMove {
	// convert a regular move to a tiny move.
	if m.Action() == move.MoveTypePass {
		return 0
	} else if m.Action() != move.MoveTypePlay {
		// only allow tile plays otherwise.
		return 0
	}
	// We are definitely in a tile play.
	var moveCode uint64
	tidx := 0
	bts := 20 // start at a bitshift of 20 for the first tile
	var blanksMask int

	for _, t := range m.Tiles() {
		if t == 0 {
			// Play-through tile
			continue
		}
		it := t.IntrinsicTileIdx()
		val := t
		if it == 0 {
			blanksMask |= (1 << tidx)
			// this would be a designated blank
			val = t.Unblank()
		}

		moveCode |= (uint64(val) << bts)

		tidx++
		bts += 6
	}
	row, col, vert := m.CoordsAndVertical()
	if vert {
		moveCode |= 1
	}
	moveCode |= (uint64(col) << 1)
	moveCode |= (uint64(row) << 6)
	moveCode |= (uint64(blanksMask) << 12)
	return TinyMove(moveCode)
}

// tinyMoveToMove creates a very minimal Move from the TinyMove code.
// This return value does not contain score info, leave info, alphabet info,
// etc. It's up to the caller to use a good scheme to compare it to an existing
// move. It should not be used directly on a board!
func tinyMoveToMove(t TinyMove, b *board.GameBoard) *move.Move {
	if t == 0 {
		return move.NewPassMove(nil, nil)
	}
	// assume it's a tile play move
	row := int(t&RowBitMask) >> 6
	col := int(t&ColBitMask) >> 1
	vert := false
	if t&1 > 0 {
		vert = true
	}
	ri, ci := 0, 1
	if vert {
		ri, ci = 1, 0
	}
	bdim := b.Dim()
	r, c := row, col
	mls := []tilemapping.MachineLetter{}
	blankMask := int(t & BlanksBitMask)

	tidx := 0
	tileShift := 20
	outOfBounds := false
	for !outOfBounds {
		onBoard := b.GetLetter(r, c)
		r += ri
		c += ci
		if r >= bdim || c >= bdim {
			outOfBounds = true
		}

		if onBoard != 0 {
			mls = append(mls, 0)
			continue
		}

		shifted := uint64(t) & TBitMasks[tidx]

		tile := tilemapping.MachineLetter(shifted >> tilemapping.MachineLetter(tileShift))
		if tile == 0 {
			break
		}
		if blankMask&(1<<(tidx+12)) > 0 {
			tile = tile.Blank()
		}
		tidx++
		tileShift += 6

		mls = append(mls, tile)
		if tidx > 6 {
			break
		}
	}
	return move.NewScoringMove(0, mls, nil, vert, tidx, nil, row, col)
}

func tinyMoveToFullMove(t TinyMove, g *game.Game, onTurnRack *tilemapping.Rack) (*move.Move, error) {
	m := tinyMoveToMove(t, g.Board())
	// populate move with missing fields.
	m.SetAlphabet(g.Alphabet())

	leave, err := tilemapping.Leave(onTurnRack.TilesOn(), m.Tiles(), false)
	if err != nil {
		return nil, err
	}
	m.SetLeave(leave)
	// score the play
	r, c, v := m.CoordsAndVertical()

	crossDir := board.VerticalDirection
	if v {
		crossDir = board.HorizontalDirection
		r, c = c, r
		g.Board().Transpose()
	}

	m.SetScore(g.Board().ScoreWord(m.Tiles(), r, c, m.TilesPlayed(), crossDir, g.Bag().LetterDistribution()))

	if v {
		g.Board().Transpose()
	}

	return m, nil
}

func minimallyEqual(m1 *move.Move, m2 *move.Move) bool {
	if m1.Action() != m2.Action() {
		return false
	}
	if m1.TilesPlayed() != m2.TilesPlayed() {
		return false
	}
	if len(m1.Tiles()) != len(m2.Tiles()) {
		return false
	}
	r1, c1, v1 := m1.CoordsAndVertical()
	r2, c2, v2 := m2.CoordsAndVertical()
	if r1 != r2 || c1 != c2 || v1 != v2 {
		return false
	}
	for idx, i := range m1.Tiles() {
		if m2.Tiles()[idx] != i {
			return false
		}
	}
	return true
}
