package conversions

import (
	"fmt"
	"time"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tinymove"
)

func SmallMoveToMove(sm tinymove.SmallMove, m *move.Move, tm *tilemapping.TileMapping,
	bd *board.GameBoard, onTurnRack *tilemapping.Rack) {
	TinyMoveToMove(sm.TinyMove(), bd, m)
	// populate move with missing fields.
	m.SetAlphabet(tm)
	leave, err := tilemapping.Leave(onTurnRack.TilesOn(), m.Tiles(), false)
	if err != nil {
		// this is happening very rarely.. figure out wtf is going on.
		fmt.Println("Trying to convert small move to move did not succeed")
		fmt.Printf("sm: %v m: %v tm: %v\n", sm, m, tm)
		fmt.Printf("rack: %v\n", onTurnRack.TilesOn())
		fmt.Printf("m.Tiles: %v\n", m.Tiles())
		fmt.Println("board")
		fmt.Println(bd.ToDisplayText(tm))
		fmt.Println(time.Now())
		panic(err)
	}
	m.SetLeave(leave)
	m.SetScore(int(sm.Score()))
}

func MoveToTinyMove(m *move.Move) tinymove.TinyMove {
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
	return tinymove.TinyMove(moveCode)
}

// TinyMoveToMove creates a very minimal Move from the TinyMove code.
// This return value does not contain score info, leave info, alphabet info,
// etc. It's up to the caller to use a good scheme to compare it to an existing
// move. It should not be used directly on a board!
func TinyMoveToMove(t tinymove.TinyMove, b *board.GameBoard, om *move.Move) {
	if t == 0 {
		om.Set(nil, nil, 0, 0, 0, 0, false, move.MoveTypePass, nil)
		return
	}
	// assume it's a tile play move
	row := int(t&tinymove.RowBitMask) >> 6
	col := int(t&tinymove.ColBitMask) >> 1
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
	blankMask := int(t & tinymove.BlanksBitMask)

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
		if tidx > 6 {
			break
		}
		shifted := uint64(t) & tinymove.TBitMasks[tidx]

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
	}
	om.Set(mls, nil, 0, row, col, tidx, vert, move.MoveTypePlay, nil)
}

func TinyMoveToFullMove(t tinymove.TinyMove, bd *board.GameBoard, ld *tilemapping.LetterDistribution,
	onTurnRack *tilemapping.Rack) (*move.Move, error) {

	m := &move.Move{}
	TinyMoveToMove(t, bd, m)
	// populate move with missing fields.
	m.SetAlphabet(ld.TileMapping())

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
		bd.Transpose()
	}

	m.SetScore(bd.ScoreWord(m.Tiles(), r, c, m.TilesPlayed(), crossDir, ld))

	if v {
		bd.Transpose()
	}

	return m, nil
}
