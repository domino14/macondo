package move

import "github.com/domino14/macondo/tilemapping"

const (
	// constants used for the key for a minimal move.

	// layout
	// 32       24       16       8
	// xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
	// ssssssss ssssssss pppttdrr rrrccccc
	// s - score (2 ^ 16 - 1  max)
	// t - type
	// p - tiles played (7 max - phew!)
	// r - row (31 max)
	// c - col (31 max)

	mmRowShift         = 5
	mmVerticalShift    = 10
	mmTypeShift        = 11
	mmTilesPlayedShift = 13
	mmScoreShift       = 16

	mmColBitmask = (1 << 5) - 1
)

// MinimalMove should be used for our move generators.
type MinimalMove struct {
	// the tiles need to be kept separate.
	tiles tilemapping.MachineWord
	leave tilemapping.MachineWord
	// key encodes everything else we need to know about the move
	key uint32
}

// ShortDescription turns it into a regular move.
func (mm *MinimalMove) ShortDescription(alph *tilemapping.TileMapping) string {
	m := &Move{}
	mm.CopyToMove(m)
	m.SetAlphabet(alph)
	return m.ShortDescription()
}

// CopyToMove copies the minimal move to a move. If the move needs to be
// visually represented, it should contain an alphabet already.
func (mm *MinimalMove) CopyToMove(m *Move) {
	tk := (mm.key >> mmTypeShift) & 0b00000011
	var t MoveType
	switch tk {
	case 0:
		t = MoveTypePlay
	case 1:
		t = MoveTypePass
	case 2:
		t = MoveTypeExchange
	}
	m.Set(
		mm.tiles,
		mm.leave,
		int(mm.key>>mmScoreShift),
		int((mm.key>>mmRowShift)&mmColBitmask),
		int(mm.key)&mmColBitmask,
		int((mm.key>>mmTilesPlayedShift)&0b00000111),
		(mm.key>>mmVerticalShift)&1 == 1,
		t,
		m.alph,
	)
}

func (mm *MinimalMove) ColStart() int {
	return int(mm.key & mmColBitmask)
}

func (mm *MinimalMove) RowStart() int {
	return int((mm.key >> mmRowShift) & mmColBitmask)
}

func (mm *MinimalMove) Tiles() tilemapping.MachineWord {
	return mm.tiles
}

func (mm *MinimalMove) TilesPlayed() int {
	return int((mm.key >> mmTilesPlayedShift) & 0b00000111)
}

func (mm *MinimalMove) Leave() tilemapping.MachineWord {
	return mm.leave
}

func (mm *MinimalMove) Type() MoveType {
	tk := (mm.key >> mmTypeShift) & 0b00000011
	var t MoveType
	switch tk {
	case 0:
		t = MoveTypePlay
	case 1:
		t = MoveTypePass
	case 2:
		t = MoveTypeExchange
	}
	return t
}

func (mm *MinimalMove) Vertical() bool {
	return (mm.key>>mmVerticalShift)&1 == 1
}

func (mm *MinimalMove) Score() int {
	return int(mm.key >> mmScoreShift)
}

func (mm *MinimalMove) SetEmpty() {
	mm.key = 0
	mm.tiles = nil
	mm.leave = nil
}

func (mm *MinimalMove) Set(word, leave tilemapping.MachineWord, score, row, col int, vertical bool,
	t MoveType) {
	vbit := uint32(0)
	if vertical {
		vbit = 1
	}
	mm.key = uint32(col) + (uint32(row) << mmRowShift) +
		(vbit << mmVerticalShift) + (uint32(score) << mmScoreShift)
	mm.tiles = word
	mm.leave = leave
}

func (mm *MinimalMove) CopyFrom(f *MinimalMove) {
	mm.tiles = make([]tilemapping.MachineLetter, len(f.tiles))
	mm.leave = make([]tilemapping.MachineLetter, len(f.leave))
	copy(mm.tiles, f.tiles)
	copy(mm.leave, f.leave)
	mm.key = f.key
}

func (mm *MinimalMove) Copy() *MinimalMove {
	c := &MinimalMove{}
	c.CopyFrom(mm)
	return c
}

func (mm *MinimalMove) Equals(o *MinimalMove) bool {
	if o == nil {
		return false
	}
	if mm.key != o.key {
		return false
	}
	if len(mm.tiles) != len(o.tiles) {
		return false
	}
	if len(mm.leave) != len(o.leave) {
		return false
	}
	for idx, t := range mm.tiles {
		if t != o.tiles[idx] {
			return false
		}
	}
	for idx, t := range mm.leave {
		if t != o.leave[idx] {
			return false
		}
	}
	return true
}

// NewScoringMinimalMove creates a minimal move with a score.
func NewScoringMinimalMove(score int, tiles, leave tilemapping.MachineWord,
	vertical bool, rowStart, colStart int) *MinimalMove {

	vbit := uint32(0)
	if vertical {
		vbit = 1
	}
	tp := uint32(0)
	for _, t := range tiles {
		if t != 0 {
			tp++
		}
	}
	key := uint32(colStart) + (uint32(rowStart) << mmRowShift) +
		(vbit << mmVerticalShift) + (tp << mmTilesPlayedShift) +
		(uint32(score) << mmScoreShift)

	return &MinimalMove{
		tiles: tiles,
		leave: leave,
		key:   key,
	}
}

func NewPassMinimalMove(leave tilemapping.MachineWord) *MinimalMove {
	return &MinimalMove{
		key:   1 << mmTypeShift,
		leave: leave,
	}
}

func NewExchangeMinimalMove(toExchange, leave tilemapping.MachineWord) *MinimalMove {
	return &MinimalMove{
		tiles: toExchange,
		leave: leave,
		key:   2<<mmTypeShift + (uint32(len(toExchange)) << mmTilesPlayedShift), // 2 = exchange
	}
}
