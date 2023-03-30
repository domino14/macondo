package move

import "github.com/domino14/macondo/tilemapping"

const (
	// constants used for the key for a minimal move.

	// layout
	// 32       24       16       8
	// xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
	// ssssssss ssssssss    ttdrr rrrccccc
	// s - score (2 ^ 16 max)
	// t - type
	// r - row (32 max)
	// c - col (32 max)

	mmRowShift      = 5
	mmVerticalShift = 10
	mmTypeShift     = 11
	mmScoreShift    = 16
	mmColBitmask    = (1 << 5) - 1
)

// MinimalMove should be used for our move generators.
type MinimalMove struct {
	// the tiles need to be kept separate.
	tiles tilemapping.MachineWord
	// key encodes everything else we need to know about the move
	key uint32
	// leave is not included here. If leave is important to know fast,
	// don't use a MinimalMove
}

// ShortDescription turns it into a regular move.
func (mm *MinimalMove) ShortDescription(alph *tilemapping.TileMapping) string {
	m := &Move{}
	mm.CopyToMove(m)
	m.SetAlphabet(alph)
	return m.ShortDescription()
}

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
		nil,
		int(mm.key>>mmScoreShift),
		int((mm.key>>mmRowShift)&mmColBitmask),
		int(mm.key)&mmColBitmask,
		0, // tiles played can be calculated later
		(mm.key>>mmVerticalShift)&1 == 1,
		t,
		nil, // set alphabet later
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
}

func (mm *MinimalMove) Set(word tilemapping.MachineWord, score, row, col int, vertical bool,
	t MoveType) {
	vbit := uint32(0)
	if vertical {
		vbit = 1
	}
	mm.key = uint32(col) + (uint32(row) << mmRowShift) +
		(vbit << mmVerticalShift) + (uint32(score) << mmScoreShift)
	mm.tiles = word
}

// NewScoringMinimalMove creates a minimal move with a score.
func NewScoringMinimalMove(score int, tiles tilemapping.MachineWord, vertical bool,
	rowStart, colStart int) *MinimalMove {

	vbit := uint32(0)
	if vertical {
		vbit = 1
	}
	key := uint32(colStart) + (uint32(rowStart) << mmRowShift) +
		(vbit << mmVerticalShift) + (uint32(score) << mmScoreShift)

	return &MinimalMove{
		tiles: tiles,
		key:   key,
	}
}

func NewPassMinimalMove(leave tilemapping.MachineWord) *MinimalMove {
	return &MinimalMove{
		key:   1 << mmTypeShift,
		tiles: leave, // maybe ok?
	}
}

func NewExchangeMinimalMove(toExchange tilemapping.MachineWord) *MinimalMove {
	return &MinimalMove{
		tiles: toExchange,
		key:   2 << mmTypeShift, // 2 = exchange
	}
}
