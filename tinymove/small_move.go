package tinymove

import (
	"fmt"
)

// A SmallMove consists of a TinyMove which encodes all the positional
// and tile info, plus a few extra fields needed for endgames and related
// modules. We're trying to strike a balance between decreasing allocations
// and still staying speedy.
type SmallMove struct {
	tm             TinyMove
	score          int16
	estimatedValue int16
	// tilesDescriptor:
	// CCCC CPPP
	// 7    3
	// PPP = a 3-bit number (number of tiles in play that came from rack)
	// CCCCC = a 5-bit number (total number of tiles in play, including play-through)
	tilesDescriptor uint8
}

// DefaultSmallMove is a blank move (a pass)
var DefaultSmallMove = SmallMove{}

const tilesPlayedBitMask = 0b00000111

func PassMove() SmallMove {
	// everything is 0.
	return SmallMove{}
}

func TilePlayMove(tm TinyMove, score int16, tilesPlayed, playLength uint8) SmallMove {
	tilesDescriptor := tilesPlayed + (playLength << 3)

	return SmallMove{tm: tm, score: score, tilesDescriptor: tilesDescriptor}
}

// EstimatedValue is an internal value that is used in calculating endgames and related metrics.
func (m *SmallMove) EstimatedValue() int16 {
	return m.estimatedValue
}

func (m *SmallMove) ShortDescription() string {
	// depends on the board.
	return fmt.Sprintf("<tinyplay: %d score: %d nracktiles: %d nplaytiles: %d>",
		m.tm,
		m.score, m.tilesDescriptor&tilesPlayedBitMask, m.tilesDescriptor>>3)
}

// SetEstimatedValue sets the estimated value of this move. It is calculated
// outside of this package.
func (m *SmallMove) SetEstimatedValue(v int16) {
	m.estimatedValue = v
}

// AddEstimatedValue adds an estimate to the existing estimated value of this
// estimate. Estimate.
func (m *SmallMove) AddEstimatedValue(v int16) {
	m.estimatedValue += v
}

func (m *SmallMove) TilesPlayed() int {
	return int(m.tilesDescriptor) & tilesPlayedBitMask
}

func (m *SmallMove) PlayLength() int {
	return int(m.tilesDescriptor) >> 3
}

func (m *SmallMove) Score() int {
	return int(m.score)
}

func (m *SmallMove) TinyMove() TinyMove {
	return m.tm
}

func (m *SmallMove) IsPass() bool {
	return m.tm == 0
}

func (m *SmallMove) CoordsAndVertical() (int, int, bool) {
	// assume it's a tile play move
	t := m.tm
	row := int(t&RowBitMask) >> 6
	col := int(t&ColBitMask) >> 1
	vert := false
	if t&1 > 0 {
		vert = true
	}
	return row, col, vert
}
