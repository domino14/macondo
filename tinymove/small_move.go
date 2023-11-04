package tinymove

import (
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
)

// A SmallMove consists of a TinyMove which encodes all the positional
// and tile info, plus a few extra fields needed for endgames and related
// modules. We're trying to strike a balance between decreasing allocations
// and still staying speedy.
type SmallMove struct {
	tm             TinyMove
	score          int16
	estimatedValue int16
	tilesPlayed    uint8
}

func PassMove() SmallMove {
	// everything is 0.
	return SmallMove{}
}

func TilePlayMove(tm TinyMove, score int16, tilesPlayed uint8) SmallMove {
	return SmallMove{tm: tm, score: score, tilesPlayed: tilesPlayed}
}

// EstimatedValue is an internal value that is used in calculating endgames and related metrics.
func (m *SmallMove) EstimatedValue() int16 {
	return m.estimatedValue
}

func (m *SmallMove) ShortDescription(tm *tilemapping.TileMapping) string {
	// depends on the board.
	return "(n/a)"
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

func (m *SmallMove) TilesPlayed() uint8 {
	return m.tilesPlayed
}

func (m *SmallMove) Score() int16 {
	return m.score
}

func (m *SmallMove) TinyMove() TinyMove {
	return m.tm
}

func (m *SmallMove) IsPass() bool {
	return m.tm == 0
}

func SmallMoveToMove(sm *SmallMove, m *move.Move, tm *tilemapping.TileMapping,
	bd *board.GameBoard, onTurnRack *tilemapping.Rack) {
	TinyMoveToMove(sm.tm, bd, m)
	// populate move with missing fields.
	m.SetAlphabet(tm)
	leave, err := tilemapping.Leave(onTurnRack.TilesOn(), m.Tiles(), false)
	if err != nil {
		panic(err)
	}
	m.SetLeave(leave)
	m.SetScore(int(sm.score))
}
