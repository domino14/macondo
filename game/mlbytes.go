package game

import (
	"fmt"
	"slices"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/move"
)

// Plane offsets in the NN_N_PLANES board block (see BuildMLVector).
const (
	mlPlaneSize      = NN_H * NN_W
	mlBlankPlane     = 26 * mlPlaneSize
	mlHorCCPlanes    = 27 * mlPlaneSize
	mlVerCCPlanes    = 53 * mlPlaneSize
	mlBonusPlanes    = 79 * mlPlaneSize // 2L, 3L, 2W, 3W
	mlHistoryPlanes  = 83 * mlPlaneSize // 0: the candidate, 1: opponent's last move
	mlNumBonusPlanes = 4
)

// mlByteBoard holds one position's board planes as bytes before any
// candidate is played, with the cross-sets they were built from. Every
// plane is 0/1, so a byte per value carries the same features as
// BuildMLVector's floats; each candidate then copies these planes and
// rewrites only the squares its play changed.
type mlByteBoard struct {
	planes [NN_N_PLANES]byte
	hc, vc [mlPlaneSize]board.CrossSet
}

// MLByteVectorsForMoves is MLVectorsForMoves with the board planes as
// bytes (every plane is 0/1): the same features, a quarter of the size.
func (g *Game) MLByteVectorsForMoves(moves []*move.Move, leaveCalc *equity.ExhaustiveLeaveCalculator,
	lastMoves []*move.Move) (boards []byte, scalars []float32, err error) {
	return g.mlByteVectorsInto(nil, nil, moves, leaveCalc, lastMoves)
}

// mlByteVectorsInto is mlVectorsInto with the board planes as bytes,
// appended to boards (one NN_N_PLANES block per move); the scalars are
// the same floats.
func (g *Game) mlByteVectorsInto(boards []byte, scalars []float32, moves []*move.Move,
	leaveCalc *equity.ExhaustiveLeaveCalculator, lastMoves []*move.Move) ([]byte, []float32, error) {

	if g.mlBytes == nil {
		g.mlBytes = new(mlByteBoard)
	}
	base := g.mlBytes
	if err := g.fillMLByteBase(base, lastMoves); err != nil {
		return nil, nil, err
	}
	boards = slices.Grow(boards[:0], len(moves)*NN_N_PLANES)
	scalars = slices.Grow(scalars[:0], len(moves)*NN_N_SCAL)
	err := g.forEachMLMove(moves, func(m *move.Move) error {
		boards = append(boards, base.planes[:]...)
		if err := g.patchMLBytes(boards[len(boards)-NN_N_PLANES:], base, m); err != nil {
			return err
		}
		n := len(scalars)
		scalars = append(scalars, make([]float32, NN_N_SCAL)...)
		g.mlScalarsInto(scalars[n:], m, leaveCalc.LeaveValue(m.Leave()), lastMoves)
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return boards, scalars, nil
}

// fillMLByteBase builds the planes of the current board, before any
// candidate: tiles, blanks, cross-sets, bonuses of empty squares, and the
// opponent's last move in history plane 1.
func (g *Game) fillMLByteBase(base *mlByteBoard, lastMoves []*move.Move) error {
	p := &base.planes
	clear(p[:])
	b := g.Board()
	for r := range NN_H {
		for c := range NN_W {
			idx := r*NN_W + c
			if tile := b.GetLetter(r, c); tile != 0 {
				if err := mlSetTile(p[:], idx, tile, r, c); err != nil {
					return err
				}
			} else {
				mlSetBonus(p[:], idx, b.GetBonus(r, c))
			}
			hc := b.GetCrossSet(r, c, board.HorizontalDirection)
			vc := b.GetCrossSet(r, c, board.VerticalDirection)
			base.hc[idx], base.vc[idx] = hc, vc
			mlSetCrossSet(p[mlHorCCPlanes:], idx, hc)
			mlSetCrossSet(p[mlVerCCPlanes:], idx, vc)
		}
	}
	if len(lastMoves) > 0 {
		return mlMarkMove(p[mlHistoryPlanes+mlPlaneSize:], lastMoves[len(lastMoves)-1])
	}
	return nil
}

// patchMLBytes turns a copy of the base planes into the planes after m,
// which has been played on the board: its tiles (no bonus under them),
// every cross-set that changed, and m in history plane 0.
func (g *Game) patchMLBytes(p []byte, base *mlByteBoard, m *move.Move) error {
	if m.Action() != move.MoveTypePlay {
		return nil // the board is unchanged
	}
	b := g.Board()
	r0, c0, vertical := m.CoordsAndVertical()
	dr, dc := 0, 1
	if vertical {
		dr, dc = 1, 0
	}
	for i, t := range m.Tiles() {
		if t == 0 {
			continue // played through
		}
		r, c := r0+i*dr, c0+i*dc
		idx := r*NN_W + c
		if err := mlSetTile(p, idx, b.GetLetter(r, c), r, c); err != nil {
			return err
		}
		for k := range mlNumBonusPlanes {
			p[mlBonusPlanes+k*mlPlaneSize+idx] = 0
		}
	}
	for r := range NN_H {
		for c := range NN_W {
			idx := r*NN_W + c
			if hc := b.GetCrossSet(r, c, board.HorizontalDirection); hc != base.hc[idx] {
				mlSetCrossSet(p[mlHorCCPlanes:], idx, hc)
			}
			if vc := b.GetCrossSet(r, c, board.VerticalDirection); vc != base.vc[idx] {
				mlSetCrossSet(p[mlVerCCPlanes:], idx, vc)
			}
		}
	}
	return mlMarkMove(p[mlHistoryPlanes:], m)
}

func mlSetTile(p []byte, idx int, tile tilemapping.MachineLetter, r, c int) error {
	unblanked := tile.Unblank()
	if unblanked != tile {
		p[mlBlankPlane+idx] = 1
	}
	ll := int(unblanked) - 1
	if ll < 0 || ll >= 26 {
		return fmt.Errorf("invalid tile index %d for tile %d at position (%d, %d)", ll, tile, r, c)
	}
	p[ll*mlPlaneSize+idx] = 1
	return nil
}

func mlSetBonus(p []byte, idx int, bonus board.BonusSquare) {
	switch bonus {
	case board.Bonus2LS:
		p[mlBonusPlanes+idx] = 1
	case board.Bonus3LS:
		p[mlBonusPlanes+mlPlaneSize+idx] = 1
	case board.Bonus2WS:
		p[mlBonusPlanes+2*mlPlaneSize+idx] = 1
	case board.Bonus3WS:
		p[mlBonusPlanes+3*mlPlaneSize+idx] = 1
	}
}

// mlSetCrossSet writes cs's letters A-Z for square idx into 26 planes.
func mlSetCrossSet(planes []byte, idx int, cs board.CrossSet) {
	for t := range 26 {
		planes[t*mlPlaneSize+idx] = byte(cs>>(t+1)) & 1
	}
}

// mlMarkMove sets the squares where m placed tiles in plane.
func mlMarkMove(plane []byte, m *move.Move) error {
	if m.Action() != move.MoveTypePlay {
		return nil
	}
	r0, c0, vertical := m.CoordsAndVertical()
	dr, dc := 0, 1
	if vertical {
		dr, dc = 1, 0
	}
	for i, t := range m.Tiles() {
		r, c := r0+i*dr, c0+i*dc
		if r < 0 || r >= NN_H || c < 0 || c >= NN_W {
			return fmt.Errorf("last move out of bounds at (%d, %d)", r, c)
		}
		if t != 0 {
			plane[r*NN_W+c] = 1
		}
	}
	return nil
}
