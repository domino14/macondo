package movegen

import (
	"github.com/domino14/macondo/cgboard"
)

// Anchors are places on the board where a word can start.
// These are very tied to move generation so we put them in this package.
// We assume that the board will not be transposed.
type Anchors struct {
	hanchors []bool
	vanchors []bool
	board    *cgboard.GameBoard
}

func MakeAnchors(board *cgboard.GameBoard) *Anchors {
	n := board.Dim() * board.Dim()
	return &Anchors{
		board:    board,
		hanchors: make([]bool, n),
		vanchors: make([]bool, n),
	}
}

func (a *Anchors) SetAnchor(row, col int, dir cgboard.BoardDirection) {
	pos := row*a.board.Dim() + col

	if dir == cgboard.HorizontalDirection {
		a.hanchors[pos] = true
		return
	}
	a.vanchors[pos] = true
}

// IsAnchor gets whether the passed-in row and column is an anchor.
// This function, unlike the other anchor functions, can be called while
// the board is transposed.
func (a *Anchors) IsAnchor(row, col int, dir cgboard.BoardDirection) bool {
	pos := row*a.board.Dim() + col
	if a.board.IsTransposed() {
		pos = col*a.board.Dim() + row
	}
	if dir == cgboard.HorizontalDirection {
		return a.hanchors[pos]
	}
	return a.vanchors[pos]
}

func (a *Anchors) resetAnchors(pos int) {
	a.hanchors[pos] = false
	a.vanchors[pos] = false
}

func (a *Anchors) updateAnchors(row int, col int, vertical bool) {
	dim := a.board.Dim()
	pos := row*dim + col
	if vertical {
		// This helps simplify the updateAnchorsForMove algorithm.
		row, col = col, row
	}
	// Always reset the anchors before applying anything else.
	a.resetAnchors(pos)
	var tileAbove, tileBelow, tileLeft, tileRight, tileHere bool
	if row > 0 {
		tileAbove = a.board.HasLetter(row-1, col)
	}
	if col > 0 {
		tileLeft = a.board.HasLetter(row, col-1)
	}
	if row < dim-1 {
		tileBelow = a.board.HasLetter(row+1, col)
	}
	if col < dim-1 {
		tileRight = a.board.HasLetter(row, col+1)
	}
	tileHere = a.board.HasLetter(row, col)
	if tileHere {
		// The current square is not empty. It should only be an anchor
		// if it is the rightmost square of a word (actually, squares to
		// the left are probably ok, but not the leftmost square. Note
		// Gordon does not have this requirement, but the algorithm does
		// not work if we don't do this)
		if !tileRight {
			a.hanchors[pos] = true
		}
		// Apply the transverse logic too for the vertical anchor.
		if !tileBelow {
			a.vanchors[pos] = true
		}
	} else {
		// If the square is empty, it should only be an anchor if the
		// squares to its left and right are empty, and at least one of
		// the squares in the top and bottom are NOT empty.
		if !tileLeft && !tileRight && (tileAbove || tileBelow) {
			a.hanchors[pos] = true
		}
		// (And apply the transverse logic for the vertical anchor)
		if !tileAbove && !tileBelow && (tileLeft || tileRight) {
			a.vanchors[pos] = true
		}
	}
}

func (a *Anchors) UpdateAllAnchors() {
	n := a.board.Dim()
	if a.board.TilesPlayed() > 0 {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				a.updateAnchors(i, j, false)
			}
		}
	} else {
		for i := 0; i < len(a.hanchors); i++ {
			a.hanchors[i] = false
			a.vanchors[i] = false
		}
		rc := int(n / 2)
		// If the board is empty, set just one anchor, in the center square.
		a.hanchors[rc*n+rc] = true
	}
}

/*
// XXX where do we put this function??
func (g *GameBoard) updateAnchorsForMove(m *move.Move) {
	row, col, vertical := m.CoordsAndVertical()

	if vertical {
		// Transpose the logic, but NOT the board. The updateAnchors function
		// assumes the board is not transposed.
		col, row = row, col
	}

	// Update anchors all around the play.
	for i := col; i < len(m.Tiles())+col; i++ {
		g.updateAnchors(row, i, vertical)
		if row > 0 {
			g.updateAnchors(row-1, i, vertical)
		}
		if row < g.Dim()-1 {
			g.updateAnchors(row+1, i, vertical)
		}
	}

	if col-1 >= 0 {
		g.updateAnchors(row, col-1, vertical)
	}
	if len(m.Tiles())+col < g.Dim() {
		g.updateAnchors(row, col+len(m.Tiles()), vertical)
	}

}
*/
