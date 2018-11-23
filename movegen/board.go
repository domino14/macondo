package movegen

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"
)

type BoardDirection uint8
type WordDirection int

const (
	HorizontalDirection BoardDirection = iota
	VerticalDirection
)

const (
	LeftDirection  WordDirection = -1
	RightDirection WordDirection = 1
)

// A BonusSquare is a bonus square (duh)
type BonusSquare rune

// A Square is a single square in a game board. It contains the bonus markings,
// if any, a letter, if any (' ' if empty), and any cross-sets and cross-scores
type Square struct {
	letter alphabet.MachineLetter
	bonus  BonusSquare

	hcrossSet CrossSet
	vcrossSet CrossSet
	// the scores of the tiles on either side of this square.
	hcrossScore uint32
	vcrossScore uint32
	hAnchor     bool
	vAnchor     bool
}

func (s Square) String() string {
	return fmt.Sprintf("<(%v) (%s)>", s.letter, string(s.bonus))
}

func (s Square) DisplayString(alph *alphabet.Alphabet) string {
	var hadisp, vadisp, bonusdisp string
	if s.hAnchor {
		hadisp = "→"
	} else {
		hadisp = " "
	}
	if s.vAnchor {
		vadisp = "↓"
	} else {
		vadisp = " "
	}
	if s.bonus != 0 {
		bonusdisp = string(s.bonus)
	} else {
		bonusdisp = " "
	}
	if s.letter == EmptySquareMarker {
		return fmt.Sprintf("[%v%v%v]", bonusdisp, hadisp, vadisp)
	}
	return fmt.Sprintf("[%v%v%v]", s.letter.UserVisible(alph), hadisp, vadisp)

}

func (s *Square) setCrossSet(cs CrossSet, dir BoardDirection) {
	if dir == HorizontalDirection {
		s.hcrossSet = cs
	} else if dir == VerticalDirection {
		s.vcrossSet = cs
	}
}

func (s *Square) getCrossSet(dir BoardDirection) *CrossSet {
	if dir == HorizontalDirection {
		return &s.hcrossSet
	} else if dir == VerticalDirection {
		return &s.vcrossSet
	}
	return nil
}

func (s *Square) isEmpty() bool {
	return s.letter == EmptySquareMarker
}

// A GameBoard is the main board structure. It contains all of the Squares,
// with bonuses or filled letters, as well as cross-sets and cross-scores
// for computation. (See Appel & Jacobson paper for definition of the latter
// two terms)
type GameBoard struct {
	squares    [][]*Square
	transposed bool
}

const (
	// Bonus3WS Come on man I'm not going to put a comment for each of these
	Bonus3WS BonusSquare = '='
	Bonus3LS BonusSquare = '"'
	Bonus2LS BonusSquare = '\''
	Bonus2WS BonusSquare = '-'
)

func strToBoard(desc []string) GameBoard {
	// Turns an array of strings into the GameBoard structure type.
	rows := [][]*Square{}
	for _, s := range desc {
		row := []*Square{}
		for _, c := range s {
			row = append(row, &Square{letter: EmptySquareMarker, bonus: BonusSquare(c)})
		}
		rows = append(rows, row)
	}
	return GameBoard{squares: rows}
}

// All of these functions assume the board is square.
func (g *GameBoard) dim() int {
	return len(g.squares)
}

func (g *GameBoard) getCrossSet(row int, col int, dir BoardDirection) CrossSet {
	return *g.squares[row][col].getCrossSet(dir) // the actual value
}

func (g *GameBoard) transpose() {
	n := g.dim()
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			g.squares[i][j], g.squares[j][i] = g.squares[j][i], g.squares[i][j]
		}
	}
}

func (g *GameBoard) setAllCrosses() {
	// Assume square board. This should be an assertion somewhere.
	n := g.dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.squares[i][j].hcrossSet.setAll()
			g.squares[i][j].vcrossSet.setAll()
		}
	}
}

func (g *GameBoard) clearAllCrosses() {
	n := g.dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.squares[i][j].hcrossSet.clear()
			g.squares[i][j].vcrossSet.clear()
		}
	}
}
func (g *GameBoard) updateAnchors(row int, col int) {
	var tileAbove, tileBelow, tileLeft, tileRight, tileHere bool
	if row > 0 {
		tileAbove = g.squares[row-1][col].letter != EmptySquareMarker
	}
	if col > 0 {
		tileLeft = g.squares[row][col-1].letter != EmptySquareMarker
	}
	if row < g.dim()-1 {
		tileBelow = g.squares[row+1][col].letter != EmptySquareMarker
	}
	if col < g.dim()-1 {
		tileRight = g.squares[row][col+1].letter != EmptySquareMarker
	}
	tileHere = g.squares[row][col].letter != EmptySquareMarker
	if tileHere {
		// The current square is not empty. It should only be an anchor
		// if it is the rightmost square of a word (actually, squares to
		// the left are probably ok, but not the leftmost square. Note
		// Gordon does not have this requirement, but the algorithm does
		// not work if we don't do this)
		if !tileRight {
			g.squares[row][col].hAnchor = true
		}
		// Apply the transverse logic too for the vertical anchor.
		if !tileBelow {
			g.squares[row][col].vAnchor = true
		}
	} else {
		// If the square is empty, it should only be an anchor if the
		// squares to its left and right are empty, and one of the squares
		// in the top and bottom are NOT empty.
		if !tileLeft && !tileRight && (tileAbove || tileBelow) {
			g.squares[row][col].hAnchor = true
		}
		// (And apply the transverse logic for the vertical anchor)
		if !tileAbove && !tileBelow && (tileLeft || tileRight) {
			g.squares[row][col].vAnchor = true
		}
	}
}

func (g *GameBoard) updateAllAnchors() {
	n := g.dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.updateAnchors(i, j)
		}
	}
}

func (g *GameBoard) posExists(row int, col int) bool {
	d := g.dim()
	return row >= 0 && row < d && col >= 0 && col < d
}

// leftAndRightEmpty returns true if the squares at col - 1 and col + 1
// on this row are empty, checking carefully for boundary conditions.
func (g *GameBoard) leftAndRightEmpty(row int, col int) bool {
	if g.posExists(row, col-1) {
		if !g.squares[row][col-1].isEmpty() {
			return false
		}
	}
	if g.posExists(row, col+1) {
		if !g.squares[row][col+1].isEmpty() {
			return false
		}
	}
	return true
}

//      ^
// HOUSE

// wordEdge finds the edge of a word on the board, returning the column.
func (g *GameBoard) wordEdge(row int, col int, dir WordDirection) int {
	for g.posExists(row, col) && !g.squares[row][col].isEmpty() {
		col += int(dir)
	}
	return col - int(dir)
}

// func (g *GameBoard) traverseWordPart
