package movegen

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"
)

type Direction uint8

const (
	HorizontalDirection Direction = iota
	VerticalDirection
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

func (s *Square) setCrossSet(cs CrossSet, dir Direction) {
	if dir == HorizontalDirection {
		s.hcrossSet = cs
	} else if dir == VerticalDirection {
		s.vcrossSet = cs
	}
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

func (g *GameBoard) genAllCrossSets() {
	// Generate all cross-sets. Basically go through the entire board;
	// our anchor algorithm doesn't quite match the one in the Gordon
	// paper.

	// We should do this for both transpositions of the board.

	n := g.dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if g.squares[i][j].letter != EmptySquareMarker {
				// We are setting the vertical cross-set since we're
				// horizontal.
				g.squares[i][j].setCrossSet(CrossSet(0), VerticalDirection)
			} else {
				g.genCrossSet(i, j)
			}
		}
	}
	g.transpose()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if g.squares[i][j].letter != EmptySquareMarker {
				g.squares[i][j].setCrossSet(CrossSet(0), HorizontalDirection)
			} else {
				g.genCrossSet(i, j)
			}
		}
	}
	// And transpose back to the original orientation.
	g.transpose()

}

func (g *GameBoard) genCrossSet(row int, col int) {
	// This function is always called for empty squares.
	// If there's no tile adjacent to this square in any direction,
	// every letter is allowed.

	// When we find a tile, traverse it horizontally in both directions.
	if col > 0 {
		for i := col - 1; i >= 0; i-- {

		}
	}

}
