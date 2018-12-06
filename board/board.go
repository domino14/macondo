package board

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/lexicon"
	"github.com/domino14/macondo/move"
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
	hcrossScore int
	vcrossScore int
	hAnchor     bool
	vAnchor     bool
}

func (s Square) String() string {
	return fmt.Sprintf("<(%v) (%s)>", s.letter, string(s.bonus))
}

func (s Square) Letter() alphabet.MachineLetter {
	return s.letter
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
	if s.letter == alphabet.EmptySquareMarker {
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

func (s *Square) setCrossScore(score int, dir BoardDirection) {
	if dir == HorizontalDirection {
		s.hcrossScore = score
	} else if dir == VerticalDirection {
		s.vcrossScore = score
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

func (s *Square) getCrossScore(dir BoardDirection) int {
	if dir == HorizontalDirection {
		return s.hcrossScore
	} else if dir == VerticalDirection {
		return s.vcrossScore
	}
	return 0
}

func (s *Square) IsEmpty() bool {
	return s.letter == alphabet.EmptySquareMarker
}

func (s *Square) anchor(dir BoardDirection) bool {
	if dir == HorizontalDirection {
		return s.hAnchor
	}
	return s.vAnchor
}

// A GameBoard is the main board structure. It contains all of the Squares,
// with bonuses or filled letters, as well as cross-sets and cross-scores
// for computation. (See Appel & Jacobson paper for definition of the latter
// two terms)
type GameBoard struct {
	squares    [][]*Square
	transposed bool
	hasTiles   bool // has at least one tile been played?
}

const (
	// Bonus3WS is a triple word score
	Bonus3WS BonusSquare = '='
	// Bonus3LS is a triple letter score
	Bonus3LS BonusSquare = '"'
	// Bonus2LS is a double letter score
	Bonus2LS BonusSquare = '\''
	// Bonus2WS is a double word score
	Bonus2WS BonusSquare = '-'
)

// MakeBoard creates a board from a description string.
func MakeBoard(desc []string) GameBoard {
	// Turns an array of strings into the GameBoard structure type.
	rows := [][]*Square{}
	for _, s := range desc {
		row := []*Square{}
		for _, c := range s {
			row = append(row, &Square{letter: alphabet.EmptySquareMarker, bonus: BonusSquare(c)})
		}
		rows = append(rows, row)
	}
	return GameBoard{squares: rows}
}

// Dim is the dimension of the board. It assumes the board is square.
func (g *GameBoard) Dim() int {
	return len(g.squares)
}

func (g *GameBoard) GetBonus(row int, col int) BonusSquare {
	return g.squares[row][col].bonus
}

func (g *GameBoard) GetSquare(row int, col int) *Square {
	return g.squares[row][col]
}

func (g *GameBoard) SetLetter(row int, col int, letter alphabet.MachineLetter) {
	g.squares[row][col].letter = letter
}

func (g *GameBoard) GetLetter(row int, col int) alphabet.MachineLetter {
	return g.GetSquare(row, col).letter
}

func (g *GameBoard) GetCrossSet(row int, col int, dir BoardDirection) CrossSet {
	return *g.squares[row][col].getCrossSet(dir) // the actual value
}

func (g *GameBoard) ClearCrossSet(row int, col int, dir BoardDirection) {
	g.squares[row][col].getCrossSet(dir).clear()
}

func (g *GameBoard) SetCrossSetLetter(row int, col int, dir BoardDirection,
	ml alphabet.MachineLetter) {
	g.squares[row][col].getCrossSet(dir).set(ml)
}

func (g *GameBoard) GetCrossScore(row int, col int, dir BoardDirection) int {
	return g.squares[row][col].getCrossScore(dir)
}

// Transpose transposes the board, swapping rows and columns.
func (g *GameBoard) Transpose() {
	n := g.Dim()
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			g.squares[i][j], g.squares[j][i] = g.squares[j][i], g.squares[i][j]
		}
	}
}

// SetAllCrosses sets the cross sets of every square to every acceptable letter.
func (g *GameBoard) SetAllCrosses() {
	// Assume square board. This should be an assertion somewhere.
	n := g.Dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.squares[i][j].hcrossSet.setAll()
			g.squares[i][j].vcrossSet.setAll()
		}
	}
}

// ClearAllCrosses disallows all letters on all squares (more or less).
func (g *GameBoard) ClearAllCrosses() {
	n := g.Dim()
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
		tileAbove = !g.squares[row-1][col].IsEmpty()
	}
	if col > 0 {
		tileLeft = !g.squares[row][col-1].IsEmpty()
	}
	if row < g.Dim()-1 {
		tileBelow = !g.squares[row+1][col].IsEmpty()
	}
	if col < g.Dim()-1 {
		tileRight = !g.squares[row][col+1].IsEmpty()
	}
	tileHere = !g.squares[row][col].IsEmpty()
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

func (g *GameBoard) UpdateAllAnchors() {
	n := g.Dim()
	if g.hasTiles {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				g.updateAnchors(i, j)
			}
		}
	} else {
		rc := int(n / 2)
		// If the board is empty, set just one anchor, in the center square.
		g.squares[rc][rc].hAnchor = true
	}
}

// IsAnchor returns whether the row/col pair is an anchor in the given
// direction.
func (g *GameBoard) IsAnchor(row int8, col int8, dir BoardDirection) bool {
	return g.squares[row][col].anchor(dir)
}

func (g *GameBoard) posExists(row int, col int) bool {
	d := g.Dim()
	return row >= 0 && row < d && col >= 0 && col < d
}

// leftAndRightEmpty returns true if the squares at col - 1 and col + 1
// on this row are empty, checking carefully for boundary conditions.
func (g *GameBoard) leftAndRightEmpty(row int, col int) bool {
	if g.posExists(row, col-1) {
		if !g.squares[row][col-1].IsEmpty() {
			return false
		}
	}
	if g.posExists(row, col+1) {
		if !g.squares[row][col+1].IsEmpty() {
			return false
		}
	}
	return true
}

// wordEdge finds the edge of a word on the board, returning the column.
func (g *GameBoard) wordEdge(row int, col int, dir WordDirection) int {
	for g.posExists(row, col) && !g.squares[row][col].IsEmpty() {
		col += int(dir)
	}
	return col - int(dir)
}

func (g *GameBoard) traverseBackwardsForScore(row int, col int, bag lexicon.Bag) int {
	score := 0
	for g.posExists(row, col) {
		ml := g.squares[row][col].letter
		if ml == alphabet.EmptySquareMarker {
			break
		}
		score += bag.Score(ml)
		col--
	}
	return score
}

func (g *GameBoard) traverseBackwards(row int, col int, nodeIdx uint32,
	checkLetterSet bool, leftMostCol int,
	gaddag gaddag.SimpleGaddag) (uint32, bool) {
	// Traverse the letters on the board backwards (left). Return the index
	// of the node in the gaddag for the left-most letter, and a boolean
	// indicating if the gaddag path was valid.
	// If checkLetterSet is true, then we traverse until leftMostCol+1 and
	// check the letter set of this node to see if it includes the letter
	// at leftMostCol
	for g.posExists(row, col) {
		ml := g.squares[row][col].letter
		if ml == alphabet.EmptySquareMarker {
			break
		}

		if checkLetterSet && col == leftMostCol {
			if gaddag.InLetterSet(ml, nodeIdx) {
				return nodeIdx, true
			}
			// Give up early; if we're checking letter sets we only care about
			// this column.
			return nodeIdx, false
		}

		nodeIdx = gaddag.NextNodeIdx(nodeIdx, ml.Unblank())
		if nodeIdx == 0 {
			// There is no path in the gaddag for this word part; this
			// can occur if a phony was played and stayed on the board
			// and the phony has no extensions for example, or if it's
			// a real word with no further extensions.
			return nodeIdx, false
		}

		col--
	}

	return nodeIdx, true
}

// PlayMove plays a move on a board.
func (g *GameBoard) PlayMove(move *move.Move) {
	// Place tiles on the board, and regenerate cross-sets and cross-points.
	// recalculate anchors tooo
	rowStart, colStart, vertical := move.CoordsAndVertical()
	var row, col uint8
	for idx, tile := range move.Tiles() {
		if tile == alphabet.PlayedThroughMarker {
			continue
		}
		if vertical {
			row = rowStart + uint8(idx)
		} else {
			col = colStart + uint8(idx)
		}
		g.squares[row][col].letter = tile
	}
}
