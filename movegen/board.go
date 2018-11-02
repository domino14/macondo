package movegen

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"
)

const (
	// TrivialCrossSet allows every possible letter. It is the default
	// state of a square.
	TrivialCrossSet = (1 << alphabet.MaxAlphabetSize) - 1
)

// A BonusSquare is a bonus square (duh)
type BonusSquare rune

// A Square is a single square in a game board. It contains the bonus markings,
// if any, a letter, if any (' ' if empty), and any cross-sets and cross-scores
type Square struct {
	letter alphabet.MachineLetter
	bonus  BonusSquare

	// hcrossSet from the Appel&Jacobson paper; simply a bit mask of letters
	// that are allowed on this square.
	hcrossSet uint64
	vcrossSet uint64
	// the scores of the tiles on either side of this square.
	hcrossScore uint32
	vcrossScore uint32
}

func (s Square) String() string {
	return fmt.Sprintf("<(%v) (%s)>", s.letter, string(s.bonus))
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

func (g *GameBoard) transpose() {
	n := len(g.squares)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			g.squares[i][j], g.squares[j][i] = g.squares[j][i], g.squares[i][j]
		}
	}
}

func (g *GameBoard) setAllCrosses() {
	// Assume square board. This should be an assertion somewhere.
	n := len(g.squares)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.squares[i][j].hcrossSet = TrivialCrossSet
			g.squares[i][j].vcrossSet = TrivialCrossSet
		}
	}
}

func (g *GameBoard) clearAllCrosses() {
	n := len(g.squares)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.squares[i][j].hcrossSet = 0
			g.squares[i][j].vcrossSet = 0
		}
	}
}
