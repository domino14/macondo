package move

import (
	"fmt"
	"strconv"

	"github.com/domino14/macondo/alphabet"
)

// MoveType is a type of move; a play, an exchange, pass, etc.
type MoveType uint8

const (
	MoveTypePlay MoveType = iota
	MoveTypeExchange
	MoveTypePass
	MoveTypePhonyTilesReturned

	MoveTypeEndgameTiles
	MoveTypeLostTileScore
)

// Move is a move. It can have a score, position, equity, etc. It doesn't
// have to be a scoring move.
type Move struct {
	action      MoveType
	score       int
	equity      int
	desc        string
	coords      string
	tiles       alphabet.MachineWord
	leave       alphabet.MachineWord
	rowStart    int
	colStart    int
	vertical    bool
	bingo       bool
	tilesPlayed int
	alph        *alphabet.Alphabet
}

func (m Move) String() string {
	switch m.action {
	case MoveTypePlay:
		return fmt.Sprintf(
			"<action: play word: %v %v (%v) score: %v tp: %v leave: %v>",
			m.coords, m.tiles, m.tiles.UserVisible(m.alph), m.score,
			m.tilesPlayed, m.leave.UserVisible(m.alph))
	case MoveTypePass:
		return fmt.Sprint("<action: pass>")
	}
	return fmt.Sprint("<Unhandled move>")

}

func (m Move) Action() MoveType {
	return m.action
}

func (m Move) TilesPlayed() int {
	return m.tilesPlayed
}

func NewScoringMove(score int, tiles alphabet.MachineWord,
	leave alphabet.MachineWord, vertical bool, tilesPlayed int,
	alph *alphabet.Alphabet, rowStart int, colStart int, coords string) *Move {

	move := &Move{
		action: MoveTypePlay, score: score, tiles: tiles, leave: leave, vertical: vertical,
		bingo: tilesPlayed == 7, tilesPlayed: tilesPlayed, alph: alph,
		rowStart: rowStart, colStart: colStart, coords: coords,
	}
	return move
}

func (m Move) Score() int {
	return m.score
}

func (m Move) Leave() alphabet.MachineWord {
	return m.leave
}

func (m Move) Tiles() alphabet.MachineWord {
	return m.tiles
}

func (m Move) UniqueSingleTileKey() int {
	// Find the tile.
	var idx int
	var ml alphabet.MachineLetter
	for idx, ml = range m.tiles {
		if ml != alphabet.PlayedThroughMarker {
			break
		}
	}

	var row, col int
	row = m.rowStart
	col = m.colStart
	// We want to get the coordinate of the tile that is on the board itself.
	if m.vertical {
		row, col = col, row
		row += idx
	} else {
		col += idx
	}
	// A unique, fast to compute key for this play.
	return row + alphabet.MaxAlphabetSize*col +
		alphabet.MaxAlphabetSize*alphabet.MaxAlphabetSize*int(ml)
}

func (m *Move) CoordsAndVertical() (int, int, bool) {
	return m.rowStart, m.colStart, m.vertical
}

func (m *Move) BoardCoords() string {
	return m.coords
}

func ToBoardGameCoords(row int, col int, vertical bool) string {
	colCoords := string(rune('A' + col))
	rowCoords := strconv.Itoa(int(row + 1))
	var coords string
	if vertical {
		coords = colCoords + rowCoords
	} else {
		coords = rowCoords + colCoords
	}
	return coords
}

func NewPassMove() *Move {
	return &Move{
		action: MoveTypePass,
	}
}
