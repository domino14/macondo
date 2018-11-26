package movegen

import (
	"fmt"
	"log"
	"strconv"

	"github.com/domino14/macondo/alphabet"
)

type Move struct {
	action      MoveType
	score       int
	desc        string
	coords      string
	word        alphabet.MachineWord
	rowStart    uint8
	colStart    uint8
	vertical    bool
	bingo       bool
	tilesPlayed uint8
	alph        *alphabet.Alphabet
}

func (m Move) String() string {
	return fmt.Sprintf("<action: %v row: %v col: %v word: %v bingo: %v tp: %v vert: %v>",
		m.action, m.rowStart, m.colStart, m.word.UserVisible(m.alph), m.bingo,
		m.tilesPlayed, m.vertical)
}

func (m Move) uniqueKey() string {
	// Create a unique key for this play. In most cases, this will just be
	// the row/col position and the word string.
	// However, if only one tile has been played, we should only consider
	// horizontal plays
	if m.tilesPlayed != 1 {
		return fmt.Sprintf("%v%v", m.coords, m.word)
	}
	// Find the tile.
	var playedTile rune
	var idx int
	var c rune
	for idx, c = range m.word {
		if c != alphabet.PlayedThroughMarker {
			playedTile = c
			break
		}
	}
	log.Printf("[DEBUG] Played tile was %v", playedTile)
	var row, col uint8
	row = m.rowStart
	col = m.colStart
	// We want to get the coordinate of the tile that is on the board itself.
	if m.vertical {
		col += uint8(idx)
	} else {
		row += uint8(idx)
	}
	key := fmt.Sprintf("%v%v", toBoardGameCoords(row, col, m.vertical), playedTile)
	log.Printf("[DEBUG] Calculated key %v", key)
	return key
}

func toBoardGameCoords(row uint8, col uint8, vertical bool) string {
	colCoords := rune('A' + col)
	rowCoords := strconv.Itoa(int(row + 1))
	var coords string
	if vertical {
		col, row = row, col
		coords = string(colCoords) + rowCoords
	} else {
		coords = rowCoords + string(colCoords)
	}
	return coords
}
