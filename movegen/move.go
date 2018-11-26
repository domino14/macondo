package movegen

import (
	"fmt"
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
	return fmt.Sprintf("<action: %v word: %v %v bingo: %v tp: %v vert: %v>",
		m.action, m.coords, m.word.UserVisible(m.alph), m.bingo,
		m.tilesPlayed, m.vertical)
}

func (m Move) uniqueKey() string {
	// Create a unique key for this play. In most cases, this will just be
	// the row/col position and the word string.
	// However, if only one tile has been played, we should only consider
	// horizontal plays
	if m.tilesPlayed != 1 {
		// XXX: CHANGE THE USERVISIBLE AFTER DEBUGGING!!
		return fmt.Sprintf("%v%v", m.coords, m.word.UserVisible(m.alph))
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
	var row, col uint8
	row = m.rowStart
	col = m.colStart
	// We want to get the coordinate of the tile that is on the board itself.
	if m.vertical {
		row, col = col, row
		row += uint8(idx)
	} else {
		col += uint8(idx)
	}
	key := fmt.Sprintf("%v-%v-%v", row, col, playedTile)
	return key
}

func toBoardGameCoords(row uint8, col uint8, vertical bool) string {
	if vertical {
		row, col = col, row
	}
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
