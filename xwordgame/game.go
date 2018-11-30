// Package xwordgame contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package xwordgame

import (
	"math/rand"
	"time"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/movegen"
)

// XWordGame encapsulates the various components of a crossword game.
type XWordGame struct {
	// The movegen has the tilebag in it. Maybe eventually we will move it.
	movegen *movegen.GordonGenerator
	players []Player
}

// Init initializes the crossword game and seeds the random number generator.
func (game *XWordGame) Init(gd gaddag.SimpleGaddag) {
	game.movegen = movegen.NewGordonGenerator(gd)
	game.players = []Player{
		{&movegen.Rack{}, "player1"},
		{&movegen.Rack{}, "player2"},
	}
	rand.Seed(time.Now().UTC().UnixNano())
}

// StartGame determines a starting player and deals out the first set of tiles.
func (game *XWordGame) StartGame() {
	// game.movegen.
}

// PlayBestTurn generates the best move for the player and plays it on the
// board.
func (game *XWordGame) PlayBestTurn(playerID int) {

}

/*
Suggested organization:
	Rack - can probably stay in movegen, it seems fairly integral to the
		concept of 'move generation', but also leaning towards separating
	Board - should be standalone, many things need a board
	Move - probably standalone, since there are types of moves that don't
		involve the movegen at all (pass, challenge, +5)
	CrossSet - part of the board
	Bag - Use the bag in the lexicon directory; the concept of runes is ok!
		The movegen/gaddag/alphabet packages basically translate runes into
		MachineLetters. But the bag and rack don't need to be MachineLetters.
*/
