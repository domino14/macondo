// Package xwordgame contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package xwordgame

import (
	"math/rand"
	"time"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/lexicon"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// XWordGame encapsulates the various components of a crossword game.
type XWordGame struct {
	// The movegen has the tilebag in it. Maybe eventually we will move it.
	movegen *movegen.GordonGenerator
	players []Player
	board   board.GameBoard
	bag     lexicon.Bag
	gaddag  gaddag.SimpleGaddag
}

// Init initializes the crossword game and seeds the random number generator.
func (game *XWordGame) Init(gd gaddag.SimpleGaddag) {

	game.board = board.MakeBoard(board.CrosswordGameBoard)
	dist := lexicon.EnglishLetterDistribution()
	game.bag = dist.MakeBag(gd.GetAlphabet())
	game.gaddag = gd
	game.movegen = movegen.NewGordonGenerator(gd, game.bag, game.board)

	rand.Seed(time.Now().UTC().UnixNano())
}

// StartGame determines a starting player and deals out the first set of tiles.
func (game *XWordGame) StartGame() {
	game.players = []Player{
		{"", "player1"},
		{"", "player2"},
	}
	game.board.UpdateAllAnchors()
	for i := 0; i < 2; i++ {
		rack, _ := game.bag.Draw(7)
		game.players[i].rack = string(rack)
	}
}

// PlayBestStaticTurn generates the best static move for the player and
// plays it on the board.
func (game *XWordGame) PlayBestStaticTurn(playerID int) {
	game.movegen.GenAll(game.players[playerID].rack)
	game.PlayMove(game.movegen.Plays()[0])
}

func (game *XWordGame) PlayMove(m *move.Move) {
	switch m.Action() {
	case move.MoveTypePlay:
		game.board.PlayMove(m)
	case move.MoveTypePass:
		// something here.

	}
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
