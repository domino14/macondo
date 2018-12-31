// Package xwordgame contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package xwordgame

import (
	"log"
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
	movegen        *movegen.GordonGenerator
	players        []Player
	onturn         int // player index
	turn           int
	board          board.GameBoard
	bag            lexicon.Bag
	gaddag         gaddag.SimpleGaddag
	playing        bool
	scorelessTurns int
	randomizer     *rand.Rand
}

// Init initializes the crossword game and seeds the random number generator.
func (game *XWordGame) Init(gd gaddag.SimpleGaddag) {

	seed := time.Now().UTC().UnixNano()
	game.randomizer = rand.New(rand.NewSource(seed))

	game.board = board.MakeBoard(board.CrosswordGameBoard)
	dist := lexicon.EnglishLetterDistribution()
	game.bag = dist.MakeBag(gd.GetAlphabet(), false)
	game.bag.SetRandomizer(game.randomizer)
	// The randomizer seems to have no effect on this shuffle (it's not
	// deterministic). Oh well.
	// XXX: Rewrite shuffle to be deterministic based on seed for testing.
	game.bag.Shuffle()
	game.gaddag = gd
	game.movegen = movegen.NewGordonGenerator(gd, game.bag, game.board)
}

// StartGame determines a starting player and deals out the first set of tiles.
func (game *XWordGame) StartGame() {
	game.players = []Player{
		{"", "player1", 0},
		{"", "player2", 0},
	}
	game.board.UpdateAllAnchors()
	for i := 0; i < 2; i++ {
		rack, _ := game.bag.Draw(7)
		game.players[i].rack = string(rack)
	}
	game.onturn = 0
	game.turn = 0
	game.playing = true
}

// PlayBestStaticTurn generates the best static move for the player and
// plays it on the board.
func (game *XWordGame) PlayBestStaticTurn(playerID int) {
	log.Printf("[DEBUG] %v: Playing best static turn for player %v", game.turn,
		playerID)
	game.movegen.GenAll(game.players[playerID].rack)
	log.Printf("[DEBUG] Generated %v moves, best static is %v",
		len(game.movegen.Plays()), game.movegen.Plays()[0])
	game.PlayMove(game.movegen.Plays()[0])
	game.turn++
}

// PlayMove plays a move.
func (game *XWordGame) PlayMove(m *move.Move) {
	switch m.Action() {
	case move.MoveTypePlay:
		game.board.PlayMove(m, game.gaddag, game.bag)
		game.players[game.onturn].points += m.Score()
		log.Printf("[DEBUG] Player %v played %v", game.onturn, m)
		// Draw new tiles.
		rack := game.bag.DrawAtMost(m.TilesPlayed())
		game.players[game.onturn].rack = string(rack) + m.Leave().UserVisible(
			game.gaddag.GetAlphabet())
		log.Printf("[DEBUG] Player %v drew new tiles: %v, rack is now %v",
			game.onturn, string(rack), game.players[game.onturn].rack)
		game.scorelessTurns = 0
	case move.MoveTypePass:
		log.Printf("[DEBUG] Player %v passed", game.onturn)
		game.scorelessTurns++
	}
	if game.scorelessTurns == 6 {
		log.Printf("[DEBUG] Game ended after 6 scoreless turns")
		game.playing = false
	}
	game.onturn = (game.onturn + 1) % len(game.players)
}
