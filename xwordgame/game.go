// Package xwordgame contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package xwordgame

import (
	"fmt"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/endgame"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/lexicon"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// XWordGame encapsulates the various components of a crossword game.
type XWordGame struct {
	// The movegen has the tilebag in it. Maybe eventually we will move it.
	movegen            *movegen.GordonGenerator
	players            []Player
	onturn             int // player index
	turnnum            int
	board              board.GameBoard
	bag                *lexicon.Bag
	gaddag             *gaddag.SimpleGaddag
	playing            bool
	scorelessTurns     int
	numPossibleLetters int
	endgameSolver      *endgame.Solver
	logchan            chan string
	gamechan           chan string
}

// Init initializes the crossword game and seeds the random number generator.
func (game *XWordGame) Init(gd *gaddag.SimpleGaddag) {
	game.numPossibleLetters = int(gd.GetAlphabet().NumLetters())
	game.board = board.MakeBoard(board.CrosswordGameBoard)

	dist := lexicon.EnglishLetterDistribution()
	game.bag = dist.MakeBag(gd.GetAlphabet())

	game.gaddag = gd
	game.movegen = movegen.NewGordonGenerator(gd, game.bag, game.board)
	game.players = []Player{
		{nil, "", "player1", 0},
		{nil, "", "player2", 0},
	}
}

// StartGame resets everything and deals out the first set of tiles.
func (game *XWordGame) StartGame() {
	game.movegen.Reset()

	for i := 0; i < 2; i++ {
		rack, _ := game.bag.Draw(7)
		game.players[i].rackLetters = string(rack)
		game.players[i].points = 0
		if game.players[i].rack == nil {
			game.players[i].rack = movegen.RackFromString(string(rack),
				game.gaddag.GetAlphabet())
		} else {
			game.players[i].rack.Set(string(rack))
		}
	}
	game.onturn = 0
	game.turnnum = 0
	game.playing = true
}

// PlayBestStaticTurn generates the best static move for the player and
// plays it on the board.
func (game *XWordGame) PlayBestStaticTurn(playerID int) {
	// log.Printf("[DEBUG] %v: Playing best static turn for player %v (rack %v)",
	// 	game.turnnum,
	// 	playerID,
	// 	game.players[playerID].rack.TilesOn(game.numPossibleLetters).UserVisible(
	// 		game.gaddag.GetAlphabet()))
	game.movegen.GenAll(game.players[playerID].rack)
	// log.Printf("[DEBUG] Generated %v moves, best static is %v",
	// 	len(game.movegen.Plays()), game.movegen.Plays()[0])
	// log before playing move

	if game.logchan != nil {
		game.logchan <- fmt.Sprintf("%v,%v,%v\n",
			game.players[playerID].rackLetters,
			game.movegen.Plays()[0].Score(),
			game.bag.TilesRemaining())
	}
	game.PlayMove(game.movegen.Plays()[0])

	game.turnnum++
}

// PlayMove plays a move.
func (game *XWordGame) PlayMove(m *move.Move) {
	switch m.Action() {
	case move.MoveTypePlay:
		game.board.PlayMove(m, game.gaddag, game.bag)
		score := m.Score()
		if score != 0 {
			game.scorelessTurns = 0
		}
		game.players[game.onturn].points += score
		// log.Printf("[DEBUG] Player %v played %v", game.onturn, m)
		// Draw new tiles.
		drew := game.bag.DrawAtMost(m.TilesPlayed())
		rack := string(drew) + m.Leave().UserVisible(game.gaddag.GetAlphabet())
		game.players[game.onturn].rack.Set(rack)
		game.players[game.onturn].rackLetters = rack

		if game.players[game.onturn].rack.NumTiles() == 0 {
			// log.Printf("[DEBUG] Player %v played off all their tiles. Game over!",
			// 	game.onturn)
			game.playing = false
			unplayedPts := game.calculateRackPts((game.onturn+1)%len(game.players)) * 2
			// log.Printf("[DEBUG] Player %v gets %v points from unplayed tiles",
			// 	game.onturn, unplayedPts)
			game.players[game.onturn].points += unplayedPts
		} else {
			// log.Printf("[DEBUG] Player %v drew new tiles: %v, rack is now %v",
			// 	game.onturn, string(drew), rack)
		}
	case move.MoveTypePass:
		// log.Printf("[DEBUG] Player %v passed", game.onturn)
		game.scorelessTurns++
	}
	if game.scorelessTurns == 6 {
		// log.Printf("[DEBUG] Game ended after 6 scoreless turns")
		game.playing = false
	}
	game.onturn = (game.onturn + 1) % len(game.players)
}

func (game *XWordGame) calculateRackPts(onturn int) int {
	// Calculate the number of pts on the player with the `onturn` rack.
	rack := game.players[onturn].rack
	return rack.ScoreOn(game.numPossibleLetters, game.bag)
}
