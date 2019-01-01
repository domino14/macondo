package xwordgame

// Data collection for XWordGame. Allow computer vs computer games, etc.

import (
	"log"

	"github.com/domino14/macondo/gaddag"
)

// CompVsCompStatic plays out a game to the end using best static turns.
func (game *XWordGame) CompVsCompStatic(gd gaddag.SimpleGaddag) {
	game.Init(gd)
	game.StartGame()
	for game.playing {
		game.PlayBestStaticTurn(game.onturn)
		log.Printf("Turn %v", game.turnnum)
		log.Println(game.board.ToDisplayText(game.gaddag.GetAlphabet()))
	}
	log.Printf("[DEBUG] Game over. Score: %v - %v", game.players[0].points,
		game.players[1].points)
}
