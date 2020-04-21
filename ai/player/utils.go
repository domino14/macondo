package player

import (
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// GenBestStaticTurn is a useful utility function for sims and autoplaying.
func GenBestStaticTurn(game *game.Game, movegen movegen.MoveGenerator,
	aiplayer AIPlayer, playerID int) *move.Move {

	opp := (playerID + 1) % game.NumPlayers()

	// Add an exchange only if there are 7 or more tiles in the bag.
	movegen.GenAll(game.RackFor(playerID), game.Bag().TilesRemaining() >= 7)
	aiplayer.AssignEquity(movegen.Plays(), game.Board(), game.Bag(),
		game.RackFor(opp))
	return aiplayer.BestPlay(movegen.Plays())
}
