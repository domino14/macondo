package player

import (
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/rs/zerolog/log"
)

// GenBestStaticTurn is a useful utility function for sims and autoplaying.
func GenBestStaticTurn(game *game.Game, movegen movegen.MoveGenerator,
	aiplayer AIPlayer, playerIdx int) *move.Move {

	opp := (playerIdx + 1) % game.NumPlayers()

	// Add an exchange only if there are 7 or more tiles in the bag.
	log.Debug().Msgf("tiles remaining: %v", game.Bag().TilesRemaining())
	movegen.GenAll(game.RackFor(playerIdx), game.Bag().TilesRemaining() >= 7)
	aiplayer.AssignEquity(movegen.Plays(), game.Board(), game.Bag(),
		game.RackFor(opp))
	log.Info().Str("rack",
		game.RackFor(playerIdx).TilesOn().UserVisible(game.Alphabet())).
		Int("plays", len(movegen.Plays())).Msg("generated plays")
	return aiplayer.BestPlay(movegen.Plays())
}
