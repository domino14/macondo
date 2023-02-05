package player

import (
	"context"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// GenBestStaticTurn is a useful utility function for sims and autoplaying.
func GenBestStaticTurn(g *game.Game, aiplayer AIPlayer, playerIdx int) *move.Move {
	mg := aiplayer.Movegen()
	mg.SetPlayRecorder(movegen.TopPlayOnlyRecorder)
	mg.SetStrategizer(aiplayer.Strategizer())

	// XXX: This is not ideal, but refactor later:
	mg.(*movegen.GordonGenerator).SetGame(g)

	// Add an exchange only if there are 7 or more tiles in the bag.
	mg.GenAll(g.RackFor(playerIdx), g.Bag().TilesRemaining() >= game.ExchangeLimit)
	return aiplayer.BestPlay(context.Background(), mg.Plays())
}
