package player

import (
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// GenBestStaticTurn is a useful utility function for sims and autoplaying.
func GenBestStaticTurn(g *game.Game, mg movegen.MoveGenerator,
	aiplayer AIPlayer, playerIdx int) *move.Move {

	mg.SetPlayRecorder(movegen.TopPlayOnlyRecorder)
	mg.SetStrategizer(aiplayer.Strategizer())

	// XXX: This is not ideal, but refactor later:
	mg.(*movegen.GordonGenerator).SetGame(g)

	// Add an exchange only if there are 7 or more tiles in the bag.
	mg.GenAll(g.RackFor(playerIdx), g.Bag().TilesRemaining() >= game.ExchangeLimit)
	return aiplayer.BestPlay(mg.Plays())
}
