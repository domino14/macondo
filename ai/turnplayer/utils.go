package turnplayer

import (
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// GenBestStaticTurn is a useful utility function for sims and autoplaying.
func GenBestStaticTurn(g *game.Game, mg movegen.MoveGenerator,
	calcs []equity.EquityCalculator, playerIdx int) *move.Move {

	mg.SetPlayRecorder(movegen.TopPlayOnlyRecorder)
	mg.SetEquityCalculators(calcs)

	// XXX: This is not ideal, but refactor later:
	mg.(*movegen.GordonGenerator).SetGame(g)

	// Add an exchange only if there are 7 or more tiles in the bag.
	mg.GenAll(g.RackFor(playerIdx), g.Bag().TilesRemaining() >= game.ExchangeLimit)
	return mg.Plays()[0]
}
