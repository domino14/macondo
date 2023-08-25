package turnplayer

import (
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// GenBestStaticTurn is a useful utility function for sims and autoplaying.
func GenBestStaticTurn(g *game.Game, p AITurnPlayer, playerIdx int) *move.Move {

	mg := p.MoveGenerator()
	mg.SetPlayRecorder(movegen.TopPlayOnlyRecorder)
	// the equity calculators for its movegen should already have been set if this
	// AITurnPlayer was initialized properly.

	// XXX: This is not ideal, but refactor later:
	mg.(*movegen.GordonGenerator).SetGame(g)

	// Add an exchange only if there are 7 or more tiles in the bag.
	// in case we don't have full rack info:
	oppRack := g.RackFor(1 - playerIdx)
	unseen := oppRack.NumTiles() + uint8(g.Bag().TilesRemaining())
	exchAllowed := unseen-game.RackTileLimit >= game.ExchangeLimit
	mg.GenAll(g.RackFor(playerIdx), exchAllowed)
	return mg.Plays()[0]
}
