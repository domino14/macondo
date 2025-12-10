package turnplayer

import (
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"lukechampine.com/frand"
)

// GenBestStaticTurn is a useful utility function for sims and autoplaying.
func GenBestStaticTurn(g *game.Game, p AITurnPlayer, playerIdx int) *move.Move {

	mg := p.MoveGenerator().(*movegen.GordonGenerator)
	mg.SetTopPlayOnlyRecorder()
	mg.SetShadowEnabled(true)
	// the equity calculators for its movegen should already have been set if this
	// AITurnPlayer was initialized properly.
	// mg.(*movegen.GordonGenerator).SetGame(g)

	// Add an exchange only if there are 7 or more tiles in the bag.
	// in case we don't have full rack info:
	oppRack := g.RackFor(1 - playerIdx)
	unseen := int(oppRack.NumTiles()) + g.Bag().TilesRemaining()
	exchAllowed := unseen-game.RackTileLimit >= g.ExchangeLimit()
	mg.SetMaxCanExchange(game.MaxCanExchange(unseen-game.RackTileLimit, g.ExchangeLimit()))
	mg.GenAll(g.RackFor(playerIdx), exchAllowed)

	return mg.Plays()[0]
}

// GenStochasticStaticTurn generates one of the best static turns with a stochastic
// formula. It doesn't always return the top one by equity.
// Important note: This function assumes the move generator has been set to
// record the top N plays and thus properly calculates equity.
func GenStochasticStaticTurn(g *game.Game, p AITurnPlayer, playerIdx int) *move.Move {
	mg := p.MoveGenerator()

	oppRack := g.RackFor(1 - playerIdx)
	unseen := int(oppRack.NumTiles()) + g.Bag().TilesRemaining()
	exchAllowed := unseen-game.RackTileLimit >= g.ExchangeLimit()
	mg.SetMaxCanExchange(game.MaxCanExchange(unseen-game.RackTileLimit, g.ExchangeLimit()))
	mg.GenAll(g.RackFor(playerIdx), exchAllowed)

	plays := mg.Plays()
	if len(plays) == 1 {
		return plays[0]
	}
	r := frand.Float64()
	// note: fix for negative equities
	eq1 := plays[0].Equity()
	eq2 := plays[1].Equity()

	eq1sq := eq1 * eq1
	denom := eq1sq + eq2*eq2
	chance := eq1sq / denom
	if chance > r {
		return plays[0]
	}
	return plays[1]
}
