package turnplayer

import (
	"lukechampine.com/frand"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
)

// GenBestStaticTurn is a useful utility function for sims and autoplaying:
// the best static move for playerIdx, who must be on turn in g.
func GenBestStaticTurn(g *game.Game, p AITurnPlayer, playerIdx int) *move.Move {
	return p.GenerateMovesFor(g, 1)[0]
}

// GenStochasticStaticTurn generates one of the best static turns with a stochastic
// formula. It doesn't always return the top one by equity.
func GenStochasticStaticTurn(g *game.Game, p AITurnPlayer, playerIdx int) *move.Move {
	plays := p.GenerateMovesFor(g, 2)
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
