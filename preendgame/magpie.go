package preendgame

import (
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/magpie"
)

// topScoreFor is player's best score in g (0 if passing is best), for
// ordering bag draws: what the opponent can do with a draw.
func topScoreFor(g *game.Game, player int) float64 {
	pos, err := g.MagpiePosition()
	if err != nil {
		panic(err)
	}
	if player != g.PlayerOnTurn() {
		pos.SetOnTurn(player)
		defer pos.SetOnTurn(g.PlayerOnTurn())
	}
	moves := pos.Generate(magpie.RecordBest, magpie.SortScore, 0)
	if len(moves) == 0 {
		return 0
	}
	return float64(moves[0].Score) / magpie.EquityResolution
}
