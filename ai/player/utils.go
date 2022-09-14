package player

import (
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	//"github.com/rs/zerolog/log"
)

// GenBestStaticTurn is a useful utility function for sims and autoplaying.
func GenBestStaticTurn(g *game.Game, mg movegen.MoveGenerator,
	aiplayer AIPlayer, playerIdx int) *move.Move {

	mg.SetPlayRecorder(movegen.TopPlayOnlyRecorder)
	mg.SetStrategizer(aiplayer.Strategizer())

	// XXX: This is not ideal, but refactor later:
	mg.(*movegen.GordonGenerator).SetGame(g)

	// Add an exchange only if there are 7 or more tiles in the bag.
	mg.(*movegen.GordonGenerator).ResetCrossesAndAnchors()
	mg.GenAll(g.RackFor(playerIdx), g.Bag().TilesRemaining() >= game.ExchangeLimit)
	//log.Debug().Str("rack",
	// 	g.RackFor(playerIdx).TilesOn().UserVisible(g.Alphabet())).
	// 	Int("plays", len(mg.Plays())).Msg("plays generated")
	return aiplayer.BestPlay(mg.Plays())
}
