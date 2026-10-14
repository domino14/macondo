package turnplayer

import (
	"context"

	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

type AITurnPlayer interface {
	BestPlay(context.Context) (*move.Move, error)
	// GenerateMovesFor generates the best numPlays moves for the player on
	// turn in g, best first, by this player's static evaluation.
	GenerateMovesFor(g *game.Game, numPlays int) []*move.Move
	GetBotType() pb.BotRequest_BotCode
	MoveGenerator() movegen.MoveGenerator
	SetLastMoves([]*move.Move)
	AddLastMove(*move.Move)
}
