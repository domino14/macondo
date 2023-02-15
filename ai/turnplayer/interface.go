package turnplayer

import (
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

type AITurnPlayer interface {
	GenerateMoves(numPlays int) []*move.Move
	GetBotType() pb.BotRequest_BotCode
	MoveGenerator() movegen.MoveGenerator
}
