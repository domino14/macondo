package turnplayer

import (
	"context"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

type AITurnPlayer interface {
	BestPlay(context.Context) (*move.Move, error)
	GetBotType() pb.BotRequest_BotCode
	MoveGenerator() movegen.MoveGenerator
	AddLastMove(*move.Move)
	Reset()
	GetPertinentLogs() []string
}
