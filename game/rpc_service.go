package game

import (
	"context"

	pb "github.com/domino14/macondo/rpc/api/proto"
)

// AnnotationService will be the main API that the front end will talk to
// to annotate, simulate, etc. a game.
type AnnotationService struct{}

func (a *AnnotationService) NewGame(ctx context.Context, gameReq *pb.NewGameRequest) (*pb.GameHistory, error) {
	// log := zerolog.Ctx(ctx)
	return nil, nil
}
