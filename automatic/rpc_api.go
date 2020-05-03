package automatic

import (
	"context"
	"log"
	"os"

	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/rpc/autoplayer"
)

var LexiconPath = os.Getenv("LEXICON_PATH")

type Server struct{ Config *config.Config }

func (s *Server) Play(ctx context.Context, args *pb.PlayRequest) (*pb.PlayResponse, error) {
	log.Println("Got args", args)

	err := StartCompVCompStaticGames(ctx, s.Config, int(args.NumGames), int(args.NumCores),
		"", "", args.OutputFile)
	if err != nil {
		return nil, err
	}
	return &pb.PlayResponse{}, nil
}
