package automatic

import (
	"context"
	"errors"
	"log"
	"os"
	"path"

	"github.com/domino14/macondo/gaddag"
	pb "github.com/domino14/macondo/rpc/autoplayer"
)

var LexiconPath = os.Getenv("LEXICON_PATH")

type Server struct{}

func (s *Server) Play(ctx context.Context, args *pb.PlayRequest) (*pb.PlayResponse, error) {
	log.Println("Got args", args)

	gd, _ := gaddag.LoadGaddag(path.Join(LexiconPath, "gaddag", args.LexiconName+".gaddag"))
	if gd.Nodes == nil {
		return nil, errors.New("GADDAG did not seem to exist")
	}

	err := StartCompVCompStaticGames(gd, int(args.NumGames), int(args.NumCores),
		args.OutputFile)
	if err != nil {
		return nil, err
	}
	return &pb.PlayResponse{}, nil
}
