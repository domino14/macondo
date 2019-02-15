package xwordgame

import (
	"crypto/subtle"
	"errors"
	"net/http"
	"os"
	"path"

	"github.com/domino14/macondo/gaddag"
)

var AuthorizationKey = os.Getenv("AUTH_KEY")
var GaddagDir = os.Getenv("GADDAG_DIR")

type CompVCompServiceArgs struct {
	NumGames        int    `json:"numGames"`
	NumCores        int    `json:"numCores"`
	Computer1Engine string `json:"comp1Engine"`
	Computer2Engine string `json:"comp2Engine"`
	OutputFile      string `json:"outputFile"`
	// Assume GADDAG_DIR is a defined env var where we can find gaddags etc.

	LexiconName string `json:"lexiconName"`
	AuthToken   string `json:"authToken"`
}

type CompVCompServiceReply struct {
	Message string `json:"message"`
}

type CompVCompService struct{}

func (c *CompVCompService) Play(r *http.Request, args *CompVCompServiceArgs,
	reply *CompVCompServiceReply) error {

	if subtle.ConstantTimeCompare([]byte(args.AuthToken), []byte(AuthorizationKey)) != 1 {
		return errors.New("missing or bad auth token")
	}

	gd := gaddag.LoadGaddag(path.Join(GaddagDir, args.LexiconName+".gaddag"))
	if gd.Nodes == nil {
		return errors.New("GADDAG did not seem to exist")
	}

	err := StartCompVCompStaticGames(gd, args.NumGames, args.NumCores, args.OutputFile)
	if err != nil {
		return err
	}
	reply.Message = "ok"
	return nil

}
