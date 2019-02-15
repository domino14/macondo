// This is the RPC API for the gaddag package.
package gaddagmaker

import (
	"crypto/subtle"
	"errors"
	"log"
	"net/http"
	"os"
)

// AuthorizationKey is used for non-user exposed methods
var AuthorizationKey = os.Getenv("AUTH_KEY")

func init() {
	if AuthorizationKey == "" {
		log.Printf("No auth key defined")
	}
}

type GaddagServiceArgs struct {
	Filename  string `json:"filename"`
	Minimize  bool   `json:"minimize"`
	AuthToken string `json:"authToken"`
}

type GaddagServiceReply struct {
	Message string `json:"message"`
}

type GaddagService struct{}

func (g *GaddagService) Generate(r *http.Request, args *GaddagServiceArgs,
	reply *GaddagServiceReply) error {

	if subtle.ConstantTimeCompare([]byte(args.AuthToken), []byte(AuthorizationKey)) != 1 {
		return errors.New("missing or bad auth token")
	}

	GenerateGaddag(args.Filename, args.Minimize, true)
	reply.Message = "Done"
	return nil
}

func (g *GaddagService) GenerateDawg(r *http.Request, args *GaddagServiceArgs,
	reply *GaddagServiceReply) error {

	if subtle.ConstantTimeCompare([]byte(args.AuthToken), []byte(AuthorizationKey)) != 1 {
		return errors.New("missing or bad auth token")
	}

	GenerateDawg(args.Filename, args.Minimize, true)
	reply.Message = "Done"
	return nil
}
