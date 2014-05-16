// This is the RPC API for the anagrammer package.
package anagrammer

import (
	"errors"
	"github.com/domino14/macondo/gaddag"
	"net/http"
)

type AnagramServiceArgs struct {
	Mode    string
	Letters string
}

type AnagramServiceReply struct {
	Message string
}

type AnagramService struct{}

func (a *AnagramService) Anagram(r *http.Request, args *AnagramServiceArgs,
	reply *AnagramServiceReply) error {
	var mode uint8
	if len(gaddag.RpcGaddag) == 0 {
		return errors.New("Must first load a GADDAG")
	}

	switch args.Mode {
	case "build":
		mode = ModeBuild
	case "anagram":
		mode = ModeAnagram
	default:
		return errors.New("Must choose a mode: build or anagram")
	}
	Anagram(gaddag.RpcGaddag, args.Letters, mode)
	reply.Message = "Done"
	return nil
}
