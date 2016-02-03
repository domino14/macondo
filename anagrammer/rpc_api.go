package anagrammer

import (
	"errors"
	"github.com/domino14/macondo/gaddag"
	"net/http"
)

type AnagramServiceArgs struct {
	Letters string
	Mode    string
}

type AnagramServiceReply struct {
	Words []string
}

type AnagramService struct{}

func (a *AnagramService) Anagram(r *http.Request, args *AnagramServiceArgs,
	reply *AnagramServiceReply) error {
	// Requires rpcDawg to be initialized.
	// We cast to a SimpleGaddag here because only that one has
	// GetAlphabet defined.
	if gaddag.SimpleGaddag(gaddag.RpcDawg).GetAlphabet() == nil {
		return errors.New("Please load a DAWG first.")
	}
	mode := ModeBuild
	if args.Mode == "exact" {
		mode = ModeExact
	}
	sols := Anagram(args.Letters, gaddag.RpcDawg, mode)
	reply.Words = sols
	return nil
}
