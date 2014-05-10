// This is the RPC API for the gaddag package.
package gaddag

import "net/http"

type GaddagServiceArgs struct {
	Filename string
}

type GaddagServiceReply struct {
	Message string
}

type GaddagService struct{}

func (g *GaddagService) Generate(r *http.Request, args *GaddagServiceArgs,
	reply *GaddagServiceReply) error {
	GenerateGaddag(args.Filename)
	reply.Message = "Done"
	return nil
}
