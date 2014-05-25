// This is the RPC API for the gaddag package.
package gaddag

import "net/http"

type GaddagServiceArgs struct {
	Filename string
	Minimize bool
}

type GaddagServiceReply struct {
	Message string
}

type GaddagService struct{}

type RPCGaddag []uint32

var RpcGaddag RPCGaddag

func (g *GaddagService) Generate(r *http.Request, args *GaddagServiceArgs,
	reply *GaddagServiceReply) error {
	GenerateGaddag(args.Filename, args.Minimize)
	reply.Message = "Done"
	return nil
}

func (g *GaddagService) Load(r *http.Request, args *GaddagServiceArgs,
	reply *GaddagServiceReply) error {
	RpcGaddag = LoadGaddag(args.Filename)
	reply.Message = "Done"
	return nil
}
