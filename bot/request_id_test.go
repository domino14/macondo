package bot

import (
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// The response carries the request's ID, so the sender can drop an answer
// to a request it no longer cares about.
func TestHandleEchoesRequestID(t *testing.T) {
	cfg := config.DefaultConfig()
	hist, err := gcgio.ParseGCG(cfg, "../gcgio/testdata/guy_vs_bot_almost_complete.gcg")
	if err != nil {
		t.Fatal(err)
	}
	data, err := proto.Marshal(&pb.BotRequest{GameHistory: hist, RequestId: "17"})
	if err != nil {
		t.Fatal(err)
	}
	b := &Bot{config: cfg}
	resp := b.handle(data)
	if resp.GetError() != "" {
		t.Fatal(resp.GetError())
	}
	if resp.GetMove() == nil {
		t.Fatal("no move in response")
	}
	if resp.RequestId != "17" {
		t.Errorf("request_id = %q, want 17", resp.RequestId)
	}
}
