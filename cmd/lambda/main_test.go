package main

import (
	"context"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/bot"
	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

func TestHandleRequest(t *testing.T) {
	// t.Skip()
	is := is.New(t)
	evt := bot.LambdaEvent{
		CGP:     "15/15/15/15/10N4/10O4/10S4/5INEPTER3/10L4/10I4/10T4/10E4/15/15/15 ?AAGSTZ/ 70/82 0 lex CSW21; ld english; tmr 824793/1000;",
		GameID:  "foo",
		BotType: int(pb.BotRequest_SIMMING_BOT),
	}
	dc := config.DefaultConfig()
	cfg = &dc
	ctx := context.Background()
	ret, err := HandleRequest(ctx, evt)
	is.NoErr(err)
	is.Equal(ret, " 6A .DEA")
}
