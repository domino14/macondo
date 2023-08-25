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
	is := is.New(t)
	evt := bot.LambdaEvent{
		CGP:     "15/15/15/D2OBO9/ROARERS8/I5COMMoVED1/P4JAPE2I1U1/T2BIOGEN2R1XU/1WHINE4LA2V/1EA1TYG2DUG1KA/4R4AZO1HE/3WAILFUL1E1A1/4N4I1S1N1/NIT1E8S1/1FORTuITY6 ADEELNQ/ 415/450 0 lex CSW21; tmr 60000/0;",
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
