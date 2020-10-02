package game

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/matryer/is"
)

func TestEventFromMove(t *testing.T) {
	is := is.New(t)
	alph := alphabet.EnglishAlphabet()

	tiles, err := alphabet.ToMachineWord("?EGKMNO", alph)
	is.NoErr(err)
	leave, err := alphabet.ToMachineWord("", alph)
	is.NoErr(err)

	m := move.NewExchangeMove(tiles, leave, alph)
	g := &Game{}
	g.players = []*playerState{&playerState{
		PlayerInfo: pb.PlayerInfo{
			Nickname: "foo",
			UserId:   "abcdef",
			RealName: "Foo Bar",
		},
	},
		&playerState{
			PlayerInfo: pb.PlayerInfo{
				Nickname: "botty",
				UserId:   "botbar",
				RealName: "Botty McBotFace",
			},
		},
	}
	g.onturn = 1 // botty's turn
	evt := g.EventFromMove(m)

	is.Equal(evt, &pb.GameEvent{
		Nickname:   "botty",
		Cumulative: 0,
		Rack:       "?EGKMNO",
		Exchanged:  "?EGKMNO",
		Type:       pb.GameEvent_EXCHANGE,
	})

}
