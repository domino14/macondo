package game

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/testhelpers"
)

func TestEventFromMove(t *testing.T) {
	is := is.New(t)
	alph := testhelpers.EnglishAlphabet()

	tiles, err := tilemapping.ToMachineWord("?EGKMNO", alph)
	is.NoErr(err)

	m := move.NewExchangeMove(tiles, alph)
	g := &Game{}
	g.players = []*playerState{
		{
			PlayerInfo: pb.PlayerInfo{
				Nickname: "foo",
				UserId:   "abcdef",
				RealName: "Foo Bar",
			},
		},
		{
			PlayerInfo: pb.PlayerInfo{
				Nickname: "botty",
				UserId:   "botbar",
				RealName: "Botty McBotFace",
			},
		},
	}
	g.onturn = 1 // botty's turn
	evt := g.EventFromMove(m, "?EGKMNO")

	is.Equal(evt, &pb.GameEvent{
		Cumulative:       0,
		Rack:             "?EGKMNO",
		Exchanged:        "?EGKMNO",
		Type:             pb.GameEvent_EXCHANGE,
		PlayerIndex:      1,
		NumTilesFromRack: 7,
	})

}

func TestMoveFromEventExchange(t *testing.T) {
	is := is.New(t)
	evt := &pb.GameEvent{
		Cumulative:       0,
		Rack:             "?EGKMNO",
		Exchanged:        "GKMO",
		Type:             pb.GameEvent_EXCHANGE,
		PlayerIndex:      1,
		NumTilesFromRack: 4,
	}

	alph := testhelpers.EnglishAlphabet()

	m, err := MoveFromEvent(evt, alph, nil)
	is.NoErr(err)
	is.Equal(m, move.NewExchangeMove(tilemapping.MachineWord{7, 11, 13, 15}, alph))
}

func TestMoveFromEventExchangeBlank(t *testing.T) {
	is := is.New(t)
	evt := &pb.GameEvent{
		Cumulative:       0,
		Rack:             "?EGKMNO",
		Exchanged:        "?",
		Type:             pb.GameEvent_EXCHANGE,
		PlayerIndex:      1,
		NumTilesFromRack: 1,
	}

	alph := testhelpers.EnglishAlphabet()

	m, err := MoveFromEvent(evt, alph, nil)
	is.NoErr(err)
	is.Equal(m, move.NewExchangeMove(tilemapping.MachineWord{0}, alph))
}
