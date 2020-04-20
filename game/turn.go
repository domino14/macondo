package game

import (
	"github.com/domino14/macondo/move"
	pb "github.com/domino14/macondo/rpc/api/proto"
)

func (g *Game) curPlayer() *playerState {
	return g.players[g.onturn]
}

func (g *Game) oppPlayer() *playerState {
	return g.players[(g.onturn+1)%2]
}

func (g *Game) eventFromMove(m *move.Move) *pb.GameEvent {
	curPlayer := g.curPlayer()

	evt := &pb.GameEvent{
		Nickname:   curPlayer.Nickname,
		Cumulative: int32(curPlayer.points),
		Rack:       m.FullRack(),
	}

	switch m.Action() {
	case move.MoveTypePlay:
		evt.Position = m.BoardCoords()
		evt.PlayedTiles = m.Tiles().UserVisible(m.Alphabet())
		evt.Score = int32(m.Score())
		evt.Type = pb.GameEvent_TILE_PLACEMENT_MOVE
		CalculateCoordsFromStringPosition(evt)

	case move.MoveTypePass:
		evt.Type = pb.GameEvent_PASS

	case move.MoveTypeExchange:
		evt.Exchanged = m.Tiles().UserVisible(m.Alphabet())
		evt.Type = pb.GameEvent_EXCHANGE

	}
	return evt
}

func (g *Game) endRackEvt(bonusPts int) *pb.GameEvent {
	curPlayer := g.curPlayer()
	otherPlayer := g.oppPlayer()

	evt := &pb.GameEvent{
		Nickname:      curPlayer.Nickname,
		Cumulative:    int32(curPlayer.points),
		Rack:          otherPlayer.rack.String(),
		EndRackPoints: int32(bonusPts),
		Type:          pb.GameEvent_END_RACK_PTS,
	}
	return evt
}
