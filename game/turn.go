package game

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/move"
	pb "github.com/domino14/macondo/rpc/api/proto"
	"github.com/rs/zerolog/log"
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

// Generate a move from an event
func moveFromEvent(evt *pb.GameEvent, alph *alphabet.Alphabet) *move.Move {
	var m *move.Move

	rack, err := alphabet.ToMachineWord(evt.Rack, alph)
	if err != nil {
		log.Error().Err(err).Msg("")
		return nil
	}

	switch evt.Type {
	case pb.GameEvent_TILE_PLACEMENT_MOVE:
		// Calculate tiles, leave, tilesPlayed
		tiles, err := alphabet.ToMachineWord(evt.PlayedTiles, alph)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil
		}

		leaveMW, err := Leave(rack, tiles)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil
		}
		// log.Debug().Msgf("calculated leave %v from rack %v, tiles %v",
		// 	leaveMW.UserVisible(alph), rack.UserVisible(alph),
		// 	tiles.UserVisible(alph))
		m = move.NewScoringMove(int(evt.Score), tiles, leaveMW,
			evt.Direction == pb.GameEvent_VERTICAL,
			len(rack)-len(leaveMW), alph, int(evt.Row), int(evt.Column), evt.Position)

	case pb.GameEvent_EXCHANGE:
		tiles, err := alphabet.ToMachineWord(evt.Exchanged, alph)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil
		}
		leaveMW, err := Leave(rack, tiles)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil
		}
		m = move.NewExchangeMove(tiles, leaveMW, alph)
	case pb.GameEvent_PASS:
		m = move.NewPassMove(rack, alph)

	case pb.GameEvent_CHALLENGE_BONUS:
		m = move.NewBonusScoreMove(move.MoveTypeChallengeBonus,
			rack, int(evt.Bonus))

	case pb.GameEvent_END_RACK_PTS:

		m = move.NewBonusScoreMove(move.MoveTypeEndgameTiles,
			rack, int(evt.EndRackPoints))

		// point loss events:
		// This either happens for:
		// - game over after 6 passes
		// - phony came off the board
		// - international rules at the end of a game
		// - time penalty

	case pb.GameEvent_PHONY_TILES_RETURNED,
		pb.GameEvent_TIME_PENALTY,
		pb.GameEvent_END_RACK_PENALTY:

		var mt move.MoveType

		if evt.Type == pb.GameEvent_PHONY_TILES_RETURNED {
			mt = move.MoveTypePhonyTilesReturned
		} else if evt.Type == pb.GameEvent_END_RACK_PENALTY {
			mt = move.MoveTypeLostTileScore
		} else if evt.Type == pb.GameEvent_TIME_PENALTY {
			mt = move.MoveTypeLostScoreOnTime
		}
		m = move.NewLostScoreMove(mt, rack, int(evt.LostScore))
	default:
		log.Error().Msgf("Unhandled event %v", evt)

	}
	return m
}

// Leave calculates the leave from the rack and the made play.
func Leave(rack alphabet.MachineWord, play alphabet.MachineWord) (alphabet.MachineWord, error) {
	rackmls := map[alphabet.MachineLetter]int{}
	for _, t := range rack {
		rackmls[t]++
	}
	for _, t := range play {
		if t == alphabet.PlayedThroughMarker {
			continue
		}
		if t.IsBlanked() {
			t = alphabet.BlankMachineLetter
		}
		if rackmls[t] != 0 {
			// It should never be 0 unless the GCG is malformed somehow.
			rackmls[t]--
		} else {
			return nil, fmt.Errorf("Tile in play but not in rack: %v %v",
				string(t.UserVisible(alphabet.EnglishAlphabet())), rackmls[t])
		}
	}
	leave := []alphabet.MachineLetter{}
	for k, v := range rackmls {
		if v > 0 {
			for i := 0; i < v; i++ {
				leave = append(leave, k)
			}
		}
	}
	return leave, nil
}
