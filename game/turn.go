package game

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	pb "github.com/domino14/macondo/gen/api/proto"
	"github.com/domino14/macondo/move"
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
	log.Debug().
		Interface("curplayer", curPlayer).
		Interface("move", m.String()).
		Interface("evt", evt).
		Msg("eventFromMove")

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

func modifyForPlaythrough(tiles alphabet.MachineWord, board *board.GameBoard,
	vertical bool, row int, col int) error {

	// modify the tiles array to account for situations in which a letter
	// being played through is not specified as the playthrough marker
	log.Debug().
		Str("tiles", tiles.UserVisible(alphabet.EnglishAlphabet())).
		Msg("Modifying for playthrough")

	currow := row
	curcol := col
	for idx := range tiles {

		if vertical {
			currow = row + idx
		} else {
			curcol = col + idx
		}

		if tiles[idx] != alphabet.PlayedThroughMarker {
			// log.Debug().Int("ml", int(tiles[idx])).Msg("not playthru")
			// This is either a tile we are placing or a tile on the board.
			if !board.GetSquare(currow, curcol).IsEmpty() {
				// We specified a tile on the board already. Make sure
				// that it's the same tile we specified.
				onboard := board.GetSquare(currow, curcol).Letter()
				if onboard != tiles[idx] && onboard.Unblank() != tiles[idx].Unblank() {
					return fmt.Errorf("the play-through tile is incorrect (board %v, specified %v)",
						int(onboard), int(tiles[idx]))
				}
				// Overwrite to be playthroughmarker
				log.Debug().Int("idx", idx).Int("ml", int(tiles[idx])).Msg("Overwriting tile at idx")
				tiles[idx] = alphabet.PlayedThroughMarker
			}
			// Otherwise it's a tile we are placing. Do nothing.

		}

	}
	return nil
}

// Generate a move from an event
func moveFromEvent(evt *pb.GameEvent, alph *alphabet.Alphabet, board *board.GameBoard) *move.Move {
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

		err = modifyForPlaythrough(tiles, board, evt.Direction == pb.GameEvent_VERTICAL,
			int(evt.Row), int(evt.Column))
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

func summary(t *pb.GameTurn) string {
	summary := ""
	firstEvtPlacement := false
	for _, evt := range t.Events {
		switch evt.Type {
		case pb.GameEvent_TILE_PLACEMENT_MOVE:
			summary += fmt.Sprintf("%s played %s %s for %d pts from a rack of %s",
				evt.Nickname, evt.Position, evt.PlayedTiles, evt.Score, evt.Rack)
			firstEvtPlacement = true

		case pb.GameEvent_PASS:
			summary += fmt.Sprintf("%s passed, holding a rack of %s",
				evt.Nickname, evt.Rack)

		case pb.GameEvent_EXCHANGE:
			summary += fmt.Sprintf("%s exchanged %s from a rack of %s",
				evt.Nickname, evt.Exchanged, evt.Rack)

		case pb.GameEvent_CHALLENGE_BONUS:
			summary += fmt.Sprintf(" (+%d)", evt.Bonus)

		case pb.GameEvent_END_RACK_PTS:
			summary += fmt.Sprintf(" (+%d from opponent rack)", evt.EndRackPoints)

		case pb.GameEvent_PHONY_TILES_RETURNED:
			if firstEvtPlacement {
				summary += " (challenged off)"
			}

		case pb.GameEvent_TIME_PENALTY:
			summary += fmt.Sprintf("%s lost %d on time", evt.Nickname, evt.LostScore)

		case pb.GameEvent_END_RACK_PENALTY:
			summary += fmt.Sprintf("%s lost %d from their rack", evt.Nickname,
				evt.LostScore)
		}

	}
	return summary
}
