package game

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

func (g *Game) curPlayer() *playerState {
	return g.players[g.onturn]
}

func (g *Game) oppPlayer() *playerState {
	return g.players[(g.onturn+1)%2]
}

func (g *Game) EventFromMove(m *move.Move, rack string) *pb.GameEvent {
	curPlayer := g.curPlayer()

	evt := &pb.GameEvent{
		PlayerIndex: uint32(g.onturn),
		Cumulative:  int32(curPlayer.points),
		Rack:        rack,
	}

	switch m.Action() {
	case move.MoveTypePlay:
		evt.Position = m.BoardCoords()
		evt.PlayedTiles = m.Tiles().UserVisiblePlayedTiles(m.Alphabet())
		evt.Score = int32(m.Score())
		evt.Type = pb.GameEvent_TILE_PLACEMENT_MOVE
		evt.IsBingo = m.TilesPlayed() == RackTileLimit
		evt.NumTilesFromRack = uint32(m.TilesPlayed())
		CalculateCoordsFromStringPosition(evt)

	case move.MoveTypePass:
		evt.Type = pb.GameEvent_PASS

	case move.MoveTypeChallenge:
		evt.Type = pb.GameEvent_CHALLENGE

	case move.MoveTypeUnsuccessfulChallengePass:
		evt.Type = pb.GameEvent_UNSUCCESSFUL_CHALLENGE_TURN_LOSS

	case move.MoveTypeExchange:
		evt.Exchanged = m.Tiles().UserVisible(m.Alphabet())
		evt.Type = pb.GameEvent_EXCHANGE
		evt.NumTilesFromRack = uint32(m.TilesPlayed())

	}
	log.Debug().
		Interface("curplayer", curPlayer).
		Interface("move", m.String()).
		Interface("evt", evt).
		Msg("EventFromMove")

	return evt
}

func (g *Game) endRackEvt(pidx int, bonusPts int) *pb.GameEvent {
	curPlayer := g.players[pidx]
	otherPlayer := g.players[otherPlayer(pidx)]

	evt := &pb.GameEvent{
		PlayerIndex:   uint32(pidx),
		Cumulative:    int32(curPlayer.points),
		Rack:          otherPlayer.rack.String(),
		EndRackPoints: int32(bonusPts),
		Type:          pb.GameEvent_END_RACK_PTS,
	}
	return evt
}

func (g *Game) endRackPenaltyEvt(penalty int) *pb.GameEvent {
	curPlayer := g.curPlayer()

	evt := &pb.GameEvent{
		PlayerIndex: uint32(g.onturn),
		Cumulative:  int32(curPlayer.points),
		Rack:        curPlayer.rack.String(),
		LostScore:   int32(penalty),
		Type:        pb.GameEvent_END_RACK_PENALTY,
	}
	return evt
}

func modifyForPlaythrough(tiles tilemapping.MachineWord, board *board.GameBoard,
	vertical bool, row int, col int) error {

	// modify the tiles array to account for situations in which a letter
	// being played through is not specified as the playthrough marker
	currow := row
	curcol := col
	for idx := range tiles {

		if vertical {
			currow = row + idx
		} else {
			curcol = col + idx
		}
		if currow > board.Dim()-1 || curcol > board.Dim()-1 {
			log.Error().Int("currow", currow).Int("curcol", curcol).Int("dim", board.Dim()).Msg("err-out-of-bounds")
			return errors.New("play out of bounds of board")
		}

		if tiles[idx] != 0 {
			// log.Debug().Int("ml", int(tiles[idx])).Msg("not playthru")
			// This is either a tile we are placing or a tile on the board.
			if board.HasLetter(currow, curcol) {
				// We specified a tile on the board already. Make sure
				// that it's the same tile we specified.
				onboard := board.GetLetter(currow, curcol)
				if onboard != tiles[idx] && onboard.Unblank() != tiles[idx].Unblank() {
					return fmt.Errorf("the play-through tile is incorrect (board %v, specified %v)",
						int(onboard), int(tiles[idx]))
				}
				// Overwrite to be playthroughmarker
				log.Debug().Int("idx", idx).Int("ml", int(tiles[idx])).Msg("Overwriting tile at idx")
				tiles[idx] = 0
			}
			// Otherwise it's a tile we are placing. Do nothing.

		}

	}
	return nil
}

// MoveFromEvent generates a move from an event
func MoveFromEvent(evt *pb.GameEvent, alph *tilemapping.TileMapping, board *board.GameBoard) (*move.Move, error) {
	var m *move.Move

	rack, err := tilemapping.ToMachineWord(evt.Rack, alph)
	if err != nil {
		log.Error().Err(err).Msg("")
		return nil, err
	}

	log.Debug().Int("evt-type", int(evt.Type)).Msg("creating-move-from-event")
	switch evt.Type {
	case pb.GameEvent_TILE_PLACEMENT_MOVE:
		// Calculate tiles, leave, tilesPlayed
		tiles, err := tilemapping.ToMachineWord(evt.PlayedTiles, alph)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil, err
		}

		err = modifyForPlaythrough(tiles, board, evt.Direction == pb.GameEvent_VERTICAL,
			int(evt.Row), int(evt.Column))
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil, err
		}

		leaveMW, err := tilemapping.Leave(rack, tiles, false)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil, err
		}
		// log.Debug().Msgf("calculated leave %v from rack %v, tiles %v",
		// 	leaveMW.UserVisible(alph), rack.UserVisible(alph),
		// 	tiles.UserVisible(alph))
		m = move.NewScoringMove(int(evt.Score), tiles,
			evt.Direction == pb.GameEvent_VERTICAL,
			len(rack)-len(leaveMW), alph, int(evt.Row), int(evt.Column))

	case pb.GameEvent_EXCHANGE:
		ct, err := strconv.Atoi(evt.Exchanged)
		var tiles tilemapping.MachineWord
		if err == nil {
			// The event contains a number of exchanged tiles, instead
			// of the actual tiles.
			// Set the exchanged tiles to just the first N tiles that
			// are provided.
			if len(rack) < ct {
				ct = len(rack)
			}
			tiles = rack[:ct]
		} else {
			tiles, err = tilemapping.ToMachineWord(evt.Exchanged, alph)
			if err != nil {
				log.Error().Err(err).Msg("")
				return nil, err
			}
		}
		_, err = tilemapping.Leave(rack, tiles, true)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil, err
		}
		m = move.NewExchangeMove(tiles, alph)

	case pb.GameEvent_PASS:
		m = move.NewPassMove(alph)

	case pb.GameEvent_CHALLENGE:
		m = move.NewChallengeMove(alph)

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
		log.Debug().Int("mt", int(mt)).Msg("lost-score-move")
		m = move.NewLostScoreMove(mt, rack, int(evt.LostScore))
	case pb.GameEvent_UNSUCCESSFUL_CHALLENGE_TURN_LOSS:
		m = move.NewUnsuccessfulChallengePassMove(alph)
	default:
		log.Error().Msgf("Unhandled event %v", evt)

	}
	return m, nil
}

func summary(players []*pb.PlayerInfo, evt *pb.GameEvent) string {
	summary := ""
	who := players[evt.PlayerIndex].Nickname

	switch evt.Type {
	case pb.GameEvent_TILE_PLACEMENT_MOVE:
		summary = fmt.Sprintf("%s played %s %s for %d pts from a rack of %s",
			who, evt.Position, evt.PlayedTiles, evt.Score, evt.Rack)

	case pb.GameEvent_PASS:
		summary = fmt.Sprintf("%s passed, holding a rack of %s",
			who, evt.Rack)

	case pb.GameEvent_CHALLENGE:
		summary = fmt.Sprintf("%s challenged, holding a rack of %s",
			who, evt.Rack)

	case pb.GameEvent_UNSUCCESSFUL_CHALLENGE_TURN_LOSS:
		summary = fmt.Sprintf("%s challenged unsuccessfully, holding a rack of %s",
			who, evt.Rack)

	case pb.GameEvent_EXCHANGE:
		summary = fmt.Sprintf("%s exchanged %s from a rack of %s",
			who, evt.Exchanged, evt.Rack)

	case pb.GameEvent_CHALLENGE_BONUS:
		summary = fmt.Sprintf(" (+%d)", evt.Bonus)

	case pb.GameEvent_END_RACK_PTS:
		summary = fmt.Sprintf(" (+%d from opponent rack)", evt.EndRackPoints)

	case pb.GameEvent_PHONY_TILES_RETURNED:
		summary = fmt.Sprintf("(%s challenged off)", evt.PlayedTiles)

	case pb.GameEvent_TIME_PENALTY:
		summary = fmt.Sprintf("%s lost %d on time", who, evt.LostScore)

	case pb.GameEvent_END_RACK_PENALTY:
		summary = fmt.Sprintf("%s lost %d from their rack", who,
			evt.LostScore)
	}

	return summary
}
