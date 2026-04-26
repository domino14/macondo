package game

import (
	"errors"
	"strconv"
	"strings"

	"github.com/domino14/word-golib/tilemapping"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// ExtractLastOppLeave finds the opponent's last tile-placement or exchange event
// in the game history and returns the leave they held after playing.
//
// We do NOT use game.MoveFromEvent here because that function calls
// modifyForPlaythrough against the current board, which by this point already
// has those tiles on it — causing every tile to be flagged as a play-through
// (set to 0) and the computed leave to equal the full rack.
// Instead we parse the leave directly from the event fields without touching
// the board: strip '.' play-through markers from PlayedTiles (GCG notation),
// leaving only the tiles that came from the rack.
func ExtractLastOppLeave(g *Game) ([]tilemapping.MachineLetter, error) {
	evts := g.History().Events[:g.Turn()]
	if len(evts) == 0 {
		return nil, errors.New("no events")
	}
	oppEvtIdx := len(evts) - 1
	oppIdx := evts[oppEvtIdx].PlayerIndex
	for oppEvtIdx >= 0 {
		evt := evts[oppEvtIdx]
		if evt.PlayerIndex != oppIdx {
			break
		}
		if evt.Type == pb.GameEvent_CHALLENGE_BONUS {
			oppEvtIdx--
			continue
		}
		if evt.Type == pb.GameEvent_TILE_PLACEMENT_MOVE {
			rack, err := tilemapping.ToMachineWord(evt.Rack, g.Alphabet())
			if err != nil {
				return nil, err
			}
			// Strip '.' (play-through markers in GCG notation) so we're left
			// with only the tiles that came from the rack.
			playedStr := strings.ReplaceAll(evt.PlayedTiles, ".", "")
			played, err := tilemapping.ToMachineWord(playedStr, g.Alphabet())
			if err != nil {
				return nil, err
			}
			leave, err := tilemapping.Leave(rack, played, true)
			if err != nil {
				return nil, err
			}
			return []tilemapping.MachineLetter(leave), nil
		}
		if evt.Type == pb.GameEvent_EXCHANGE {
			rack, err := tilemapping.ToMachineWord(evt.Rack, g.Alphabet())
			if err != nil {
				return nil, err
			}
			// If only a count was stored we can't determine which tiles were
			// exchanged, so we can't compute the leave.
			if _, err := strconv.Atoi(evt.Exchanged); err == nil {
				return nil, errors.New("exchange event only stores tile count, not tiles")
			}
			exchanged, err := tilemapping.ToMachineWord(evt.Exchanged, g.Alphabet())
			if err != nil {
				return nil, err
			}
			leave, err := tilemapping.Leave(rack, exchanged, false)
			if err != nil {
				return nil, err
			}
			return []tilemapping.MachineLetter(leave), nil
		}
		oppEvtIdx--
	}
	return nil, errors.New("no opponent tile-placement or exchange event found")
}
