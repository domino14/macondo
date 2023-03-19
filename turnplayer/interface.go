package turnplayer

import (
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
)

// TurnPlayer encapsulates all the functions needed to play a single turn
// of our crossword board game.
type TurnPlayer interface {
	SetPlayerRack(playerid int, letters string) error
	SetCurrentRack(letters string) error
	NewPlacementMove(playerid int, coords string, word string) (*move.Move, error)
	NewPassMove(playerid int) (*move.Move, error)
	NewChallengeMove(playerid int) (*move.Move, error)
	NewExchangeMove(playerid int, letters string) (*move.Move, error)
	MoveFromEvent(evt *pb.GameEvent) (*move.Move, error)
	IsPlaying() bool
	ParseMove(playerid int, lowercase bool, fields []string) (*move.Move, error)
	AssignEquity(plays []*move.Move, oppRack *tilemapping.Rack)
}
