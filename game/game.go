// Package game encapsulates the main mechanics for a Crossword Game. It
// interacts heavily with the protobuf data structures.
package game

import (
	"github.com/domino14/macondo/move"
	pb "github.com/domino14/macondo/rpc/api/proto"
)

func CalculateCoordsFromStringPosition(evt *pb.GameEvent) {
	row, col, vertical := move.FromBoardGameCoords(evt.Position)
	if vertical {
		evt.Direction = pb.GameEvent_VERTICAL
	} else {
		evt.Direction = pb.GameEvent_HORIZONTAL
	}
	evt.Row = int32(row)
	evt.Column = int32(col)
}
