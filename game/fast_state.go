package game

import pb "github.com/domino14/macondo/gen/api/proto/macondo"

// Some helper functions to rapidly serialize from and to a pb.GameState

// ToGameState turns the game into a pb.GameState. A GameState is a fast,
// minimal representation of the game state.
func (g *Game) ToGameState() *pb.GameState {

	st := &pb.GameState{
		Board:          g.board.ToBoardState(),
		Bag:            g.bag.ToBagState(),
		History:        g.history,
		ScorelessTurns: int32(g.scorelessTurns),
		Onturn:         int32(g.onturn),
	}
	return st
}
