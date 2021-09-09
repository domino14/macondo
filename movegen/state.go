package movegen

import (
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
)

type State struct {
	// Movegen-specific current board state
	anchors *Anchors
	csetGen *cross_set.GaddagCrossSetGenerator
}

func (s *State) CopyFrom(other game.BackupableState, b *board.GameBoard) {
	s.anchors.CopyFrom(other.(*State).anchors, b)
	s.csetGen.CopyFrom(other.(*State).csetGen, b)
}

func (s *State) Copy(b *board.GameBoard) game.BackupableState {
	return &State{
		anchors: s.anchors.Copy(b),
		csetGen: s.csetGen.Copy(b),
	}
}

func (s *State) UpdateForMove(b *board.GameBoard, m *move.Move) {
	s.anchors.UpdateAnchorsForMove(m)
	s.csetGen.UpdateForMove(b, m)
}

func (s *State) RecalculateFromBoard(b *board.GameBoard) {
	s.csetGen.GenerateAll(b)
	s.anchors.UpdateAllAnchors()
}
