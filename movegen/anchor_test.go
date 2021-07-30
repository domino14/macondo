package movegen

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
)

func TestUpdateAnchors(t *testing.T) {
	alph := alphabet.EnglishAlphabet()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetToGame(alph, board.VsEd)

	anchors := MakeAnchors(b)
	anchors.UpdateAllAnchors()

	if anchors.IsAnchor(3, 3, board.HorizontalDirection) ||
		anchors.IsAnchor(3, 3, board.VerticalDirection) {
		t.Errorf("Should not be an anchor at all")
	}
	if !anchors.IsAnchor(12, 12, board.HorizontalDirection) ||
		!anchors.IsAnchor(12, 12, board.VerticalDirection) {
		t.Errorf("Should be a two-way anchor")
	}
	if !anchors.IsAnchor(4, 3, board.VerticalDirection) ||
		anchors.IsAnchor(4, 3, board.HorizontalDirection) {
		t.Errorf("Should be a vertical but not horizontal anchor")
	}
	// I could do more but it's all right for now?
}
