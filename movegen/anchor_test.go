package movegen

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/cgboard"
)

func TestUpdateAnchors(t *testing.T) {
	alph := alphabet.EnglishAlphabet()

	b := cgboard.MakeBoard(cgboard.CrosswordGameBoard)
	b.SetToGame(alph, cgboard.VsEd)

	anchors := MakeAnchors(b)
	anchors.UpdateAllAnchors()

	if anchors.IsAnchor(3, 3, cgboard.HorizontalDirection) ||
		anchors.IsAnchor(3, 3, cgboard.VerticalDirection) {
		t.Errorf("Should not be an anchor at all")
	}
	if !anchors.IsAnchor(12, 12, cgboard.HorizontalDirection) ||
		!anchors.IsAnchor(12, 12, cgboard.VerticalDirection) {
		t.Errorf("Should be a two-way anchor")
	}
	if !anchors.IsAnchor(4, 3, cgboard.VerticalDirection) ||
		anchors.IsAnchor(4, 3, cgboard.HorizontalDirection) {
		t.Errorf("Should be a vertical but not horizontal anchor")
	}
	// I could do more but it's all right for now?
}
