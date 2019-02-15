package board

import (
	"testing"

	"github.com/domino14/macondo/gaddag"
)

func BenchmarkBoardTranspose(b *testing.B) {
	// Roughly 270 ns per transpose on my 2013 macbook pro. Two transpositions
	// are needed per full-board move generation; then 2 more per ply
	// So 6 for a 2-ply iteration; assuming 1000 iterations, this is still
	// about 1.6 milliseconds, so we should use board transposition instead
	// of repetitive code.
	board := MakeBoard(CrosswordGameBoard)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		board.Transpose()
	}
}

func TestUpdateAnchors(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")

	b := MakeBoard(CrosswordGameBoard)
	b.SetToGame(gd.GetAlphabet(), VsEd)

	b.UpdateAllAnchors()

	if b.IsAnchor(3, 3, HorizontalDirection) ||
		b.IsAnchor(3, 3, VerticalDirection) {
		t.Errorf("Should not be an anchor at all")
	}
	if !b.IsAnchor(12, 12, HorizontalDirection) ||
		!b.IsAnchor(12, 12, VerticalDirection) {
		t.Errorf("Should be a two-way anchor")
	}
	if !b.IsAnchor(4, 3, VerticalDirection) ||
		b.IsAnchor(4, 3, HorizontalDirection) {
		t.Errorf("Should be a vertical but not horizontal anchor")
	}
	// I could do more but it's all right for now?
}
