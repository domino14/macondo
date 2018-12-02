package board

import "testing"

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
