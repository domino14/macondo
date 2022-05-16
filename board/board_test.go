package board

import (
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/move"
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
	alph := alphabet.EnglishAlphabet()

	b := MakeBoard(CrosswordGameBoard)
	b.SetToGame(alph, VsEd)

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

func TestFormedWords(t *testing.T) {
	is := is.New(t)
	b := MakeBoard(CrosswordGameBoard)
	alph := alphabet.EnglishAlphabet()

	b.SetToGame(alph, VsOxy)

	m := move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph)
	words, err := b.FormedWords(m)
	is.NoErr(err)

	is.Equal(len(words), 8)
	// convert all words to user-visible
	uvWords := make([]string, 8)
	for idx, w := range words {
		uvWords[idx] = w.UserVisible(alph)
	}
	is.Equal(uvWords, []string{"OXYPHENBUTAZONE", "OPACIFYING", "XIS", "PREQUALIFIED", "BRAINWASHING",
		"AWAKENERS", "ZONETIME", "EJACULATING"})

}

func TestFormedWordsOneTile(t *testing.T) {
	is := is.New(t)
	b := MakeBoard(CrosswordGameBoard)
	alph := alphabet.EnglishAlphabet()

	b.SetToGame(alph, VsOxy)

	m := move.NewScoringMoveSimple(4, "E8", ".O", "", alph)
	words, err := b.FormedWords(m)
	is.NoErr(err)

	is.Equal(len(words), 2)
	// convert all words to user-visible
	uvWords := make([]string, 2)
	for idx, w := range words {
		uvWords[idx] = w.UserVisible(alph)
	}
	is.Equal(uvWords, []string{"NO", "OO"})

}

func TestFormedWordsHoriz(t *testing.T) {
	is := is.New(t)
	b := MakeBoard(CrosswordGameBoard)
	alph := alphabet.EnglishAlphabet()

	b.SetToGame(alph, VsOxy)

	m := move.NewScoringMoveSimple(12, "14J", "EAR", "", alph)
	words, err := b.FormedWords(m)
	is.NoErr(err)

	is.Equal(len(words), 3)
	// convert all words to user-visible
	uvWords := make([]string, 3)
	for idx, w := range words {
		uvWords[idx] = w.UserVisible(alph)
	}
	is.Equal(uvWords, []string{"EAR", "EN", "AG"})

}

func TestFormedWordsThrough(t *testing.T) {
	is := is.New(t)
	b := MakeBoard(CrosswordGameBoard)
	alph := alphabet.EnglishAlphabet()

	b.SetToGame(alph, VsMatt)

	m := move.NewScoringMoveSimple(4, "K9", "TAEL", "", alph)
	words, err := b.FormedWords(m)
	is.NoErr(err)

	is.Equal(len(words), 5)
	// convert all words to user-visible
	uvWords := make([]string, 5)
	for idx, w := range words {
		uvWords[idx] = w.UserVisible(alph)
	}
	is.Equal(uvWords, []string{"TAEL", "TA", "AN", "RESPONDED", "LO"})
}

func TestFormedWordsBlank(t *testing.T) {
	is := is.New(t)
	b := MakeBoard(CrosswordGameBoard)
	alph := alphabet.EnglishAlphabet()

	b.SetToGame(alph, VsMatt)

	m := move.NewScoringMoveSimple(4, "K9", "TAeL", "", alph)
	words, err := b.FormedWords(m)
	is.NoErr(err)

	is.Equal(len(words), 5)
	// convert all words to user-visible
	uvWords := make([]string, 5)
	for idx, w := range words {
		uvWords[idx] = w.UserVisible(alph)
	}
	is.Equal(uvWords, []string{"TAEL", "TA", "AN", "RESPONDED", "LO"})
}
