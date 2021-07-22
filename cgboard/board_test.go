package cgboard

import (
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/move"
)

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
