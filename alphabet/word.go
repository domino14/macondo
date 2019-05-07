package alphabet

import (
	"sort"
)

type Word struct {
	Word    string
	Dist    *LetterDistribution
	letters []rune
}

func (w Word) String() string {
	return w.Word
}

func (w Word) Len() int {
	return len(w.letters)
}

func (w Word) Less(i, j int) bool {
	return w.Dist.SortOrder[w.letters[i]] < w.Dist.SortOrder[w.letters[j]]
}

func (w Word) Swap(i, j int) {
	w.letters[i], w.letters[j] = w.letters[j], w.letters[i]
}

func (w Word) MakeAlphagram() string {
	w.letters = []rune{}
	for _, char := range w.Word {
		w.letters = append(w.letters, char)
	}
	sort.Sort(w)
	return string(w.letters)
}
