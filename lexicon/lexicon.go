package lexicon

import (
	"github.com/domino14/macondo/alphabet"
)

type Word = alphabet.MachineWord

type Lexicon interface {
	Name() string
	GetAlphabet() *alphabet.Alphabet
	HasWord(word Word) bool
	HasAnagram(word Word) bool
}

type AcceptAll struct {
	Alph *alphabet.Alphabet
}

func (lex AcceptAll) Name() string {
	return "AcceptAll"
}

func (lex AcceptAll) GetAlphabet() *alphabet.Alphabet {
	return lex.Alph
}

func (lex AcceptAll) HasWord(word Word) bool {
	return true
}

func (lex AcceptAll) HasAnagram(word Word) bool {
	return true
}
