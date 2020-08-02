package lexicon

import (
	"github.com/domino14/macondo/alphabet"
)

type Word = alphabet.MachineWord

type Lexicon interface {
	Name() string
	GetAlphabet() *alphabet.Alphabet
	HasWord(word Word) bool
}
