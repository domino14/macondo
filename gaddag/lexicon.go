package gaddag

import (
	"github.com/domino14/macondo/alphabet"
)

type Lexicon struct {
	GenericDawg
}

func (l Lexicon) Name() string {
	return l.LexiconName()
}

func (l Lexicon) HasWord(word alphabet.MachineWord) bool {
	return FindMachineWord(l, word)
}
