package lexicon

import (
	"github.com/domino14/macondo/tilemapping"
)

type Word = tilemapping.MachineWord

type Lexicon interface {
	Name() string
	GetAlphabet() *tilemapping.TileMapping
	HasWord(word Word) bool
	HasAnagram(word Word) bool
}

type AcceptAll struct {
	Alph *tilemapping.TileMapping
}

func (lex AcceptAll) Name() string {
	return "AcceptAll"
}

func (lex AcceptAll) GetAlphabet() *tilemapping.TileMapping {
	return lex.Alph
}

func (lex AcceptAll) HasWord(word Word) bool {
	return true
}

func (lex AcceptAll) HasAnagram(word Word) bool {
	return true
}
