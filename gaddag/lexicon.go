package gaddag

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/rs/zerolog/log"
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

func (l Lexicon) HasAnagram(word alphabet.MachineWord) bool {
	// count the letters
	// XXX:replace with Andy's anagrammer
	log.Debug().Str("word", word.UserVisible(l.GetAlphabet())).Msg("has-anagram?")
	return true
}
