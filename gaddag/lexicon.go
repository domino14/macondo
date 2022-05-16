package gaddag

import (
	"sync"

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

var daPool = sync.Pool{
	New: func() interface{} {
		return DawgAnagrammer{}
	},
}

func (l Lexicon) HasAnagram(word alphabet.MachineWord) bool {
	log.Debug().Str("word", word.UserVisible(l.GetAlphabet())).Msg("has-anagram?")

	da := daPool.Get().(DawgAnagrammer)
	defer daPool.Put(da)

	v, err := da.IsValidJumble(l, word)
	if err != nil {
		log.Err(err).Str("word", word.UserVisible(l.GetAlphabet())).Msg("has-anagram?-error")
		return false
	}

	return v
}
