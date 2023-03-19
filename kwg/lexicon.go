package kwg

import (
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/tilemapping"
)

type Lexicon struct {
	KWG
}

func (l Lexicon) Name() string {
	return l.LexiconName()
}

func (l Lexicon) HasWord(word tilemapping.MachineWord) bool {
	return FindMachineWord(&l.KWG, word)
}

var daPool = sync.Pool{
	New: func() interface{} {
		return &KWGAnagrammer{}
	},
}

func (l Lexicon) HasAnagram(word tilemapping.MachineWord) bool {
	log.Debug().Str("word", word.UserVisible(l.GetAlphabet())).Msg("has-anagram?")

	da := daPool.Get().(*KWGAnagrammer)
	defer daPool.Put(da)

	v, err := da.IsValidJumble(&l.KWG, word)
	if err != nil {
		log.Err(err).Str("word", word.UserVisible(l.GetAlphabet())).Msg("has-anagram?-error")
		return false
	}

	return v
}
