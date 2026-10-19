package alphadawg

import (
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

// Lexicon adapts an alpha DAWG to the lexicon.Lexicon interface. In WordSmog
// a word is valid exactly when some anagram of it is in the lexicon, so
// HasWord and HasAnagram are the same question.
type Lexicon struct {
	*kwg.KWG
}

func (l Lexicon) Name() string {
	return l.LexiconName()
}

func (l Lexicon) GetAlphabet() *tilemapping.TileMapping {
	return l.KWG.GetAlphabet()
}

// HasWord reports whether some anagram of word is in the lexicon. Words of
// fewer than two letters are never valid.
//
// The letters are expected to be designated already -- board letters and
// formed words always are -- so an undesignated blank would be tallied as a
// blank and ignored, same as magpie's is_word_valid_alpha.
func (l Lexicon) HasWord(word tilemapping.MachineWord) bool {
	if len(word) < 2 {
		return false
	}
	var t Tally
	t.AddWord(word)
	return Accepts(l.KWG, &t, int(l.GetAlphabet().NumLetters()))
}

// HasAnagram is identical to HasWord for an alpha DAWG.
func (l Lexicon) HasAnagram(word tilemapping.MachineWord) bool {
	return l.HasWord(word)
}
