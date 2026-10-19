package lexicon

import (
	"errors"
	"fmt"
	"maps"

	"github.com/domino14/word-golib/tilemapping"
)

// Hybrid accepts a word if the primary lexicon has it, or if the word is short
// enough that the secondary lexicon's answer is allowed to stand in.
//
// It exists to build boards a casual player can read. Generating a position
// from a common-word list alone gives an unnaturally sparse board, because
// almost all of the two-letter words that make parallel plays and hooks
// possible are missing from it: ECWL has 66 twos where NWL23 has 107. Those 41
// extra twos are learnable in an afternoon, while the 450 threes NWL23 adds
// over ECWL are not. So a Hybrid with MaxSecondaryLength 2 takes the twos and
// leaves everything else common.
//
// Both lexicons must share a letter distribution; NewHybrid checks that.
type Hybrid struct {
	Primary            Lexicon
	Secondary          Lexicon
	MaxSecondaryLength int
}

// NewHybrid builds a Hybrid, rejecting a pair whose alphabets disagree -- a
// MachineWord would mean different letters to each of them, so ORing their
// answers would be meaningless.
//
// The check compares the tile-to-value maps rather than the TileMapping
// pointers: every KWG load builds its own TileMapping, so two lexicons on the
// same letter distribution never share one.
func NewHybrid(primary, secondary Lexicon, maxSecondaryLength int) (Hybrid, error) {
	if primary == nil || secondary == nil {
		return Hybrid{}, errors.New("both lexicons must be non-nil")
	}
	pa, sa := primary.GetAlphabet(), secondary.GetAlphabet()
	if pa == nil || sa == nil {
		return Hybrid{}, errors.New("both lexicons must have an alphabet")
	}
	if !maps.Equal(pa.Vals(), sa.Vals()) {
		return Hybrid{}, fmt.Errorf("lexicons %s and %s do not share a letter distribution",
			primary.Name(), secondary.Name())
	}
	return Hybrid{
		Primary:            primary,
		Secondary:          secondary,
		MaxSecondaryLength: maxSecondaryLength,
	}, nil
}

func (h Hybrid) Name() string {
	return fmt.Sprintf("%s+%d:%s", h.Primary.Name(), h.MaxSecondaryLength, h.Secondary.Name())
}

func (h Hybrid) GetAlphabet() *tilemapping.TileMapping {
	return h.Primary.GetAlphabet()
}

func (h Hybrid) HasWord(word Word) bool {
	if h.Primary.HasWord(word) {
		return true
	}
	return len(word) <= h.MaxSecondaryLength && h.Secondary.HasWord(word)
}

func (h Hybrid) HasAnagram(word Word) bool {
	if h.Primary.HasAnagram(word) {
		return true
	}
	return len(word) <= h.MaxSecondaryLength && h.Secondary.HasAnagram(word)
}
