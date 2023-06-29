package board

import (
	"github.com/domino14/macondo/tilemapping"
)

const (
	// TrivialCrossSet allows every possible letter. It is the default
	// state of a square.
	TrivialCrossSet = (1 << tilemapping.MaxAlphabetSize) - 1
)

// A CrossSet is a bit mask of letters that are allowed on a square. It is
// inherently directional, as it depends on which direction we are generating
// moves in. If we are generating moves HORIZONTALLY, we check in the
// VERTICAL cross set to make sure we can play a letter there.
// Therefore, a VERTICAL cross set is created by looking at the tile(s)
// above and/or below the relevant square and seeing what letters lead to
// valid words.
type CrossSet uint64

func (c CrossSet) Allowed(letter tilemapping.MachineLetter) bool {
	return c&(1<<uint8(letter)) != 0
}

func (c *CrossSet) Set(letter tilemapping.MachineLetter) {
	*c = *c | (1 << letter)
}

// CrossSetFromString is used for testing only and has undefined
// behavior for multi-char tiles.
func CrossSetFromString(letters string, alph *tilemapping.TileMapping) CrossSet {
	c := CrossSet(0)
	for _, l := range letters {
		v, err := alph.Val(string(l))
		if err != nil {
			panic("Letter error: " + string(l))
		}
		c.Set(v)
	}
	return c
}

func (c *CrossSet) SetAll() {
	*c = TrivialCrossSet
}

func (c *CrossSet) Clear() {
	*c = 0
}
