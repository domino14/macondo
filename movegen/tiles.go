package movegen

import (
	"log"

	"github.com/domino14/macondo/alphabet"
)

// Bag - Tile distributions and scores. Hard-code for now, and load from
// some sort of config file later.
type Bag struct {
	numUniqueTiles uint8
	// distributions is a slice of ordered tile distributions, in machine
	// letter order. (0 to max size; max size corresponds to the blank)
	distributions []uint8
	// scores is a slice of ordered scores. The order is the machine letter
	// order.
	scores []int
}

// The functions in this file should eventually be moved to some sort
// of configuration file. We are loading things like tile distributions
// and scores into the alphabet. For now we're just going to hardcode it.
func (b *Bag) Init() {
	// I am hard-coded!!
	b.numUniqueTiles = 27
	// Letter distributions.
	b.distributions = []uint8{
		9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8,
		2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2,
	}
	b.scores = []int{
		1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10,
		1, 1, 1, 1, 4, 4, 8, 4, 10, 0,
	}
}

func (b *Bag) score(ml alphabet.MachineLetter) int {
	if ml >= alphabet.BlankOffset {
		return b.scores[b.numUniqueTiles-1]
	}
	log.Printf("[DEBUG] Looking up machine letter %v", ml)
	return b.scores[ml]
}
