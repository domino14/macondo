package lexicon

import (
	"fmt"
	"math/rand"

	"github.com/domino14/macondo/alphabet"
)

// A Bag is the bag o'tiles!
type Bag struct {
	numUniqueTiles int
	tiles          []rune
	alphabet       *alphabet.Alphabet
	// scores is a slice of ordered scores, in machine letter order.
	scores     []int
	randomizer *rand.Rand
}

func (b *Bag) SetRandomizer(r *rand.Rand) {
	b.randomizer = r
}

// DrawAtMost draws at most n tiles from the bag. It can draw fewer if there
// are fewer tiles than n, and even draw no tiles at all :o
func (b *Bag) DrawAtMost(n int) []rune {
	if n > len(b.tiles) {
		n = len(b.tiles)
	}
	drawn, _ := b.Draw(n)
	return drawn
}

// Draw draws n tiles from the bag.
func (b *Bag) Draw(n int) ([]rune, error) {
	if n > len(b.tiles) {
		return nil, fmt.Errorf("tried to draw %v tiles, tile bag has %v",
			n, len(b.tiles))
	}
	drawn := make([]rune, n)
	for i := 0; i < n; i++ {
		drawn[i] = b.tiles[i]
	}
	b.tiles = b.tiles[n:]
	return drawn, nil
}

// Shuffle shuffles the bag.
func (b *Bag) Shuffle() {
	b.randomizer.Shuffle(len(b.tiles), func(i, j int) {
		b.tiles[i], b.tiles[j] = b.tiles[j], b.tiles[i]
	})
}

// Exchange exchanges the junk in your rack with new tiles.
func (b *Bag) Exchange(letters []rune) ([]rune, error) {
	newTiles, err := b.Draw(len(letters))
	if err != nil {
		return nil, err
	}
	// put exchanged tiles back into the bag and re-shuffle
	b.tiles = append(b.tiles, letters...)
	b.Shuffle()
	return newTiles, nil
}

// Score gives the score of the given machine letter. This is used by the
// move generator to score plays more rapidly than looking up a map.
func (b *Bag) Score(ml alphabet.MachineLetter) int {
	if ml >= alphabet.BlankOffset {
		return b.scores[b.numUniqueTiles-1]
	}
	return b.scores[ml]
}

func (b *Bag) TilesRemaining() int {
	return len(b.tiles)
}
