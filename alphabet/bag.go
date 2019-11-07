package alphabet

import (
	"fmt"
	"math/rand"

	"github.com/rs/zerolog/log"
)

// A Bag is the bag o'tiles!
type Bag struct {
	numUniqueTiles int
	initialTiles   []MachineLetter
	tiles          []MachineLetter
	alphabet       *Alphabet
	// scores is a slice of ordered scores, in machine letter order.
	scores []int
}

// Refill refills the bag.
func (b *Bag) Refill() {
	b.tiles = append([]MachineLetter(nil), b.initialTiles...)
	b.Shuffle()
}

// DrawAtMost draws at most n tiles from the bag. It can draw fewer if there
// are fewer tiles than n, and even draw no tiles at all :o
func (b *Bag) DrawAtMost(n int) []MachineLetter {
	if n > len(b.tiles) {
		n = len(b.tiles)
	}
	drawn, _ := b.Draw(n)
	return drawn
}

// Draw draws n tiles from the bag.
func (b *Bag) Draw(n int) ([]MachineLetter, error) {
	if n > len(b.tiles) {
		return nil, fmt.Errorf("tried to draw %v tiles, tile bag has %v",
			n, len(b.tiles))
	}
	drawn := make([]MachineLetter, n)
	for i := 0; i < n; i++ {
		drawn[i] = b.tiles[i]
	}
	b.tiles = b.tiles[n:]
	return drawn, nil
}

// Shuffle shuffles the bag.
func (b *Bag) Shuffle() {
	rand.Shuffle(len(b.tiles), func(i, j int) {
		b.tiles[i], b.tiles[j] = b.tiles[j], b.tiles[i]
	})
}

// Exchange exchanges the junk in your rack with new tiles.
func (b *Bag) Exchange(letters []MachineLetter) ([]MachineLetter, error) {
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
func (b *Bag) Score(ml MachineLetter) int {
	if ml >= BlankOffset || ml == BlankMachineLetter {
		return b.scores[b.numUniqueTiles-1]
	}
	return b.scores[ml]
}

func (b *Bag) TilesRemaining() int {
	return len(b.tiles)
}

func (b *Bag) GetAlphabet() *Alphabet {
	return b.alphabet
}

func (b *Bag) remove(t MachineLetter) {
	for idx, tile := range b.tiles {
		if tile == t {
			b.tiles[idx] = b.tiles[len(b.tiles)-1]
			b.tiles = b.tiles[:len(b.tiles)-1]
			return
		}
	}
	log.Fatal().Msgf("Tile %c not found in bag", t)
}

// RemoveTiles removes the given tiles from the bag.
func (b *Bag) RemoveTiles(tiles []MachineLetter) {
	for _, t := range tiles {
		if t.IsBlanked() {
			b.remove(BlankMachineLetter)
		} else {
			b.remove(t)
		}
	}
	log.Debug().Msgf("Removed %v tiles", len(tiles))
}

func NewBag(tiles []MachineLetter, numUniqueTiles int,
	alphabet *Alphabet, scores []int) *Bag {

	return &Bag{
		tiles:          tiles,
		initialTiles:   append([]MachineLetter(nil), tiles...),
		numUniqueTiles: numUniqueTiles,
		alphabet:       alphabet,
		scores:         scores,
	}
}

// Copy copies to a new bag and returns it. Note that the initialTiles,
// alphabet, and scores are only shallowly copied. This is fine because
// we don't ever expect these to change after initialization.
func (b *Bag) Copy() *Bag {
	tiles := make([]MachineLetter, len(b.tiles))
	copy(tiles, b.tiles)

	return &Bag{
		tiles:          tiles,
		initialTiles:   b.initialTiles,
		numUniqueTiles: b.numUniqueTiles,
		alphabet:       b.alphabet,
		scores:         b.scores,
	}
}

// CopyFrom copies back the tiles from another bag into this bag. It
// is a shallow copy.
func (b *Bag) CopyFrom(other *Bag) {
	// It's ok to make this a shallow copy. The GC will work its magic.
	// Only copy tiles over; the other stuff is assumed to be fine.
	b.tiles = other.tiles
}
