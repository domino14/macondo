package alphabet

import (
	"errors"
	"fmt"

	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"
)

// A Bag is the bag o'tiles!
type Bag struct {
	initialTiles []MachineLetter
	tiles        []MachineLetter
	// the tile "map" is just a slice with the index being the machine letter
	// and the value being the number of tiles.
	initialTileMap     []uint8
	tileMap            []uint8
	letterDistribution *LetterDistribution
	fixedOrder         bool
}

func copyTileMap(orig []uint8) []uint8 {
	tm := make([]uint8, len(orig))
	copy(tm, orig)
	return tm
}

// SetFixedOrder makes the bag drawing algorithm repeatable if the bag is copied
// before drawing. This can be useful for more accurate Monte Carlo sims.
// It is extremely recommended to do a shuffle first if you want to use fixed order.
func (b *Bag) SetFixedOrder(f bool) {
	b.fixedOrder = f
}

// Refill refills the bag.
func (b *Bag) Refill() {
	b.tiles = append([]MachineLetter(nil), b.initialTiles...)
	b.tileMap = copyTileMap(b.initialTileMap)
	b.Shuffle()
}

// DrawAtMost draws at most n tiles from the bag. It can draw fewer if there
// are fewer tiles than n, and even draw no tiles at all :o
// This is a zero-alloc draw into the passed-in slice.
func (b *Bag) DrawAtMost(n int, ml []MachineLetter) int {
	if n > len(b.tiles) {
		n = len(b.tiles)
	}
	b.Draw(n, ml)
	return n
}

// Draw draws n random tiles from the bag. Shuffling is immaterial if fixedOrder is false.
// Returns the number of tiles actually drawn.
// This is a zero-alloc draw into the passed-in slice.
// NOTE: this function does not resize ml at all. It must
// be the correct size to allow tiles to fit in!
func (b *Bag) Draw(n int, ml []MachineLetter) error {
	if n > len(b.tiles) {
		return fmt.Errorf("tried to draw %v tiles, tile bag has %v",
			n, len(b.tiles))
	}
	l := len(b.tiles)
	k := l - n

	if !b.fixedOrder {
		for i := l; i > k; i-- {
			xi := frand.Intn(i)
			// move the selected tile to the end
			b.tiles[i-1], b.tiles[xi] = b.tiles[xi], b.tiles[i-1]
		}
	}

	copy(ml, b.tiles[k:l])

	// now update tileMap
	for _, v := range ml[:n] {
		b.tileMap[v]--
	}
	b.tiles = b.tiles[:k]
	// log.Debug().Int("numtiles", len(b.tiles)).Int("drew", n).Msg("drew from bag")
	return nil
}

func (b *Bag) Peek() []MachineLetter {
	ret := make([]MachineLetter, len(b.tiles))
	copy(ret, b.tiles)
	return ret
}

// Shuffle shuffles the bag.
func (b *Bag) Shuffle() {
	// log.Debug().Int("numtiles", len(b.tiles)).Msg("shuffling bag")
	frand.Shuffle(len(b.tiles), func(i, j int) {
		b.tiles[i], b.tiles[j] = b.tiles[j], b.tiles[i]
	})
}

// Exchange exchanges the junk in your rack with new tiles.
func (b *Bag) Exchange(letters []MachineLetter, ml []MachineLetter) error {
	err := b.Draw(len(letters), ml)
	if err != nil {
		return err
	}
	// put exchanged tiles back into the bag and re-shuffle
	b.PutBack(letters)
	return nil
}

// PutBack puts the tiles back in the bag, and shuffles the bag.
func (b *Bag) PutBack(letters []MachineLetter) {
	if len(letters) == 0 {
		return
	}
	b.tiles = append(b.tiles, letters...)
	for _, ml := range letters {
		b.tileMap[ml]++
	}
	if b.fixedOrder {
		b.Shuffle()
	}
}

// hasRack returns a boolean indicating whether the passed-in rack is
// in the bag, in its entirety.
func (b *Bag) hasRack(letters []MachineLetter) bool {
	submap := make(map[MachineLetter]uint8)

	for _, ml := range letters {
		if ml.IsBlanked() {
			submap[BlankMachineLetter]++
		} else {
			submap[ml]++
		}
	}
	// check every single letter we have.
	for ml, ct := range submap {
		if b.tileMap[ml] < ct {
			return false
		}
	}

	return true
}

func (b *Bag) TilesRemaining() int {
	return len(b.tiles)
}

func (b *Bag) remove(t MachineLetter) {
	if b.tileMap[t] == 0 {
		log.Fatal().Msgf("Tile %c not found in bag", t)
	}
	b.tileMap[t]--
}

// rebuildTileSlice reconciles the bag slice with the tile map.
func (b *Bag) rebuildTileSlice(numTilesInBag int) error {
	log.Trace().Msgf("reconciling tiles, num in bag are %v, map %v",
		numTilesInBag, b.tileMap)
	if numTilesInBag > len(b.initialTiles) {
		return errors.New("more tiles in the bag that there were to begin with")
	}
	if cap(b.tiles) < numTilesInBag {
		b.tiles = make([]MachineLetter, numTilesInBag)
	}
	b.tiles = b.tiles[:numTilesInBag]
	idx := 0
	for ml, ct := range b.tileMap {
		for j := uint8(0); j < ct; j++ {
			b.tiles[idx] = MachineLetter(ml)
			idx++
		}
	}
	b.Shuffle()
	return nil
}

// Redraw is basically a do-over; throw the current rack in the bag
// and draw a new rack.
func (b *Bag) Redraw(currentRack []MachineLetter, ml []MachineLetter) int {
	b.PutBack(currentRack)
	return b.DrawAtMost(7, ml)
}

// RemoveTiles removes the given tiles from the bag, and returns an error
// if it can't.
func (b *Bag) RemoveTiles(tiles []MachineLetter) error {
	if !b.hasRack(tiles) {
		return fmt.Errorf("cannot remove the tiles %v from the bag, as they are not in the bag",
			MachineWord(tiles).UserVisible(b.LetterDistribution().alph))
	}
	for _, t := range tiles {
		if t.IsBlanked() {
			b.remove(BlankMachineLetter)
		} else {
			b.remove(t)
		}
	}
	return b.rebuildTileSlice(len(b.tiles) - len(tiles))
}

func NewBag(ld *LetterDistribution, alph *Alphabet) *Bag {

	tiles := make([]MachineLetter, ld.numLetters)
	// gotta fix this, this is dumb. A should start at 1. Blank should be 0.
	tileMap := make([]uint8, MaxAlphabetSize+1)

	idx := 0
	for rn, ct := range ld.Distribution {
		val, err := alph.Val(rn)
		if err != nil {
			log.Fatal().Msgf("Attempt to initialize bag failed: %v", err)
		}
		tileMap[val] = ct
		for j := uint8(0); j < ct; j++ {
			tiles[idx] = val
			idx++
		}
	}
	return &Bag{
		tiles:              tiles,
		tileMap:            tileMap,
		initialTiles:       append([]MachineLetter(nil), tiles...),
		initialTileMap:     copyTileMap(tileMap),
		letterDistribution: ld,
	}
}

// Copy copies to a new bag and returns it. Note that the initialTiles
// are only shallowly copied. This is fine because
// we don't ever expect these to change after initialization.
func (b *Bag) Copy() *Bag {
	tiles := make([]MachineLetter, len(b.tiles))
	tileMap := make([]uint8, MaxAlphabetSize+1)
	copy(tiles, b.tiles)
	copy(tileMap, b.tileMap)

	return &Bag{
		tiles:              tiles,
		tileMap:            tileMap,
		initialTiles:       b.initialTiles,
		initialTileMap:     b.initialTileMap,
		letterDistribution: b.letterDistribution,
		fixedOrder:         b.fixedOrder,
	}
}

// CopyFrom copies back the tiles from another bag into this bag. The caller
// of this function is responsible for ensuring `other` has the other
// structures we need! (letter distribution, etc).
// It should have been created from the Copy function above.
func (b *Bag) CopyFrom(other *Bag) {

	if cap(b.tiles) < len(other.tiles) {
		b.tiles = make([]MachineLetter, len(other.tiles))
	}
	b.tiles = b.tiles[:len(other.tiles)]
	copy(b.tiles, other.tiles)

	if cap(b.tileMap) < len(other.tileMap) {
		b.tileMap = make([]uint8, len(other.tileMap))
	}
	b.tileMap = b.tileMap[:len(other.tileMap)]
	copy(b.tileMap, other.tileMap)
	b.fixedOrder = other.fixedOrder
}

func (b *Bag) LetterDistribution() *LetterDistribution {
	return b.letterDistribution
}
