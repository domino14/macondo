package alphabet

import (
	crypto_rand "crypto/rand"
	"errors"
	"fmt"
	"math/big"

	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"
)

// A Bag is the bag o'tiles!
type Bag struct {
	initialTiles       []MachineLetter
	tiles              []MachineLetter
	initialTileMap     map[MachineLetter]uint8
	tileMap            map[MachineLetter]uint8
	letterDistribution *LetterDistribution
}

func copyTileMap(orig map[MachineLetter]uint8) map[MachineLetter]uint8 {
	tm := make(map[MachineLetter]uint8)
	for k, v := range orig {
		tm[k] = v
	}
	return tm
}

// Refill refills the bag.
func (b *Bag) Refill() {
	b.tiles = append([]MachineLetter(nil), b.initialTiles...)
	b.tileMap = copyTileMap(b.initialTileMap)
	b.Shuffle()
}

// FastDrawAtMost is DrawAtMost using FastDraw.
func (b *Bag) FastDrawAtMost(n int) []MachineLetter {
	if n > len(b.tiles) {
		n = len(b.tiles)
	}
	drawn, _ := b.FastDraw(n)
	return drawn
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

// FastDraw draws n random tiles from the bag. Shuffling is immaterial.
func (b *Bag) FastDraw(n int) ([]MachineLetter, error) {
	if n > len(b.tiles) {
		return nil, fmt.Errorf("tried to draw %v tiles, tile bag has %v",
			n, len(b.tiles))
	}
	// first shuffle the tiles in-place although frand does not error
	l := len(b.tiles)
	k := l - n
	for i := l; i > k; i-- {
		xi := frand.Intn(i)
		// move the selected tile to the end
		b.tiles[i-1], b.tiles[xi] = b.tiles[xi], b.tiles[i-1]
	}
	// now update tileMap
	drawn := make([]MachineLetter, n)
	copy(drawn, b.tiles[k:l])
	for _, v := range drawn {
		b.tileMap[v]--
	}
	b.tiles = b.tiles[:k]
	// log.Debug().Int("numtiles", len(b.tiles)).Int("drew", n).Msg("drew from bag")
	return drawn, nil
}

// Draw draws n crypto/random tiles from the bag. Shuffling is immaterial.
func (b *Bag) Draw(n int) ([]MachineLetter, error) {
	if n > len(b.tiles) {
		return nil, fmt.Errorf("tried to draw %v tiles, tile bag has %v",
			n, len(b.tiles))
	}
	// first shuffle the tiles in-place so we don't lose any tiles if crypto/rand errors out
	l := len(b.tiles)
	k := l - n
	var max big.Int
	for i := l; i > k; i-- {
		max.SetInt64(int64(i))
		x, err := crypto_rand.Int(crypto_rand.Reader, &max)
		if err != nil {
			return nil, fmt.Errorf("tried to draw %v tiles, crypto/rand returned %w", n, err)
		}
		xi := x.Int64()
		// move the selected tile to the end
		b.tiles[i-1], b.tiles[xi] = b.tiles[xi], b.tiles[i-1]
	}
	// now update tileMap
	drawn := make([]MachineLetter, n)
	copy(drawn, b.tiles[k:l])
	for _, v := range drawn {
		b.tileMap[v]--
	}
	b.tiles = b.tiles[:k]
	// log.Debug().Int("numtiles", len(b.tiles)).Int("drew", n).Msg("drew from bag")
	return drawn, nil
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

// FastExchange is Exchange using FastDraw.
func (b *Bag) FastExchange(letters []MachineLetter) ([]MachineLetter, error) {
	newTiles, err := b.FastDraw(len(letters))
	if err != nil {
		return nil, err
	}
	// put exchanged tiles back into the bag and re-shuffle
	b.PutBack(letters)
	return newTiles, nil
}

// Exchange exchanges the junk in your rack with new tiles.
func (b *Bag) Exchange(letters []MachineLetter) ([]MachineLetter, error) {
	newTiles, err := b.Draw(len(letters))
	if err != nil {
		return nil, err
	}
	// put exchanged tiles back into the bag and re-shuffle
	b.PutBack(letters)
	return newTiles, nil
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
	b.Shuffle()
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
	log.Debug().Msgf("reconciling tiles, num in bag are %v, map %v",
		numTilesInBag, b.tileMap)
	if numTilesInBag > len(b.initialTiles) {
		return errors.New("more tiles in the bag that there were to begin with")
	}
	b.tiles = make([]MachineLetter, numTilesInBag)
	idx := 0
	for let, ct := range b.tileMap {
		for j := uint8(0); j < ct; j++ {
			b.tiles[idx] = let
			idx++
		}
	}
	b.Shuffle()
	return nil
}

// FastRedraw is Redraw using FastDraw.
func (b *Bag) FastRedraw(currentRack []MachineLetter) []MachineLetter {
	b.PutBack(currentRack)
	return b.FastDrawAtMost(7)
}

// Redraw is basically a do-over; throw the current rack in the bag
// and draw a new rack.
func (b *Bag) Redraw(currentRack []MachineLetter) []MachineLetter {
	b.PutBack(currentRack)
	return b.DrawAtMost(7)
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
	tileMap := map[MachineLetter]uint8{}

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
	tileMap := make(map[MachineLetter]uint8)
	copy(tiles, b.tiles)
	// Copy map as well
	for k, v := range b.tileMap {
		tileMap[k] = v
	}

	return &Bag{
		tiles:              tiles,
		tileMap:            tileMap,
		initialTiles:       b.initialTiles,
		initialTileMap:     b.initialTileMap,
		letterDistribution: b.letterDistribution,
	}
}

// CopyFrom copies back the tiles from another bag into this bag. The caller
// of this function is responsible for ensuring `other` has the other
// structures we need! (letter distribution, etc).
// It should have been created from the Copy function above.
func (b *Bag) CopyFrom(other *Bag) {
	// This is a deep copy and can be kind of wasteful, but we don't use
	// the bag often.
	if len(other.tiles) == 0 {
		b.tiles = []MachineLetter{}
		b.tileMap = map[MachineLetter]uint8{}
		return
	}
	b.tiles = make([]MachineLetter, len(other.tiles))
	copy(b.tiles, other.tiles)
	b.tileMap = copyTileMap(other.tileMap)
}

func (b *Bag) LetterDistribution() *LetterDistribution {
	return b.letterDistribution
}
