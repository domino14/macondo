package alphabet

import (
	"errors"
	"fmt"
	"sort"

	"github.com/dgryski/go-pcgr"
	"github.com/rs/zerolog/log"
)

// A Bag is the bag o'tiles!
type Bag struct {
	numTiles        int
	initialNumTiles int

	initialUniqueLetters []MachineLetter
	initialTileMap       map[MachineLetter]uint8
	tileMap              map[MachineLetter]uint8
	letterDistribution   *LetterDistribution
	randSource           *pcgr.Rand
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
	b.tileMap = copyTileMap(b.initialTileMap)
	b.numTiles = b.initialNumTiles
}

// DrawAtMost draws at most n tiles from the bag. It can draw fewer if there
// are fewer tiles than n, and even draw no tiles at all :o
func (b *Bag) DrawAtMost(n int) ([]MachineLetter, error) {
	if n > b.numTiles {
		n = b.numTiles
	}
	return b.Draw(n)
}

func (b *Bag) drawTileAt(idx uint8) (MachineLetter, error) {
	// Draw the tile "at" the given index. We count up the current bag.
	// Count up to "drawn"
	if idx >= uint8(b.numTiles) {
		return 0, errors.New("tile index out of range")
	}
	counter := uint8(0)
	potentialLetterIdx := 0
	var drawn MachineLetter
	for {
		potentialLetter := b.initialUniqueLetters[potentialLetterIdx]
		ct := b.tileMap[potentialLetter]
		counter += ct
		if counter > idx {
			drawn = potentialLetter
			break
		}
		potentialLetterIdx++
	}
	b.tileMap[drawn]--
	b.numTiles--
	return drawn, nil
}

// Draw draws n tiles from the bag.
func (b *Bag) Draw(n int) ([]MachineLetter, error) {
	if n > b.numTiles {
		return nil, fmt.Errorf("tried to draw %v tiles, tile bag has %v",
			n, b.numTiles)
	}
	drawnTiles := make([]MachineLetter, n)
	var err error
	for i := 0; i < n; i++ {
		drawn := uint8(b.randSource.Bound(uint32(b.numTiles)))
		drawnTiles[i], err = b.drawTileAt(drawn)
		if err != nil {
			return nil, err
		}
	}
	return drawnTiles, nil
}

func (b *Bag) Peek() []MachineLetter {
	ret := make([]MachineLetter, b.numTiles)
	idx := 0
	for lt, ct := range b.tileMap {
		for i := uint8(0); i < ct; i++ {
			ret[idx] = lt
			idx++
		}
	}
	return ret
}

// Exchange exchanges the junk in your rack with new tiles.
func (b *Bag) Exchange(letters []MachineLetter) ([]MachineLetter, error) {
	newTiles, err := b.Draw(len(letters))
	if err != nil {
		return nil, err
	}
	// put exchanged tiles back into the bag
	b.PutBack(letters)
	return newTiles, nil
}

// PutBack puts the tiles back in the bag.
func (b *Bag) PutBack(letters []MachineLetter) {
	if len(letters) == 0 {
		return
	}
	for _, ml := range letters {
		b.tileMap[ml]++
	}
	b.numTiles += len(letters)
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
	return b.numTiles
}

func (b *Bag) remove(t MachineLetter) {
	if b.tileMap[t] == 0 {
		log.Fatal().Msgf("Tile %c not found in bag", t)
	}
	b.tileMap[t]--
}

// Redraw is basically a do-over; throw the current rack in the bag
// and draw a new rack.
func (b *Bag) Redraw(currentRack []MachineLetter) ([]MachineLetter, error) {
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
	b.numTiles -= len(tiles)
	return nil
}

func NewBag(ld *LetterDistribution, alph *Alphabet, randSource *pcgr.Rand) *Bag {
	tileMap := map[MachineLetter]uint8{}

	idx := 0
	initialUniqueLetters := []MachineLetter{}
	for rn, ct := range ld.Distribution {
		val, err := alph.Val(rn)
		if err != nil {
			log.Fatal().Msgf("Attempt to initialize bag failed: %v", err)
		}
		tileMap[val] = ct
		idx += int(ct)

		initialUniqueLetters = append(initialUniqueLetters, val)
	}

	sort.Slice(initialUniqueLetters, func(a, b int) bool {
		return initialUniqueLetters[a] < initialUniqueLetters[b]
	})

	return &Bag{
		tileMap:         tileMap,
		numTiles:        idx,
		initialNumTiles: idx,
		initialTileMap:  copyTileMap(tileMap),

		initialUniqueLetters: initialUniqueLetters,
		letterDistribution:   ld,
		randSource:           randSource,
	}
}

// Copy copies to a new bag and returns it. Note that the initialTiles
// are only shallowly copied. This is fine because
// we don't ever expect these to change after initialization.
// If randSource is not nil, it is set as the rand source for the copy.
// Otherwise, use the original's rand source.
func (b *Bag) Copy(randSource *pcgr.Rand) *Bag {
	tileMap := make(map[MachineLetter]uint8)
	// Copy map
	for k, v := range b.tileMap {
		tileMap[k] = v
	}
	if randSource == nil {
		randSource = b.randSource
	}

	return &Bag{
		tileMap:              tileMap,
		numTiles:             b.numTiles,
		initialNumTiles:      b.initialNumTiles,
		initialUniqueLetters: b.initialUniqueLetters,

		initialTileMap:     b.initialTileMap,
		letterDistribution: b.letterDistribution,
		randSource:         randSource,
	}
}

// CopyFrom copies back the tiles from another bag into this bag. The caller
// of this function is responsible for ensuring `other` has the other
// structures we need! (letter distribution, etc).
// It should have been created from the Copy function above.
func (b *Bag) CopyFrom(other *Bag) {
	// This is a deep copy and can be kind of wasteful, but we don't use
	// the bag often.
	if len(other.tileMap) == 0 {
		b.tileMap = map[MachineLetter]uint8{}
		return
	}
	b.tileMap = copyTileMap(other.tileMap)
	b.randSource = other.randSource
}

func (b *Bag) LetterDistribution() *LetterDistribution {
	return b.letterDistribution
}
