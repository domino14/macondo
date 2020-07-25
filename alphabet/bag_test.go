package alphabet

import (
	"errors"
	"os"
	"testing"
	"time"

	"github.com/dgryski/go-pcgr"
	"github.com/domino14/macondo/config"
	"github.com/matryer/is"
)

var randSource = pcgr.New(time.Now().UnixNano(), 42)

var DefaultConfig = config.Config{
	StrategyParamsPath:        os.Getenv("STRATEGY_PARAMS_PATH"),
	LetterDistributionPath:    os.Getenv("LETTER_DISTRIBUTION_PATH"),
	LexiconPath:               os.Getenv("LEXICON_PATH"),
	DefaultLexicon:            "NWL18",
	DefaultLetterDistribution: "English",
}

func TestBag(t *testing.T) {
	is := is.New(t)
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	bag := ld.MakeBag(&randSource)
	is.Equal(bag.numTiles, ld.numLetters)

	tileMap := make(map[rune]uint8)
	numTiles := 0
	for i := 0; i < bag.initialNumTiles; i++ {
		tiles, err := bag.Draw(1)
		is.NoErr(err)
		numTiles++
		uv := tiles[0].UserVisible(ld.Alphabet())
		t.Logf("Drew a %c!, %v (in bag %v)", uv, numTiles, bag.numTiles)
		is.NoErr(err)
		tileMap[uv]++
	}
	is.Equal(tileMap, ld.Distribution)
	_, err = bag.Draw(1)
	is.True(err != nil)
	is.Equal(bag.initialUniqueLetters,
		[]MachineLetter{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
			16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 50})

}

func TestDraw(t *testing.T) {
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bag := ld.MakeBag(&randSource)

	letters, _ := bag.Draw(7)
	if len(letters) != 7 {
		t.Errorf("Length was %v, expected 7", len(letters))
	}
	if bag.numTiles != 93 {
		t.Errorf("Length was %v, expected 93", bag.numTiles)
	}
}

func TestDrawAtMost(t *testing.T) {
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bag := ld.MakeBag(&randSource)

	for i := 0; i < 14; i++ {
		letters, _ := bag.Draw(7)
		if len(letters) != 7 {
			t.Errorf("Length was %v, expected 7", len(letters))
		}
	}
	if bag.TilesRemaining() != 2 {
		t.Errorf("TilesRemaining was %v, expected 2", bag.TilesRemaining())
	}
	letters, err := bag.DrawAtMost(7)
	if err != nil {
		t.Error(err)
	}
	if len(letters) != 2 {
		t.Errorf("Length was %v, expected 2", len(letters))
	}
	if bag.TilesRemaining() != 0 {
		t.Errorf("TilesRemaining was %v, expected 0", bag.TilesRemaining())
	}
	// Try to draw one more time.
	letters, err = bag.DrawAtMost(7)
	if err != nil {
		t.Error(err)
	}
	if len(letters) != 0 {
		t.Errorf("Length was %v, expected 0", len(letters))
	}
	if bag.TilesRemaining() != 0 {
		t.Errorf("TilesRemaining was %v, expected 0", bag.TilesRemaining())
	}
}

func TestExchange(t *testing.T) {
	is := is.New(t)
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bag := ld.MakeBag(&randSource)

	letters, _ := bag.Draw(7)
	newLetters, _ := bag.Exchange(letters[:5])
	is.Equal(len(newLetters), 5)
	is.Equal(bag.numTiles, 93)
}

func TestRemoveTiles(t *testing.T) {
	is := is.New(t)
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bag := ld.MakeBag(&randSource)
	is.Equal(bag.numTiles, 100)
	toRemove := []MachineLetter{
		9, 14, 24, 4, 3, 20, 4, 11, 21, 6, 22, 14, 8, 0, 8, 15, 6, 5, 4,
		19, 0, 24, 8, 17, 17, 18, 2, 11, 8, 14, 1, 8, 0, 20, 7, 0, 8, 10,
		0, 11, 13, 25, 11, 14, 5, 8, 19, 4, 12, 8, 18, 4, 3, 19, 14, 19,
		1, 0, 13, 4, 19, 14, 4, 17, 20, 6, 21, 104, 3, 7, 0, 3, 14, 22,
		4, 8, 13, 16, 20, 4, 18, 19, 4, 23, 4, 2, 17, 12, 14, 0, 13,
	}
	is.Equal(len(toRemove), 91)
	err = bag.RemoveTiles(toRemove)
	if err != nil {
		t.Error(err)
	}
	is.Equal(bag.numTiles, 9)
}

func TestDrawTileAt(t *testing.T) {
	is := is.New(t)
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bag := ld.MakeBag(&randSource)

	tile, err := bag.drawTileAt(0)
	is.NoErr(err)
	is.Equal(MachineLetter(0), tile)
	is.Equal(bag.numTiles, 99)

	tile, err = bag.drawTileAt(99)
	is.Equal(MachineLetter(0), tile)
	is.Equal(err, errors.New("tile index out of range"))

	tile, err = bag.drawTileAt(98)
	is.Equal(MachineLetter(BlankMachineLetter), tile)
	is.NoErr(err)

	tile, err = bag.drawTileAt(8)
	is.Equal(MachineLetter(1), tile)
	is.NoErr(err)

	tile, err = bag.drawTileAt(8)
	is.Equal(MachineLetter(1), tile)
	is.NoErr(err)

	tile, err = bag.drawTileAt(8)
	is.Equal(MachineLetter(2), tile)
	is.NoErr(err)

	// is.Equal(MachineLetter(BlankMachineLetter), bag.drawTileAt(99))
	// is.Equal(MachineLetter(BlankMachineLetter), bag.drawTileAt(98))

}

func TestDrawTileAtSimple(t *testing.T) {
	is := is.New(t)

	bag := &Bag{
		numTiles:             3,
		initialNumTiles:      3,
		initialUniqueLetters: []MachineLetter{0, 1, 2},
		initialTileMap:       map[MachineLetter]uint8{0: 1, 1: 1, 2: 1},
		tileMap:              map[MachineLetter]uint8{0: 1, 1: 1, 2: 1},
		letterDistribution:   nil,
		randSource:           nil,
	}
	tile, err := bag.drawTileAt(1)
	is.NoErr(err)
	is.Equal(MachineLetter(1), tile)
	is.Equal(bag.numTiles, 2)
	is.Equal(bag.tileMap, map[MachineLetter]uint8{0: 1, 1: 0, 2: 1})

	tile, err = bag.drawTileAt(1)
	is.NoErr(err)
	is.Equal(MachineLetter(2), tile)
	is.Equal(bag.numTiles, 1)
}

func TestDrawTileAtSimple2(t *testing.T) {
	is := is.New(t)

	bag := &Bag{
		numTiles:             3,
		initialNumTiles:      3,
		initialUniqueLetters: []MachineLetter{0, 1, 2},
		initialTileMap:       map[MachineLetter]uint8{0: 1, 1: 1, 2: 1},
		tileMap:              map[MachineLetter]uint8{0: 1, 1: 1, 2: 1},
		letterDistribution:   nil,
		randSource:           nil,
	}
	tile, err := bag.drawTileAt(2)
	is.NoErr(err)
	is.Equal(MachineLetter(2), tile)
	is.Equal(bag.numTiles, 2)
	is.Equal(bag.tileMap, map[MachineLetter]uint8{0: 1, 1: 1, 2: 0})

	tile, err = bag.drawTileAt(1)
	is.NoErr(err)
	is.Equal(MachineLetter(1), tile)
	is.Equal(bag.numTiles, 1)

	tile, err = bag.drawTileAt(0)
	is.NoErr(err)
	is.Equal(MachineLetter(0), tile)
	is.Equal(bag.numTiles, 0)
}

func TestCopyBag(t *testing.T) {
	is := is.New(t)
	seed := int64(42)
	inc := int64(69)

	randSource := pcgr.New(seed, inc)

	ld, err := EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	bag := ld.MakeBag(&randSource)

	drawn := []MachineLetter{}

	for i := 0; i < 100; i++ {
		t, err := bag.Draw(1)
		is.NoErr(err)
		drawn = append(drawn, t...)
	}

	// New bag
	randSource = pcgr.New(seed, inc)
	newBag := ld.MakeBag(&randSource)
	newDrawn := []MachineLetter{}

	for i := 0; i < 51; i++ {
		t, err := newBag.Draw(1)
		is.NoErr(err)
		newDrawn = append(newDrawn, t...)
	}

	newBag2 := newBag.Copy(&randSource)
	for i := 0; i < 7; i++ {
		t, err := newBag2.Draw(7)
		is.NoErr(err)
		newDrawn = append(newDrawn, t...)
	}
	is.Equal(drawn, newDrawn)
}
