package tilemapping

import (
	"reflect"
	"sort"
	"testing"

	"github.com/matryer/is"
)

func TestBag(t *testing.T) {
	is := is.New(t)

	ld, err := EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	bag := ld.MakeBag()
	if len(bag.tiles) != ld.numLetters {
		t.Error("Tile bag and letter distribution do not match.")
	}
	tileMap := make(map[rune]uint8)
	numTiles := 0
	ml := make([]MachineLetter, 7)

	for range bag.tiles {
		err := bag.Draw(1, ml)
		numTiles++
		uv := ml[0].UserVisible(ld.tilemapping, false)
		t.Logf("Drew a %c! , %v", uv, numTiles)
		if err != nil {
			t.Error("Error drawing from tile bag.")
		}
		tileMap[uv]++
	}
	if !reflect.DeepEqual(tileMap, ld.Distribution) {
		t.Error("Distribution and tilemap were not identical.")
	}
	err = bag.Draw(1, ml)
	if err == nil {
		t.Error("Should not have been able to draw from an empty bag.")
	}
}

func TestDraw(t *testing.T) {
	is := is.New(t)

	ld, err := EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	bag := ld.MakeBag()
	ml := make([]MachineLetter, 7)
	err = bag.Draw(7, ml)
	is.NoErr(err)

	if len(bag.tiles) != 93 {
		t.Errorf("Length was %v, expected 93", len(bag.tiles))
	}
}

func TestDrawAtMost(t *testing.T) {
	is := is.New(t)

	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bag := ld.MakeBag()
	ml := make([]MachineLetter, 7)
	for i := 0; i < 14; i++ {
		err := bag.Draw(7, ml)
		is.NoErr(err)
	}
	if bag.TilesRemaining() != 2 {
		t.Errorf("TilesRemaining was %v, expected 2", bag.TilesRemaining())
	}
	drawn := bag.DrawAtMost(7, ml)
	if drawn != 2 {
		t.Errorf("drawn was %v, expected 2", drawn)
	}
	if bag.TilesRemaining() != 0 {
		t.Errorf("TilesRemaining was %v, expected 0", bag.TilesRemaining())
	}
	// Try to draw one more time.
	drawn = bag.DrawAtMost(7, ml)
	if drawn != 0 {
		t.Errorf("drawn was %v, expected 0", drawn)
	}
	if bag.TilesRemaining() != 0 {
		t.Errorf("TilesRemaining was %v, expected 0", bag.TilesRemaining())
	}
}

func TestExchange(t *testing.T) {
	is := is.New(t)

	ld, err := EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	bag := ld.MakeBag()
	ml := make([]MachineLetter, 7)
	err = bag.Draw(7, ml)
	is.NoErr(err)
	newML := make([]MachineLetter, 7)
	err = bag.Exchange(ml[:5], newML)
	is.NoErr(err)
	is.Equal(len(bag.tiles), 93)
}

func TestRemoveTiles(t *testing.T) {
	is := is.New(t)

	ld, err := EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	bag := ld.MakeBag()
	is.Equal(len(bag.tiles), 100)
	toRemove := []MachineLetter{
		10, 15, 25, 5, 4, 21, 5, 12, 22, 7, 23, 15, 9, 1, 9, 16, 7, 6, 5,
		20, 1, 25, 9, 18, 18, 19, 3, 12, 9, 15, 2, 9, 1, 21, 8, 1, 9, 11,
		1, 12, 14, 26, 12, 15, 6, 9, 20, 5, 13, 9, 19, 5, 4, 20, 15, 20,
		2, 1, 14, 5, 20, 15, 5, 18, 21, 7, 22, 0x85, 4, 8, 1, 4, 15, 23,
		5, 9, 14, 17, 21, 5, 19, 20, 5, 24, 5, 3, 18, 13, 15, 1, 14,
	}
	is.Equal(len(toRemove), 91)
	err = bag.RemoveTiles(toRemove)
	is.NoErr(err)
	is.Equal(len(bag.tiles), 9)
}

func TestFixedOrder(t *testing.T) {
	is := is.New(t)

	ld, err := EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	bag := NewBag(ld, ld.tilemapping)
	sort.Slice(bag.tiles, func(i, j int) bool { return bag.tiles[i] > bag.tiles[j] })

	bag.SetFixedOrder(true)
	is.Equal(len(bag.tiles), 100)
	ml := make([]MachineLetter, 17)
	err = bag.Draw(17, ml)
	is.NoErr(err)

	is.Equal(ml, []MachineLetter{4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0})
}
