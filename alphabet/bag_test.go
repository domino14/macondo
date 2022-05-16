package alphabet

import (
	"reflect"
	"testing"

	"github.com/domino14/macondo/config"
	"github.com/matryer/is"
)

var DefaultConfig = config.DefaultConfig()

func TestBag(t *testing.T) {
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bag := ld.MakeBag()
	if len(bag.tiles) != ld.numLetters {
		t.Error("Tile bag and letter distribution do not match.")
	}
	tileMap := make(map[rune]uint8)
	numTiles := 0
	for range bag.tiles {
		tiles, err := bag.Draw(1)
		numTiles++
		uv := tiles[0].UserVisible(ld.Alphabet())
		t.Logf("Drew a %c! , %v", uv, numTiles)
		if err != nil {
			t.Error("Error drawing from tile bag.")
		}
		tileMap[uv]++
	}
	if !reflect.DeepEqual(tileMap, ld.Distribution) {
		t.Error("Distribution and tilemap were not identical.")
	}
	_, err = bag.Draw(1)
	if err == nil {
		t.Error("Should not have been able to draw from an empty bag.")
	}
}

func TestDraw(t *testing.T) {
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bag := ld.MakeBag()

	letters, _ := bag.Draw(7)
	if len(letters) != 7 {
		t.Errorf("Length was %v, expected 7", len(letters))
	}
	if len(bag.tiles) != 93 {
		t.Errorf("Length was %v, expected 93", len(bag.tiles))
	}
}

func TestDrawAtMost(t *testing.T) {
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bag := ld.MakeBag()

	for i := 0; i < 14; i++ {
		letters, _ := bag.Draw(7)
		if len(letters) != 7 {
			t.Errorf("Length was %v, expected 7", len(letters))
		}
	}
	if bag.TilesRemaining() != 2 {
		t.Errorf("TilesRemaining was %v, expected 2", bag.TilesRemaining())
	}
	letters := bag.DrawAtMost(7)
	if len(letters) != 2 {
		t.Errorf("Length was %v, expected 2", len(letters))
	}
	if bag.TilesRemaining() != 0 {
		t.Errorf("TilesRemaining was %v, expected 0", bag.TilesRemaining())
	}
	// Try to draw one more time.
	letters = bag.DrawAtMost(7)
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
	bag := ld.MakeBag()

	letters, _ := bag.Draw(7)
	newLetters, _ := bag.Exchange(letters[:5])
	is.Equal(len(newLetters), 5)
	is.Equal(len(bag.tiles), 93)
}

func TestRemoveTiles(t *testing.T) {
	is := is.New(t)
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	bag := ld.MakeBag()
	is.Equal(len(bag.tiles), 100)
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
	is.Equal(len(bag.tiles), 9)
}
