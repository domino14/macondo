package alphabet

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func defaultEnglishAlphabet() *Alphabet {
	return FromSlice([]uint32{
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
		'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
		'W', 'X', 'Y', 'Z',
	})
}

func TestBag(t *testing.T) {
	ld := EnglishLetterDistribution()
	alph := defaultEnglishAlphabet()
	bag := ld.MakeBag(alph)
	if len(bag.tiles) != ld.numLetters {
		t.Error("Tile bag and letter distribution do not match.")
	}
	tileMap := make(map[rune]uint8)
	numTiles := 0
	for range bag.tiles {
		tiles, err := bag.Draw(1)
		numTiles++
		uv := tiles[0].UserVisible(alph)
		t.Logf("Drew a %c! , %v", uv, numTiles)
		if err != nil {
			t.Error("Error drawing from tile bag.")
		}
		tileMap[uv]++
	}
	if !reflect.DeepEqual(tileMap, ld.Distribution) {
		t.Error("Distribution and tilemap were not identical.")
	}
	_, err := bag.Draw(1)
	if err == nil {
		t.Error("Should not have been able to draw from an empty bag.")
	}
}

func TestDraw(t *testing.T) {
	ld := EnglishLetterDistribution()
	alph := defaultEnglishAlphabet()
	bag := ld.MakeBag(alph)

	letters, _ := bag.Draw(7)
	if len(letters) != 7 {
		t.Errorf("Length was %v, expected 7", len(letters))
	}
	if len(bag.tiles) != 93 {
		t.Errorf("Length was %v, expected 93", len(bag.tiles))
	}
}

func TestDrawAtMost(t *testing.T) {
	ld := EnglishLetterDistribution()
	alph := defaultEnglishAlphabet()
	bag := ld.MakeBag(alph)

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
	ld := EnglishLetterDistribution()
	alph := defaultEnglishAlphabet()
	bag := ld.MakeBag(alph)

	letters, _ := bag.Draw(7)
	newLetters, _ := bag.Exchange(letters[:5])
	assert.Equal(t, 5, len(newLetters))
	assert.Equal(t, 93, len(bag.tiles))

}

func TestRemoveTiles(t *testing.T) {
	ld := EnglishLetterDistribution()
	alph := defaultEnglishAlphabet()
	bag := ld.MakeBag(alph)
	assert.Equal(t, 100, len(bag.tiles))
	toRemove := []MachineLetter{
		9, 14, 24, 4, 3, 20, 4, 11, 21, 6, 22, 14, 8, 0, 8, 15, 6, 5, 4,
		19, 0, 24, 8, 17, 17, 18, 2, 11, 8, 14, 1, 8, 0, 20, 7, 0, 8, 10,
		0, 11, 13, 25, 11, 14, 5, 8, 19, 4, 12, 8, 18, 4, 3, 19, 14, 19,
		1, 0, 13, 4, 19, 14, 4, 17, 20, 6, 21, 104, 3, 7, 0, 3, 14, 22,
		4, 8, 13, 16, 20, 4, 18, 19, 4, 23, 4, 2, 17, 12, 14, 0, 13,
	}
	assert.Equal(t, 91, len(toRemove))
	bag.RemoveTiles(toRemove)
	assert.Equal(t, len(bag.tiles), 9)
}