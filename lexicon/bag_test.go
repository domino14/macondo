package lexicon

import (
	"os"
	"reflect"
	"testing"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
)

var LexiconDir = os.Getenv("LEXICON_DIR")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/gen_america.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(LexiconDir+"America.txt", true, true)
		os.Rename("out.gaddag", "/tmp/gen_america.gaddag")
	}
	os.Exit(m.Run())
}

func TestBag(t *testing.T) {
	ld := EnglishLetterDistribution()
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	bag := ld.MakeBag(gd.GetAlphabet())
	if len(bag.tiles) != ld.numLetters {
		t.Error("Tile bag and letter distribution do not match.")
	}
	tileMap := make(map[rune]uint8)
	numTiles := 0
	for range bag.tiles {
		tiles, err := bag.Draw(1)
		numTiles++
		t.Logf("Drew a %v! , %v", string(tiles[0]), numTiles)
		if err != nil {
			t.Error("Error drawing from tile bag.")
		}
		tileMap[tiles[0]]++
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
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	bag := ld.MakeBag(gd.GetAlphabet())

	letters, _ := bag.Draw(7)
	if len(letters) != 7 {
		t.Errorf("Length was %v, expected 7", len(letters))
	}
	if len(bag.tiles) != 93 {
		t.Errorf("Length was %v, expected 93", len(bag.tiles))
	}
}

func TestExchange(t *testing.T) {
	ld := EnglishLetterDistribution()
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	bag := ld.MakeBag(gd.GetAlphabet())

	letters, _ := bag.Draw(7)
	newLetters, _ := bag.Exchange(letters[:5])
	if len(newLetters) != 5 {
		t.Errorf("Length was %v, expected 5", len(newLetters))
	}
	if len(bag.tiles) != 93 {
		t.Errorf("Length was %v, expected 93", len(bag.tiles))
	}
}
