package lexicon

import "testing"
import "reflect"

func TestBag(t *testing.T) {
	ld := EnglishLetterDistribution()
	bag := ld.MakeBag()
	if len(bag.tiles) != ld.numLetters {
		t.Error("Tile bag and letter distribution do not match.")
	}
	tileMap := make(map[rune]uint8)
	numTiles := 0
	for range bag.tiles {
		tile, err := bag.Draw()
		numTiles += 1
		t.Logf("Drew a %v! , %v", string(tile), numTiles)
		if err != nil {
			t.Error("Error drawing from tile bag.")
		}
		tileMap[tile] += 1
	}
	if !reflect.DeepEqual(tileMap, ld.distribution) {
		t.Error("Distribution and tilemap were not identical.")
	}
	_, err := bag.Draw()
	if err == nil {
		t.Error("Should not have been able to draw from an empty bag.")
	}
}
