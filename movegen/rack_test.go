package movegen

import (
	"testing"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/lexicon"
)

func TestScoreOn(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	alph := gd.GetAlphabet()
	dist := lexicon.EnglishLetterDistribution()
	bag := dist.MakeBag(gd.GetAlphabet(), true)
	type racktest struct {
		rack string
		pts  int
	}
	testCases := []racktest{
		{"ABCDEFG", 16},
		{"XYZ", 22},
		{"??", 0},
		{"?QWERTY", 21},
		{"RETINAO", 7},
	}
	for _, tc := range testCases {
		r := RackFromString(tc.rack, alph)
		score := r.ScoreOn(int(alph.NumLetters()), bag)
		if score != tc.pts {
			t.Errorf("For %v, expected %v, got %v", tc.rack, tc.pts, score)
		}
	}
}

// func TestRackInitialize(t *testing.T) {
// 	rack := &Rack{}
// 	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
// 	rack.Initialize("AENPPSW", gd.GetAlphabet())

// 	if !reflect.DeepEqual(rack.uniqueLetters, map[alphabet.MachineLetter]uint8{
// 		alphabet.MachineLetter(0):  1,
// 		alphabet.MachineLetter(4):  1,
// 		alphabet.MachineLetter(13): 1,
// 		alphabet.MachineLetter(15): 2,
// 		alphabet.MachineLetter(18): 1,
// 		alphabet.MachineLetter(22): 1,
// 	}) {
// 		t.Errorf("Unique letters did not equal")
// 	}
// }

// func TestRackTake(t *testing.T) {
// 	rack := &Rack{}
// 	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
// 	rack.Initialize("AENPPSW", gd.GetAlphabet())
// 	rack.take(alphabet.MachineLetter(15))
// 	if !reflect.DeepEqual(rack.uniqueLetters, map[alphabet.MachineLetter]uint8{
// 		alphabet.MachineLetter(0):  1,
// 		alphabet.MachineLetter(4):  1,
// 		alphabet.MachineLetter(13): 1,
// 		alphabet.MachineLetter(15): 1,
// 		alphabet.MachineLetter(18): 1,
// 		alphabet.MachineLetter(22): 1,
// 	}) {
// 		t.Errorf("Unique letters did not equal (1)")
// 	}
// 	rack.take(alphabet.MachineLetter(15))
// 	if !reflect.DeepEqual(rack.uniqueLetters, map[alphabet.MachineLetter]uint8{
// 		alphabet.MachineLetter(0):  1,
// 		alphabet.MachineLetter(4):  1,
// 		alphabet.MachineLetter(13): 1,
// 		alphabet.MachineLetter(18): 1,
// 		alphabet.MachineLetter(22): 1,
// 	}) {
// 		t.Errorf("Unique letters did not equal (2)")
// 	}
// }

// func TestRackTakeAll(t *testing.T) {
// 	rack := &Rack{}
// 	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
// 	rack.Initialize("AENPPSW", gd.GetAlphabet())
// 	rack.take(alphabet.MachineLetter(15))
// 	rack.take(alphabet.MachineLetter(15))
// 	rack.take(alphabet.MachineLetter(0))
// 	rack.take(alphabet.MachineLetter(4))
// 	rack.take(alphabet.MachineLetter(13))
// 	rack.take(alphabet.MachineLetter(18))
// 	rack.take(alphabet.MachineLetter(22))
// 	if !reflect.DeepEqual(rack.uniqueLetters, map[alphabet.MachineLetter]uint8{}) {
// 		t.Errorf("Map wasn't empty")
// 	}
// }

// func TestRackTakeAndAdd(t *testing.T) {
// 	rack := &Rack{}
// 	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
// 	rack.Initialize("AENPPSW", gd.GetAlphabet())
// 	rack.take(alphabet.MachineLetter(15))
// 	rack.take(alphabet.MachineLetter(15))
// 	rack.take(alphabet.MachineLetter(0))
// 	rack.add(alphabet.MachineLetter(0))
// 	if !reflect.DeepEqual(rack.uniqueLetters, map[alphabet.MachineLetter]uint8{
// 		alphabet.MachineLetter(0):  1,
// 		alphabet.MachineLetter(4):  1,
// 		alphabet.MachineLetter(13): 1,
// 		alphabet.MachineLetter(18): 1,
// 		alphabet.MachineLetter(22): 1,
// 	}) {
// 		t.Errorf("Map didn't match")
// 	}
// }
