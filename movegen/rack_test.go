package movegen

import (
	"reflect"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
)

func TestRackInitialize(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("AENPPSW", gd.GetAlphabet())

	if !reflect.DeepEqual(rack.uniqueLetters, map[alphabet.MachineLetter]uint8{
		alphabet.MachineLetter(0):  1,
		alphabet.MachineLetter(4):  1,
		alphabet.MachineLetter(13): 1,
		alphabet.MachineLetter(15): 2,
		alphabet.MachineLetter(18): 1,
		alphabet.MachineLetter(22): 1,
	}) {
		t.Errorf("Unique letters did not equal")
	}
}

func TestRackTake(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("AENPPSW", gd.GetAlphabet())
	rack.take(alphabet.MachineLetter(15))
	if !reflect.DeepEqual(rack.uniqueLetters, map[alphabet.MachineLetter]uint8{
		alphabet.MachineLetter(0):  1,
		alphabet.MachineLetter(4):  1,
		alphabet.MachineLetter(13): 1,
		alphabet.MachineLetter(15): 1,
		alphabet.MachineLetter(18): 1,
		alphabet.MachineLetter(22): 1,
	}) {
		t.Errorf("Unique letters did not equal (1)")
	}
	rack.take(alphabet.MachineLetter(15))
	if !reflect.DeepEqual(rack.uniqueLetters, map[alphabet.MachineLetter]uint8{
		alphabet.MachineLetter(0):  1,
		alphabet.MachineLetter(4):  1,
		alphabet.MachineLetter(13): 1,
		alphabet.MachineLetter(18): 1,
		alphabet.MachineLetter(22): 1,
	}) {
		t.Errorf("Unique letters did not equal (2)")
	}
}

func TestRackTakeAll(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("AENPPSW", gd.GetAlphabet())
	rack.take(alphabet.MachineLetter(15))
	rack.take(alphabet.MachineLetter(15))
	rack.take(alphabet.MachineLetter(0))
	rack.take(alphabet.MachineLetter(4))
	rack.take(alphabet.MachineLetter(13))
	rack.take(alphabet.MachineLetter(18))
	rack.take(alphabet.MachineLetter(22))
	if !reflect.DeepEqual(rack.uniqueLetters, map[alphabet.MachineLetter]uint8{}) {
		t.Errorf("Map wasn't empty")
	}
}

func TestRackTakeAndAdd(t *testing.T) {
	rack := &Rack{}
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	rack.Initialize("AENPPSW", gd.GetAlphabet())
	rack.take(alphabet.MachineLetter(15))
	rack.take(alphabet.MachineLetter(15))
	rack.take(alphabet.MachineLetter(0))
	rack.add(alphabet.MachineLetter(0))
	if !reflect.DeepEqual(rack.uniqueLetters, map[alphabet.MachineLetter]uint8{
		alphabet.MachineLetter(0):  1,
		alphabet.MachineLetter(4):  1,
		alphabet.MachineLetter(13): 1,
		alphabet.MachineLetter(18): 1,
		alphabet.MachineLetter(22): 1,
	}) {
		t.Errorf("Map didn't match")
	}
}
