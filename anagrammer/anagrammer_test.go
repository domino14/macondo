package anagrammer

import "testing"
import "github.com/domino14/macondo/gaddag"

type testpair struct {
	rack string
	num  int
}

var buildTests = []testpair{
	{"AEHILORT", 269},
	{"CINEMATOGRAPHER", 3015},
	{"KERYGMA", 88},
	{"LOCOFOCO", 14},
	{"VIVIFIC", 2},
	{"ZYZZYVA", 6},
	{"HHHHHHHH", 0},
	{"OCTOROON", 34},
	// {"FIREFANG????", 53618},
	// {"AEINST??", 9246},
	// {"ZZZZ?", 3},
	// {"???", 1116},
}

func TestAnagram(t *testing.T) {
	gaddag.GenerateDawg("/Users/cesar/coding/webolith/words/OWL2.txt", true,
		true)
	d := gaddag.SimpleDawg(gaddag.LoadGaddag("out.dawg"))
	for _, pair := range buildTests {
		answers := Anagram(pair.rack, d)
		if len(answers) != pair.num {
			t.Error("For", pair.rack, "expected", pair.num, "got", answers)
		}
	}
}
