package anagrammer

import (
	"os"
	"testing"

	"github.com/domino14/macondo/gaddag"
)

var LexiconDir = os.Getenv("LEXICON_DIR")

type testpair struct {
	rack string
	num  int
}

var buildTests = []testpair{
	{"aehilort", 269},
	{"CINEMATOGRAPHER", 3015},
	{"AEINRST", 262},
	{"KERYGMA", 88},
	{"LOCOFOCO", 14},
	{"VIVIFIC", 2},
	{"ZYZZYVA", 6},
	{"HHHHHHHH", 0},
	{"OCTOROON", 34},
	{"FIREFANG????", 53618},
	{"AEINST??", 9246},
	{"ZZZZ?", 3},
	{"???", 1116},
}
var exactTests = []testpair{
	{"AEHILORT", 1},
	{"CINEMATOGRAPHER", 1},
	{"KERYGMA", 1},
	{"LOCOFOCO", 1},
	{"VIVIFIC", 1},
	{"ZYZZYVA", 1},
	{"HHHHHHHH", 0},
	{"OCTOROON", 1},
	{"FIREFANG????", 2},
	{"AEINST??", 247},
	{"ZZZZ?", 0},
	{"???", 1015},
}

var spanishBuildTests = []testpair{
	{"AEHILORT", 319},
	{"CINEMATOGRAPHER", 7792},
	{"KERYGMA", 41}, // K is not in spanish alphabet though
	{"LOCOFOCO", 14},
	{"VIVIFIC", 3},
	{"123?????", 21943},
	{"ÑUBLADO", 64},
	{"CA1AÑEA", 30},
	{"WKWKKWKWWK", 0},
}

var spanishExactTests = []testpair{
	{"AEHILORT", 0},
	{"CINEMATOGRAPHER", 0},
	{"KERYGMA", 0}, // K is not in spanish alphabet though
	{"LOCOFOCO", 0},
	{"ACENORS", 26}, //!
	{"VIVIFIC", 0},
	{"123?????", 3},
	{"ÑUBLADO", 1},
	{"CA1AÑEA", 1},
	{"CA1AÑEA?", 4},
	{"WKWKWWKWKWKW", 0},
}

func TestAnagram(t *testing.T) {
	gaddag.GenerateDawg(LexiconDir+"OWL2.txt", true,
		true)
	d := gaddag.SimpleDawg(gaddag.LoadGaddag("out.dawg"))
	for _, pair := range buildTests {
		answers := Anagram(pair.rack, d, ModeBuild)
		if len(answers) != pair.num {
			t.Error("For", pair.rack, "expected", pair.num, "got", answers)
		}
	}
	for _, pair := range exactTests {
		answers := Anagram(pair.rack, d, ModeExact)
		if len(answers) != pair.num {
			t.Error("For", pair.rack, "expected", pair.num, "got", answers)
		}
	}

}

func TestAnagramSpanish(t *testing.T) {
	gaddag.GenerateDawg(LexiconDir+"FISE09.txt", true,
		true)
	d := gaddag.SimpleDawg(gaddag.LoadGaddag("out.dawg"))
	for _, pair := range spanishBuildTests {
		answers := Anagram(pair.rack, d, ModeBuild)
		if len(answers) != pair.num {
			t.Error("For", pair.rack, "expected", pair.num, "got", answers)
		}
	}
	for _, pair := range spanishExactTests {
		answers := Anagram(pair.rack, d, ModeExact)
		if len(answers) != pair.num {
			t.Error("For", pair.rack, "expected", pair.num, "got", answers)
		}
	}
}

func BenchmarkAnagramBlanks(b *testing.B) {
	// ~ 59 ms per op on my macbook pro.
	for i := 0; i < b.N; i++ {
		Anagram("RETINA??", Dawgs["CSW15"], ModeExact)
	}
}
