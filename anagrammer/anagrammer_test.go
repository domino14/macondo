package anagrammer

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
)

var LexiconDir = os.Getenv("LEXICON_DIR")

type testpair struct {
	rack string
	num  int
}

var buildTests = []testpair{
	{"aehilort", 275},
	{"CINEMATOGRAPHER", 3142},
	{"AEINRST", 276},
	{"KERYGMA", 92},
	{"LOCOFOCO", 16},
	{"VIVIFIC", 2},
	{"ZYZZYVA", 6},
	{"HHHHHHHH", 0},
	{"OCTOROON", 36},
	{"FIREFANG????", 56184},
	{"AEINST??", 9650},
	{"ZZZZ?", 4},
	{"???", 1186},
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
	{"AEINST??", 264},
	{"ZZZZ?", 0},
	{"???", 1081},
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

type wordtestpair struct {
	rack    string
	answers map[string]struct{}
}

var simpleAnagramTests = []wordtestpair{
	{"AEHILORT", wordlistToSet([]string{"AEROLITH"})},
	{"ADEEMMO?", wordlistToSet([]string{"HOMEMADE", "GAMODEME"})},
	// {"X?", wordlistToSet([]string{"AX", "EX", "XI", "OX", "XU"})},
	{"UX", wordlistToSet([]string{"XU"})},
}

func wordlistToSet(wl []string) map[string]struct{} {
	m := make(map[string]struct{})
	for _, w := range wl {
		m[w] = struct{}{}
	}
	return m
}

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/gen_america.dawg"); os.IsNotExist(err) {
		gaddagmaker.GenerateDawg(filepath.Join(LexiconDir, "America.txt"), true, true)
		os.Rename("out.dawg", "/tmp/gen_america.dawg")
	}
	if _, err := os.Stat("/tmp/gen_fise2.dawg"); os.IsNotExist(err) {
		gaddagmaker.GenerateDawg(filepath.Join(LexiconDir, "FISE2.txt"), true, true)
		os.Rename("out.dawg", "/tmp/gen_fise2.dawg")
	}
	if _, err := os.Stat("/tmp/gen_csw15.dawg"); os.IsNotExist(err) {
		gaddagmaker.GenerateDawg(filepath.Join(LexiconDir, "CSW15.txt"), true, true)
		os.Rename("out.dawg", "/tmp/gen_csw15.dawg")
	}

	os.Exit(m.Run())
}

func TestSimpleAnagram(t *testing.T) {
	gaddagmaker.GenerateDawg("test_files/small.txt", true, true)
	d := gaddag.LoadGaddag("out.dawg")
	for _, pair := range simpleAnagramTests {
		answers := Anagram(pair.rack, d, ModeExact)
		if !reflect.DeepEqual(wordlistToSet(answers), pair.answers) {
			t.Error("For", pair.rack, "expected", pair.answers, "got", answers)
		}
	}
}

func TestAnagram(t *testing.T) {
	d := gaddag.LoadGaddag("/tmp/gen_america.dawg")
	for _, pair := range buildTests {
		answers := Anagram(pair.rack, d, ModeBuild)
		if len(answers) != pair.num {
			t.Error("For", pair.rack, "expected", pair.num, "got", len(answers), answers)
		}
	}
	for _, pair := range exactTests {
		answers := Anagram(pair.rack, d, ModeExact)
		if len(answers) != pair.num {
			t.Error("For", pair.rack, "expected", pair.num, "got", len(answers), answers)
		}
	}

}

func TestAnagramSpanish(t *testing.T) {
	d := gaddag.LoadGaddag("/tmp/gen_fise2.dawg")
	for _, pair := range spanishBuildTests {
		answers := Anagram(pair.rack, d, ModeBuild)
		if len(answers) != pair.num {
			t.Error("For", pair.rack, "expected", pair.num, "got", len(answers))
		}
	}
	for _, pair := range spanishExactTests {
		answers := Anagram(pair.rack, d, ModeExact)
		if len(answers) != pair.num {
			t.Error("For", pair.rack, "expected", pair.num, "got", len(answers))
		}
	}
}

func BenchmarkAnagramBlanks(b *testing.B) {
	// ~ 21.33 ms per op on my macbook pro.
	d := gaddag.LoadGaddag("/tmp/gen_csw15.dawg")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Anagram("RETINA??", d, ModeExact)
	}
}

func BenchmarkAnagramFourBlanks(b *testing.B) {
	// ~ 453.6ms
	d := gaddag.LoadGaddag("/tmp/gen_america.dawg")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Anagram("AEINST????", d, ModeExact)
	}
}

func TestBuildFourBlanks(t *testing.T) {
	d := gaddag.LoadGaddag("/tmp/gen_america.dawg")
	answers := Anagram("AEINST????", d, ModeBuild)
	expected := 61711
	if len(answers) != expected {
		t.Errorf("Expected %v answers, got %v", expected, len(answers))
	}
}

func TestAnagramFourBlanks(t *testing.T) {
	d := gaddag.LoadGaddag("/tmp/gen_america.dawg")
	answers := Anagram("AEINST????", d, ModeExact)
	expected := 863
	if len(answers) != expected {
		t.Errorf("Expected %v answers, got %v", expected, len(answers))
	}
}
