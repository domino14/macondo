package gaddag

import (
	"os"
	"testing"

	"github.com/domino14/macondo/gaddagmaker"
)

var LexiconDir = os.Getenv("LEXICON_DIR")

type testpair struct {
	prefix string
	found  bool
}

// OWL3 tests
var findPrefixTests = []testpair{
	{"ZYZZ", true},
	{"ZYZZZ", false},
	{"BAREFIT", true},
	{"KWASH", true},
	{"KWASHO", false},
	{"BAREFITS", false},
	{"AASVOGELATION", false},
	{"FIREFANGNESS", false},
	{"TRIREMED", false},
	{"BAREFITD", false},
	{"KAFF", true},
	{"FF", false},
	{"EEE", true}, // EEEW!
	{"ABC", true},
	{"ABCD", false},
	{"FIREFANG", true},
	{"X", true},
	{"Z", true},
	{"Q", true},
	{"KWASHIORKORS", true},
	{"EE", true},
	{"RETIARII", true},
	{"CINEMATOGRAPHER", true},
	{"ANIMADVERTS", true},
	{"PRIVATDOZENT", true},
	{"INEMATOGRAPHER", false},
	{"RIIRAITE", false},
	{"GG", false},
	{"LL", true},
	{"ZZ", true},
	{"ZZZ", true},
	{"ZZZS", false},
	{"EOW", false},
}

var findWordTests = []testpair{
	{"ZYZZ", false},
	{"ZYZZZ", false},
	{"BAREFIT", true},
	{"KWASH", false},
	{"KWASHO", false},
	{"BAREFITS", false},
	{"AASVOGELATION", false},
	{"FIREFANGNESS", false},
	{"TRIREMED", false},
	{"BAREFITD", false},
	{"KAFF", false},
	{"FF", false},
	{"ABRACADABRA", true},
	{"EEE", false},
	{"ABC", false},
	{"ABCD", false},
	{"FIREFANG", true},
	{"X", false},
	{"Z", false},
	{"Q", false},
	{"KWASHIORKORS", true},
	{"EE", false},
	{"RETIARII", true},
	{"CINEMATOGRAPHER", true},
	{"ANIMADVERTS", true},
	{"PRIVATDOZENT", true},
	{"INEMATOGRAPHER", false},
	{"RIIRAITE", false},
	{"GG", false},
	{"LL", false},
	{"ZZ", false},
	{"ZZZ", true},
	{"ZZZS", false},
}

var findSpanishWordTests = []testpair{
	{"ZYZZ", false},
	{"ZYZZZ", false},
	{"BAREFIT", false},
	{"KWASH", false},
	{"KWASHO", false},
	{"BAREFITS", false},
	{"AASVOGELATION", false},
	{"FIREFANGNESS", false},
	{"TRIREMED", false},
	{"BAREFITD", false},
	{"KAFF", false},
	{"FF", false},
	{"ABRACADABRA", true},
	{"EEE", false},
	{"ABC", false},
	{"ABCD", false},
	{"FIREFANG", false},
	{"AÑO", true},
	{"X", false},
	{"Z", false},
	{"Q", false},
	{"KWASHIORKORS", false},
	{"EE", false},
	{"RETIARII", false},
	{"CINEMATOGRAPHER", false},
	{"ANIMADVERTS", false},
	{"PRIVATDOZENT", false},
	{"INEMATOGRAPHER", false},
	{"RIIRAITE", false},
	{"GG", false},
	{"LL", false},
	{"ZZ", false},
	{"ZZZ", false},
	{"ZZZS", false},
	{"CASITA", false},
	{"ÑU", true},
	{"ÑUBLADO", true},
	{"PARIR", true},
	{"2AMA", true},
	{"2AMAS", true},
	{"2AMATIUJ", false},
}

func TestMain(m *testing.M) {
	gaddagmaker.GenerateGaddag(LexiconDir+"America.txt", true, true)
	os.Rename("out.gaddag", "/tmp/gen_america.gaddag")
	gaddagmaker.GenerateDawg(LexiconDir+"America.txt", true, true)
	os.Rename("out.dawg", "/tmp/gen_america.dawg")
	gaddagmaker.GenerateGaddag(LexiconDir+"FISE09.txt", true, true)
	os.Rename("out.gaddag", "/tmp/gen_fise09.gaddag")

	os.Exit(m.Run())
}

func TestFindPrefixMinimize(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	gd := LoadGaddag("/tmp/gen_america.gaddag")
	for _, pair := range findPrefixTests {
		found := FindPrefix(gd, pair.prefix)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}

func TestFindWordMinimize(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	gd := LoadGaddag("/tmp/gen_america.gaddag")
	for _, pair := range findWordTests {
		found := FindWord(gd, pair.prefix)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}

func TestFindSpanishWordMinimize(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	gd := LoadGaddag("/tmp/gen_fise09.gaddag")
	for _, pair := range findSpanishWordTests {
		found := FindWord(gd, pair.prefix)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}

type hookpair struct {
	word  string
	hooks string
}

var backHookTests = []hookpair{
	{"HOUSE", "DLRSY"},
	{"HOUSED", ""},
	{"AB", "AOSY"},
	{"BA", "ADGHLMNPRSTY"},
	{"CINEMATOGRAPHER", ""}, // stupid 15s
	{"TELESTIC", "HS"},
	{"CHAO", "S"},
	{"ABASDSDF", ""},
	{"ILL", "SY"},
	{"KITTEN", "S"},
	{"LADYFINGER", "S"},
	{"ABYS", "MS"},
	{"ABYSS", ""},
	{"VOLUME", "DS"},
}

var frontHookTests = []hookpair{
	{"HOUSE", "C"},
	{"FIREFLY", ""},
	{"AB", "CDFGJKLNSTW"},
	{"BA", "AO"},
	{"DOG", ""},
	{"FASFSDFASDAS", ""},
	{"SATIN", "I"},
	{"CONICITY", "I"},
	{"INCITES", "Z"},
	{"ASTRONOMICALLY", "G"},
}

type innerhooktest struct {
	word  string
	front bool
	back  bool
}

var innerHookTests = []innerhooktest{
	{"HOUSE", false, false},
	{"ICONICITY", true, false},
	{"ASTHENOSPHERES", false, true},
	{"FLAMINGO", false, true},
	{"REVOLUTIONARILY", true, false},
	{"EVOLUTIONARILY", false, false},
	{"ABA", true, true},
	{"KURUS", true, true},
	{"KUMYS", false, false},
	{"MELLS", true, true},
	{"MACK", false, true},
}

func TestFindHooks(t *testing.T) {
	gd := LoadGaddag("/tmp/gen_america.gaddag")
	for _, pair := range backHookTests {
		hooks := string(FindHooks(gd, pair.word, BackHooks))
		if hooks != pair.hooks {
			t.Error("For", pair.word, "expected", pair.hooks, "found", hooks)
		}
	}
	for _, pair := range frontHookTests {
		hooks := string(FindHooks(gd, pair.word, FrontHooks))
		if hooks != pair.hooks {
			t.Error("For", pair.word, "expected", pair.hooks, "found", hooks)
		}
	}
	for _, test := range innerHookTests {
		frontInner := FindInnerHook(gd, test.word, FrontInnerHook)
		backInner := FindInnerHook(gd, test.word, BackInnerHook)
		if frontInner != test.front || backInner != test.back {
			t.Error("For", test.word, "expected", test.front, test.back, "got",
				frontInner, backInner)
		}
	}
}

func TestFindWordDawgMinimize(t *testing.T) {
	d := LoadGaddag("/tmp/gen_america.dawg")
	for _, pair := range findWordTests {
		found := FindWordDawg(d, pair.prefix)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}

func TestFindWordSmallSpanish(t *testing.T) {
	gaddagmaker.GenerateGaddag("../gaddagmaker/test_files/little_spanish.txt", false, true)
	os.Rename("out.gaddag", "/tmp/little_spanish.gaddag")

	gd := LoadGaddag("/tmp/little_spanish.gaddag")

	for _, word := range []string{"AÑO", "COMER", "COMIDA", "COMIDAS",
		"CO3AL"} {
		found := FindWord(gd, word)
		if !found {
			t.Errorf("Did not find word %v :(", word)
		}
	}

}

func TestFindWordSmallSpanish2(t *testing.T) {
	gaddagmaker.GenerateGaddag("../gaddagmaker/test_files/ñu.txt", false, true)
	os.Rename("out.gaddag", "/tmp/ñu.gaddag")
	gd := LoadGaddag("/tmp/ñu.gaddag")

	for _, word := range []string{"ÑU", "ÑUS", "ÑAME"} {
		found := FindWord(gd, word)
		if !found {
			t.Errorf("Did not find word %v :(", word)
		}
	}

}

func TestFindWordSmallEnglish(t *testing.T) {
	gaddagmaker.GenerateGaddag("../gaddagmaker/test_files/dogs.txt", false, true)
	os.Rename("out.gaddag", "/tmp/dogs.gaddag")
	gd := LoadGaddag("/tmp/dogs.gaddag")

	found := FindWord(gd, "DOG")
	if !found {
		t.Error("Did not find DOG :(")
	}
}

func TestFindWordSmallEnglish2(t *testing.T) {
	gaddagmaker.GenerateGaddag("../gaddagmaker/test_files/no.txt", false, true)
	os.Rename("out.gaddag", "/tmp/no.gaddag")
	gd := LoadGaddag("/tmp/no.gaddag")

	found := FindWord(gd, "NO")
	if !found {
		t.Error("Did not find NO :(")
	}
	found = FindWord(gd, "ON")
	if found {
		t.Error("Found ON :(")
	}
}

func TestFindPrefixSmallEnglish2(t *testing.T) {
	gaddagmaker.GenerateGaddag("../gaddagmaker/test_files/no.txt", false, true)
	os.Rename("out.gaddag", "/tmp/no.gaddag")
	gd := LoadGaddag("/tmp/no.gaddag")

	found := FindPrefix(gd, "O")
	if found {
		t.Error("Found O :(")
	}
	found = FindPrefix(gd, "N")
	if !found {
		t.Error("!Found N :(")
	}
	found = FindPrefix(gd, "ON")
	if found {
		t.Error("Found ON :(")
	}
	found = FindPrefix(gd, "NO")
	if !found {
		t.Error("!Found NO :(")
	}
}

func TestFindPrefixSmallEnglish(t *testing.T) {
	gaddagmaker.GenerateGaddag("../gaddagmaker/test_files/dogs.txt", false, true)
	os.Rename("out.gaddag", "/tmp/dogs.gaddag")
	gd := LoadGaddag("/tmp/dogs.gaddag")

	found := FindPrefix(gd, "OG")
	if found {
		t.Error("Found OG :(")
	}
}
