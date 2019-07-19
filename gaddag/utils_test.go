package gaddag

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/gaddagmaker"
)

var LexiconDir = os.Getenv("LEXICON_PATH")

type testpair struct {
	prefix string
	found  bool
}

// OWL3 tests
var findPartialWordTests = []testpair{
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
	{"ABC", true},
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
	{"CASITA", true},
	{"ÑU", true},
	{"ÑUBLADO", true},
	{"PARIR", true},
	{"2AMA", true},
	{"2AMAS", true},
	{"2AMATIUJ", false},
}

func TestMain(m *testing.M) {
	gaddagmaker.GenerateDawg(filepath.Join(LexiconDir, "America.txt"), true, true, false)
	os.Rename("out.dawg", "/tmp/gen_america.dawg")
	// Reversed dawg:
	gaddagmaker.GenerateDawg(filepath.Join(LexiconDir, "America.txt"), true, true, true)
	os.Rename("out.dawg", "/tmp/gen_america_r.dawg")

	gaddagmaker.GenerateDawg(filepath.Join(LexiconDir, "FISE2.txt"), true, true, false)
	os.Rename("out.dawg", "/tmp/gen_fise2.dawg")
	os.Exit(m.Run())

}

func TestFindPartialWordMinimize(t *testing.T) {
	d, _ := LoadDawg("/tmp/gen_america.dawg")
	for _, pair := range findPartialWordTests {
		found := findPartialWord(d, d.GetRootNodeIndex(), []rune(pair.prefix), 0)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}

func TestFindWordMinimize(t *testing.T) {
	d, _ := LoadDawg("/tmp/gen_america.dawg")
	for _, pair := range findWordTests {
		found := FindWord(d, pair.prefix)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}

func TestFindSpanishWordMinimize(t *testing.T) {
	d, _ := LoadDawg("/tmp/gen_fise2.dawg")
	for _, pair := range findSpanishWordTests {
		found := FindWord(d, pair.prefix)
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
	d, _ := LoadDawg("/tmp/gen_america.dawg")
	for _, pair := range backHookTests {
		hooks := string(FindHooks(d, pair.word, BackHooks))
		if hooks != pair.hooks {
			t.Error("For", pair.word, "expected", pair.hooks, "found", hooks)
		}
	}
	for _, pair := range frontHookTests {
		rd, _ := LoadDawg("/tmp/gen_america_r.dawg")
		hooks := string(FindHooks(rd, pair.word, FrontHooks))
		if hooks != pair.hooks {
			t.Error("For", pair.word, "expected", pair.hooks, "found", hooks)
		}
	}
	for _, test := range innerHookTests {
		frontInner := FindInnerHook(d, test.word, FrontInnerHook)
		backInner := FindInnerHook(d, test.word, BackInnerHook)
		if frontInner != test.front || backInner != test.back {
			t.Error("For", test.word, "expected", test.front, test.back, "got",
				frontInner, backInner)
		}
	}
}

func TestFindWordSmallSpanish(t *testing.T) {
	gaddagmaker.GenerateDawg("../gaddagmaker/test_files/little_spanish.txt", false, true, false)
	os.Rename("out.dawg", "/tmp/little_spanish.dawg")

	d, _ := LoadDawg("/tmp/little_spanish.dawg")

	for _, word := range []string{"AÑO", "COMER", "COMIDA", "COMIDAS",
		"CO3AL"} {
		found := FindWord(d, word)
		if !found {
			t.Errorf("Did not find word %v :(", word)
		}
	}

}

func TestFindWordSmallSpanish2(t *testing.T) {
	gaddagmaker.GenerateDawg("../gaddagmaker/test_files/ñu.txt", false, true, false)
	os.Rename("out.dawg", "/tmp/ñu.dawg")
	d, _ := LoadDawg("/tmp/ñu.dawg")

	for _, word := range []string{"ÑU", "ÑUS", "ÑAME"} {
		found := FindWord(d, word)
		if !found {
			t.Errorf("Did not find word %v :(", word)
		}
	}

}

func TestFindWordSmallEnglish(t *testing.T) {
	gaddagmaker.GenerateDawg("../gaddagmaker/test_files/dogs.txt", false, true, false)
	os.Rename("out.dawg", "/tmp/dogs.dawg")
	d, _ := LoadDawg("/tmp/dogs.dawg")

	found := FindWord(d, "DOG")
	if !found {
		t.Error("Did not find DOG :(")
	}
}

func TestFindWordSmallEnglish2(t *testing.T) {
	gaddagmaker.GenerateDawg("../gaddagmaker/test_files/no.txt", false, true, false)
	os.Rename("out.dawg", "/tmp/no.dawg")
	d, _ := LoadDawg("/tmp/no.dawg")

	found := FindWord(d, "NO")
	if !found {
		t.Error("Did not find NO :(")
	}
	found = FindWord(d, "ON")
	if found {
		t.Error("Found ON :(")
	}
}

func TestFindPrefixSmallEnglish2(t *testing.T) {
	gaddagmaker.GenerateDawg("../gaddagmaker/test_files/no.txt", false, true, false)
	os.Rename("out.dawg", "/tmp/no.dawg")
	d, _ := LoadDawg("/tmp/no.dawg")

	found := findPartialWord(d, d.GetRootNodeIndex(), []rune("O"), 0)
	if found {
		t.Error("Found O :(")
	}
	found = findPartialWord(d, d.GetRootNodeIndex(), []rune("N"), 0)
	if !found {
		t.Error("!Found N :(")
	}
	found = findPartialWord(d, d.GetRootNodeIndex(), []rune("ON"), 0)
	if found {
		t.Error("Found ON :(")
	}
	found = findPartialWord(d, d.GetRootNodeIndex(), []rune("NO"), 0)
	if !found {
		t.Error("!Found NO :(")
	}
}

func TestFindPrefixSmallEnglish(t *testing.T) {
	gaddagmaker.GenerateDawg("../gaddagmaker/test_files/dogs.txt", false, true, false)
	os.Rename("out.dawg", "/tmp/dogs.dawg")
	d, _ := LoadDawg("/tmp/dogs.dawg")

	found := findPartialWord(d, d.GetRootNodeIndex(), []rune("OG"), 0)
	if found {
		t.Error("Found OG :(")
	}
}
