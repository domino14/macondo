package gaddag

import "testing"
import _ "log"

type testpair struct {
	prefix string
	found  bool
}

// OWL2 tests
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
	{"EEE", false},
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

func TestFindPrefixMinimize(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	GenerateGaddag("/Users/cesar/coding/webolith/words/OWL2.txt", true, true)
	gd := LoadGaddag("out.gaddag")
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
	GenerateGaddag("/Users/cesar/coding/webolith/words/OWL2.txt", true, true)
	gd := LoadGaddag("out.gaddag")
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
	GenerateGaddag("/Users/cesar/coding/webolith/words/FISE.txt", true, true)
	gd := LoadGaddag("out.gaddag")
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
	{"HOUSE", "DLRS"},
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
	GenerateGaddag("/Users/cesar/coding/webolith/words/OWL2.txt", true, true)
	gd := LoadGaddag("out.gaddag")
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
	GenerateDawg("/Users/cesar/coding/webolith/words/OWL2.txt", true, true)
	d := LoadGaddag("out.dawg")
	for _, pair := range findWordTests {
		found := FindWordDawg(d, pair.prefix)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}
