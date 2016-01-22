package gaddag

import "testing"

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
