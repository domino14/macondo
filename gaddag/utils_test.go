package gaddag

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
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

var DefaultConfig = config.DefaultConfig()

func loadDawg(lexName string, reverse bool) (*SimpleDawg, error) {
	fn := lexName
	if reverse {
		fn += "-r"
	}
	fn += ".dawg"

	return LoadDawg(filepath.Join(DefaultConfig.LexiconPath, "dawg", fn))
}

func loadGaddag(lexName string) (*SimpleGaddag, error) {
	return LoadGaddag(filepath.Join(DefaultConfig.LexiconPath, "gaddag", lexName+".gaddag"))
}

func TestMain(m *testing.M) {

	for _, lex := range []string{"America", "FISE2"} {
		dwgpath := filepath.Join(DefaultConfig.LexiconPath, "dawg", lex+".dawg")
		if _, err := os.Stat(dwgpath); os.IsNotExist(err) {
			gaddagmaker.GenerateDawg(filepath.Join(DefaultConfig.LexiconPath, lex+".txt"), true, true, false)
			err = os.Rename("out.dawg", dwgpath)
			if err != nil {
				panic(err)
			}
		}
	}

	// reverse dawg

	for _, lex := range []string{"America", "FISE2"} {
		dwgpath := filepath.Join(DefaultConfig.LexiconPath, "dawg", lex+"-r.dawg")
		if _, err := os.Stat(dwgpath); os.IsNotExist(err) {
			gaddagmaker.GenerateDawg(filepath.Join(DefaultConfig.LexiconPath, lex+".txt"), true, true, true)
			err = os.Rename("out.dawg", dwgpath)
			if err != nil {
				panic(err)
			}
		}
	}

	// gaddag
	for _, lex := range []string{"America", "FISE2"} {
		gaddagpath := filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag")
		if _, err := os.Stat(gaddagpath); os.IsNotExist(err) {
			gaddagmaker.GenerateGaddag(filepath.Join(DefaultConfig.LexiconPath, lex+".txt"), true, true)
			err = os.Rename("out.gaddag", gaddagpath)
			if err != nil {
				panic(err)
			}
		}
	}
	os.Exit(m.Run())
}

func TestFindPartialWordMinimize(t *testing.T) {
	d, _ := loadDawg("America", false)
	for _, pair := range findPartialWordTests {
		found := findPartialWord(d, d.GetRootNodeIndex(), []rune(pair.prefix), 0)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}

func TestFindWordMinimize(t *testing.T) {
	d, _ := loadDawg("America", false)
	for _, pair := range findWordTests {
		found := FindWord(d, pair.prefix)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}

func TestFindMachineWord(t *testing.T) {
	d, _ := loadGaddag("America")
	for _, pair := range findWordTests {
		mw, err := alphabet.ToMachineWord(pair.prefix, d.GetAlphabet())
		if err != nil {
			t.Error("error was not nil for conversion of", pair.prefix)
		}

		found := FindMachineWord(d, mw)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}

func TestFindSpanishWordMinimize(t *testing.T) {
	d, err := loadDawg("FISE2", false)
	if err != nil {
		t.Error("loading spanish dawg")
	}
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
	d, _ := loadDawg("America", false)
	for _, pair := range backHookTests {
		hooks := string(FindHooks(d, pair.word, BackHooks))
		if hooks != pair.hooks {
			t.Error("For", pair.word, "expected", pair.hooks, "found", hooks)
		}
	}
	for _, pair := range frontHookTests {
		rd, _ := loadDawg("America", true)
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
	err := os.Rename("out.dawg", "../_tmp/little_spanish.dawg")
	if err != nil {
		log.Fatal(err)
	}

	d, _ := LoadDawg("../_tmp/little_spanish.dawg")

	for _, word := range []string{"AÑO", "COMER", "COMIDA", "COMIDAS",
		"CO3AL"} {
		found := FindWord(d, word)
		if !found {
			t.Errorf("Did not find word %v :(", word)
		}
	}

	found := FindWord(d, "KOO")
	if found {
		t.Errorf("found weird word")
	}
}

func TestFindWordSmallSpanish2(t *testing.T) {
	gaddagmaker.GenerateDawg("../gaddagmaker/test_files/ñu.txt", false, true, false)
	os.Rename("out.dawg", "../_tmp/ñu.dawg")
	d, _ := LoadDawg("../_tmp/ñu.dawg")

	for _, word := range []string{"ÑU", "ÑUS", "ÑAME"} {
		found := FindWord(d, word)
		if !found {
			t.Errorf("Did not find word %v :(", word)
		}
	}

}

func TestFindWordSmallEnglish(t *testing.T) {
	gaddagmaker.GenerateDawg("../gaddagmaker/test_files/dogs.txt", false, true, false)
	os.Rename("out.dawg", "../_tmp/dogs.dawg")
	d, _ := LoadDawg("../_tmp/dogs.dawg")

	found := FindWord(d, "DOG")
	if !found {
		t.Error("Did not find DOG :(")
	}
}

func TestFindWordSmallEnglish2(t *testing.T) {
	gaddagmaker.GenerateDawg("../gaddagmaker/test_files/no.txt", false, true, false)
	os.Rename("out.dawg", "../_tmp/no.dawg")
	d, _ := LoadDawg("../_tmp/no.dawg")

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
	os.Rename("out.dawg", "../_tmp/no.dawg")
	d, _ := LoadDawg("../_tmp/no.dawg")

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
	os.Rename("out.dawg", "../_tmp/dogs.dawg")
	d, _ := LoadDawg("../_tmp/dogs.dawg")

	found := findPartialWord(d, d.GetRootNodeIndex(), []rune("OG"), 0)
	if found {
		t.Error("Found OG :(")
	}
}

func TestDawgAnagrammer(t *testing.T) {
	d, err := loadDawg("America", false)
	if err != nil {
		t.Error("loading America dawg")
		return
	}
	alph := d.GetAlphabet()

	da := DawgAnagrammer{}
	if err = da.InitForString(d, "AEeNRT?"); err == nil {
		t.Error(err)
	}

	existErr := errors.New("found match, returning early")
	num := 0
	if err = da.InitForString(d, "AEENRT?"); err != nil {
		t.Error(err)
	} else if err = da.Anagram(d, func(word alphabet.MachineWord) error {
		num++
		return existErr
	}); err != existErr {
		t.Error(err)
	} else if num != 1 {
		t.Error(fmt.Errorf("expected %v, got %v", 1, num))
	}

	num = 0
	if err = da.InitForString(d, "AEENRT?"); err != nil {
		t.Error(err)
	} else if err = da.Anagram(d, func(word alphabet.MachineWord) error {
		num++
		return nil
	}); err != nil {
		t.Error(err)
	} else if num != 26 {
		t.Error(fmt.Errorf("expected %v, got %v", 26, num))
	}

	if err = da.InitForString(d, "AQQNRT?"); err != nil {
		t.Error(err)
	} else if err = da.Anagram(d, func(word alphabet.MachineWord) error {
		return existErr
	}); err != nil {
		t.Error(err)
	}

	expectedAnags := "[ARENITE CENTARE CRENATE EARNEST EARTHEN EASTERN ENTERAL ENTREAT ETERNAL GRANTEE GREATEN HEARTEN NEAREST NEGATER NERVATE RATTEEN REAGENT REENACT RETAKEN RETINAE STERANE TELERAN TERNATE TERRANE TRAINEE VETERAN]"
	var anags []string
	if mw, err := alphabet.ToMachineWord("AEENRT?", alph); err != nil {
		t.Error(err)
	} else if err = da.InitForMachineWord(d, mw); err != nil {
		t.Error(err)
	} else if err = da.Anagram(d, func(word alphabet.MachineWord) error {
		anags = append(anags, word.UserVisible(alph))
		return nil
	}); err != nil {
		t.Error(err)
	} else if num != len(anags) {
		t.Error(fmt.Errorf("expected %v, got %v %v", num, len(anags), anags))
	} else if fmt.Sprintf("%v", anags) != expectedAnags {
		t.Error(fmt.Errorf("expected %v, got %v", expectedAnags, anags))
	}

	num = 0
	if err = da.InitForString(d, "WOGGL?S"); err != nil {
		t.Error(err)
	} else if err = da.Subanagram(d, func(word alphabet.MachineWord) error {
		num++
		return nil
	}); err != nil {
		t.Error(err)
	} else if num != 384 {
		t.Error(fmt.Errorf("expected %v, got %v", 384, num))
	}

	num = 0
	found := false
	if err = da.InitForString(d, "??W?GGLOS"); err != nil {
		t.Error(err)
	}
	if err = da.Superanagram(d, func(word alphabet.MachineWord) error {
		if word.UserVisible(alph) == "HORNSWOGGLED" {
			found = true
		}
		num++
		return nil
	}); err != nil {
		t.Error(err)
	} else if num != 17 {
		t.Error(fmt.Errorf("expected %v, got %v", 17, num))
	} else if !found {
		t.Error(errors.New("magic word not found in superanagram"))
	}

	if mw, err := alphabet.ToMachineWord("AEENRT?", alph); err != nil {
		t.Error(err)
	} else if _, err := da.IsValidJumble(d, mw); err == nil {
		t.Error(err)
	}

	if mw, err := alphabet.ToMachineWord("AEENRTE", alph); err != nil {
		t.Error(err)
	} else if v, err := da.IsValidJumble(d, mw); err != nil {
		t.Error(err)
	} else if v != false {
		t.Error(fmt.Errorf("expected %v, got %v", false, v))
	}

	if mw, err := alphabet.ToMachineWord("AEENRTT", alph); err != nil {
		t.Error(err)
	} else if v, err := da.IsValidJumble(d, mw); err != nil {
		t.Error(err)
	} else if v != true {
		t.Error(fmt.Errorf("expected %v, got %v", true, v))
	}
}
