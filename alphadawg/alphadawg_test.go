package alphadawg

import (
	"os"
	"strings"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/config"
)

var DefaultConfig = config.DefaultConfig()

// loadCSW21 loads the CSW21 alpha dawg, skipping the test if it isn't on
// disk. The expected values below are magpie's (test/kwg_alpha_test.c).
func loadCSW21(t *testing.T) (*kwg.KWG, *tilemapping.TileMapping) {
	t.Helper()
	if os.Getenv("MACONDO_DATA_PATH") == "" {
		t.Skip("MACONDO_DATA_PATH not set")
	}
	k, err := Get(DefaultConfig.WGLConfig(), "CSW21", "english")
	if err != nil {
		t.Skip("CSW21.kad not available: " + err.Error())
	}
	return k, k.GetAlphabet()
}

func tallyFromString(t *testing.T, s string, alph *tilemapping.TileMapping) (Tally, int) {
	t.Helper()
	mw, err := tilemapping.ToMachineWord(s, alph)
	if err != nil {
		t.Fatalf("could not convert %q: %v", s, err)
	}
	var tally Tally
	tally.AddWord(mw)
	return tally, int(alph.NumLetters())
}

func crossSetToString(cs uint64, alph *tilemapping.TileMapping) string {
	var sb strings.Builder
	for ml := 1; ml < int(alph.NumLetters()); ml++ {
		if cs&(1<<uint(ml)) != 0 {
			sb.WriteString(alph.Letter(tilemapping.MachineLetter(ml)))
		}
	}
	return sb.String()
}

func TestAccepts(t *testing.T) {
	is := is.New(t)
	k, alph := loadCSW21(t)

	for _, tc := range []struct {
		word     string
		accepted bool
	}{
		{"A", false}, // single letters are not in the dawg
		{"AB", true},
		{"BA", true}, // any permutation of a word
		{"ZZZ", true},
		{"ANESTRI", true},
		{"OXYPHENBUTAZOEN", true}, // OXYPHENBUTAZONE, jumbled
		{"ENZXONHPOEUYABE", false},
		{"TRONGLE", false},
	} {
		tally, distSize := tallyFromString(t, tc.word, alph)
		is.Equal(Accepts(k, &tally, distSize), tc.accepted)
	}
}

func TestAcceptsWithBlanks(t *testing.T) {
	is := is.New(t)
	k, alph := loadCSW21(t)

	for _, tc := range []struct {
		word     string
		accepted bool
	}{
		{"??", true},
		{"Z??", true},
		{"EARWIG??", true},
		{"TRONGLE?", false},
		{"QQ??", false},
	} {
		tally, distSize := tallyFromString(t, tc.word, alph)
		is.Equal(AcceptsWithBlanks(k, &tally, distSize), tc.accepted)
	}
}

func TestComputeCrossSet(t *testing.T) {
	is := is.New(t)
	k, alph := loadCSW21(t)

	for _, tc := range []struct {
		tiles    string
		expected string
	}{
		{"A", "ABDEFGHIJKLMNPRSTWXYZ"},
		{"AA", "BCFGHIKLMNSVUW"},
		{"C", "H"},
		{"V", ""},
		{"ZZ", "IUZ"},
		{"ZZZ", "IS"},
		{"ZZZZ", ""},
		{"ABHIKSU", "Z"},
		{"ENZXONHPOEUYAB", "T"},
	} {
		tally, distSize := tallyFromString(t, tc.tiles, alph)
		cs := ComputeCrossSet(k, &tally, distSize)
		// The expected sets are magpie's, written in letter order; sort ours
		// the same way for comparison.
		expected := []byte(tc.expected)
		for i := 1; i < len(expected); i++ {
			for j := i; j > 0 && expected[j] < expected[j-1]; j-- {
				expected[j], expected[j-1] = expected[j-1], expected[j]
			}
		}
		is.Equal(crossSetToString(cs, alph), string(expected))
	}
}

func TestLexicon(t *testing.T) {
	is := is.New(t)
	k, alph := loadCSW21(t)
	lex := Lexicon{KWG: k}

	is.Equal(lex.Name(), "CSW21")

	for _, tc := range []struct {
		word  string
		valid bool
	}{
		{"A", false},
		{"BA", true},
		{"AB", true},
		{"AROUND", true},
		{"DUORGNA", true}, // AGROUND
		{"DAEEIMN", true}, // DEMAINE
		{"TRONGLE", false},
		{"AROUNDZ", false},
	} {
		mw, err := tilemapping.ToMachineWord(tc.word, alph)
		is.NoErr(err)
		is.Equal(lex.HasWord(mw), tc.valid)
		is.Equal(lex.HasAnagram(mw), tc.valid)
	}
}
