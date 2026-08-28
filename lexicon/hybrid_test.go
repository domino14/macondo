package lexicon_test

import (
	"os"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/lexicon"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}

func loadLexicon(t *testing.T, name string) lexicon.Lexicon {
	t.Helper()
	k, err := kwg.GetKWG(DefaultConfig.WGLConfig(), name)
	if err != nil {
		t.Fatalf("loading %s: %v", name, err)
	}
	return kwg.Lexicon{KWG: *k}
}

// TestHybrid pins down the word list a CEL+2s board is built from: everything
// common, plus the two-letter words the full lexicon adds and nothing longer.
func TestHybrid(t *testing.T) {
	is := is.New(t)

	cel := loadLexicon(t, "ECWL")
	nwl := loadLexicon(t, "NWL23")
	h, err := lexicon.NewHybrid(cel, nwl, 2)
	is.NoErr(err)

	mw := func(w string) lexicon.Word {
		m, err := tilemapping.ToMachineWord(w, h.GetAlphabet())
		is.NoErr(err)
		return m
	}

	for _, tc := range []struct {
		word string
		want bool
		why  string
	}{
		{"HOUSE", true, "common word"},
		{"AN", true, "two-letter word in ECWL as well"},
		{"ZA", true, "two-letter word only NWL23 has"},
		{"JO", true, "two-letter word only NWL23 has"},
		{"CWM", false, "three letters, and only NWL23 has it"},
		{"ZUZ", false, "three letters, and only NWL23 has it"},
		{"QINDARKA", false, "long, and only NWL23 has it"},
		{"XQ", false, "two letters but in neither"},
		{"BLARGHY", false, "in neither"},
	} {
		if got := h.HasWord(mw(tc.word)); got != tc.want {
			t.Errorf("HasWord(%s) = %v, want %v (%s)", tc.word, got, tc.want, tc.why)
		}
	}
}

// TestHybridRejectsMismatchedAlphabets guards the one way a Hybrid can be
// silently wrong: two lexicons that number their tiles differently, where a
// MachineWord means something different to each.
func TestHybridRejectsMismatchedAlphabets(t *testing.T) {
	is := is.New(t)

	// Polish numbers a different set of tiles, so a MachineWord built against
	// one of these is meaningless to the other. (French would pass: it uses the
	// same 26 tiles as English, so the two are machine-compatible even though
	// pairing them would be a strange thing to do.)
	_, err := lexicon.NewHybrid(loadLexicon(t, "ECWL"), loadLexicon(t, "OSPS50"), 2)
	is.True(err != nil)

	// Same distribution, different TileMapping instances -- must be accepted.
	_, err = lexicon.NewHybrid(loadLexicon(t, "ECWL"), loadLexicon(t, "NWL23"), 2)
	is.NoErr(err)
}
