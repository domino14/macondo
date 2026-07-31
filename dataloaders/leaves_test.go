package dataloaders

import (
	"os"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/config"
)

var DefaultConfig = config.DefaultConfig()

func TestLeavesNameFor(t *testing.T) {
	is := is.New(t)

	name, standard := leavesNameFor("", "NWL23")
	is.Equal(name, "NWL23.klv2")
	is.True(standard)

	name, standard = leavesNameFor("leaves.klv2", "CSW24")
	is.Equal(name, "CSW24.klv2")
	is.True(standard)

	name, standard = leavesNameFor("super-leaves.klv2", "NWL23")
	is.Equal(name, "super-NWL23.klv2")
	is.True(standard)

	// An explicit experiment file is not a lexicon's standard leaves.
	_, standard = leavesNameFor("leavesv80i.klv2", "CSW21")
	is.True(!standard)
}

// TestLeavesFileForLexicon exercises the real data directory: the primary
// lookup, the borrow-another-lexicon's-leaves fallback, and the strategy
// folder fallback for explicitly named files.
func TestLeavesFileForLexicon(t *testing.T) {
	is := is.New(t)
	if os.Getenv("MACONDO_DATA_PATH") == "" {
		t.Skip("MACONDO_DATA_PATH not set")
	}
	cfg := DefaultConfig.WGLConfig()

	for _, tc := range []struct {
		lexicon   string
		leavefile string
	}{
		{"NWL23", ""},                  // its own, in the lexicon data dir
		{"CSW24", ""},                  //
		{"NWL23", "super-leaves.klv2"}, // super variant
		{"CSW21", ""},                  // its own
		{"CSW07", ""},                  // no leaves of its own; borrows CSW24's
		{"CSW12", ""},                  // borrows CSW24's
		{"TWL06", ""},                  // borrows NWL23's
		{"CSW21", "leavesv80i.klv2"},   // explicit file, strategy folder
	} {
		f, err := LeavesFileForLexicon(cfg, tc.leavefile, tc.lexicon)
		if err != nil {
			t.Fatalf("%s (%q): %v", tc.lexicon, tc.leavefile, err)
		}
		is.True(f != nil)
		f.Close()
	}
}
