package dataloaders

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/matryer/is"

	wglconfig "github.com/domino14/word-golib/config"

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

// TestLeavesFileForLexicon pins down which leave file each lexicon actually
// gets, by comparing what the resolver hands back against the file it is
// supposed to have opened. Asserting only that resolution succeeds would let a
// lexicon quietly fall back to a relative's leaves -- which is how a test that
// said "NWL18" ended up calibrated against NWL23's numbers.
func TestLeavesFileForLexicon(t *testing.T) {
	if os.Getenv("MACONDO_DATA_PATH") == "" {
		t.Skip("MACONDO_DATA_PATH not set")
	}
	cfg := DefaultConfig.WGLConfig()
	lexicaDir := LexiconDataPath(cfg)
	strategyDir := StrategyParamsPath(cfg)

	for _, tc := range []struct {
		lexicon   string
		leavefile string
		want      string // the file the resolver must end up reading
		why       string
	}{
		{"NWL23", "", filepath.Join(lexicaDir, "NWL23.klv2"), "its own"},
		{"CSW24", "", filepath.Join(lexicaDir, "CSW24.klv2"), "its own"},
		{"CSW21", "", filepath.Join(lexicaDir, "CSW21.klv2"), "its own"},
		{"NWL18", "", filepath.Join(lexicaDir, "NWL18.klv2"), "its own"},
		{"NWL23", "super-leaves.klv2", filepath.Join(lexicaDir, "super-NWL23.klv2"), "its own super leaves"},
		{"CSW07", "", filepath.Join(lexicaDir, "CSW24.klv2"), "none of its own; borrows CSW24's"},
		{"CSW12", "", filepath.Join(lexicaDir, "CSW24.klv2"), "borrows CSW24's"},
		{"TWL06", "", filepath.Join(lexicaDir, "NWL23.klv2"), "borrows NWL23's"},
		{"CSW21", "leavesv80i.klv2", filepath.Join(strategyDir, "CSW21", "leavesv80i.klv2"),
			"named explicitly, so it comes from the strategy folder"},
	} {
		want, err := os.ReadFile(tc.want)
		if err != nil {
			t.Skipf("%s: %v", tc.want, err)
		}
		f, err := LeavesFileForLexicon(cfg, tc.leavefile, tc.lexicon)
		if err != nil {
			t.Fatalf("%s (%q): %v", tc.lexicon, tc.leavefile, err)
		}
		got, err := io.ReadAll(f)
		f.Close()
		if err != nil {
			t.Fatalf("%s (%q): %v", tc.lexicon, tc.leavefile, err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s (%q) did not read %s (%s)", tc.lexicon, tc.leavefile,
				filepath.Base(tc.want), tc.why)
		}
	}
}

// TestOwnPublishedLeavesBeatBorrowing checks that a lexicon with no leaves on
// disk fetches its own published ones rather than borrowing a relative's that
// happen to be there -- a worker holding CSW24.klv2 must still get CSW15's.
func TestOwnPublishedLeavesBeatBorrowing(t *testing.T) {
	is := is.New(t)
	defer func(orig func(string, *wglconfig.Config) error) { fetchLeaves = orig }(fetchLeaves)

	cfg := &wglconfig.Config{DataPath: t.TempDir()}
	dir := LexiconDataPath(cfg)
	is.NoErr(os.MkdirAll(dir, 0o755))
	is.NoErr(os.WriteFile(filepath.Join(dir, "CSW24.klv2"), []byte("csw24"), 0o644))

	published := map[string]string{"CSW15.klv2": "csw15"}
	fetchLeaves = func(name string, cfg *wglconfig.Config) error {
		body, ok := published[name]
		if !ok {
			return ErrNotPublished
		}
		return os.WriteFile(filepath.Join(LexiconDataPath(cfg), name), []byte(body), 0o644)
	}

	read := func(lexicon string) string {
		f, err := LeavesFileForLexicon(cfg, "", lexicon)
		is.NoErr(err)
		defer f.Close()
		b, err := io.ReadAll(f)
		is.NoErr(err)
		return string(b)
	}
	is.Equal(read("CSW15"), "csw15") // its own, fetched
	is.Equal(read("CSW12"), "csw24") // none published; borrows CSW24's
}
