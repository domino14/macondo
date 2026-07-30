package wmp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	wglconfig "github.com/domino14/word-golib/config"
)

// tempFilesIn returns the names of any leftover .wmp-tmp-* files in dir.
func tempFilesIn(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir(%s) failed: %v", dir, err)
	}
	var leftovers []string
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".wmp-tmp-") {
			leftovers = append(leftovers, e.Name())
		}
	}
	return leftovers
}

func TestWriteToFileIsReadableAndLeavesNoTempFiles(t *testing.T) {
	words := stringsToMachineWords([]string{
		"CAT", "ACT", "TAB", "BAT", "CARE", "RACE", "ACRE",
	})
	wmp, err := MakeFromWords(words, testEnglishLD(t), testBoardDim, 1)
	if err != nil {
		t.Fatalf("MakeFromWords failed: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "TEST.wmp")
	if err := wmp.WriteToFile(path); err != nil {
		t.Fatalf("WriteToFile failed: %v", err)
	}

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat after WriteToFile failed: %v", err)
	}
	// Other processes (and other users) read these files; the temp file the
	// atomic write goes through starts at 0600 and must not stay that way.
	if perm := fi.Mode().Perm(); perm != 0644 {
		t.Errorf("WMP file mode = %o, want 644", perm)
	}
	if leftovers := tempFilesIn(t, dir); len(leftovers) > 0 {
		t.Errorf("WriteToFile left temp files behind: %v", leftovers)
	}

	loaded, err := LoadFromFile("TEST", path)
	if err != nil {
		t.Fatalf("LoadFromFile after WriteToFile failed: %v", err)
	}
	br := BitRackFromMachineWord(stringsToMachineWords([]string{"CARE"})[0])
	out := make([]byte, ResultBufferSize)
	n := loaded.WriteWordsToBuffer(&br, 4, out)
	got := extractWordsFromBuffer(out[:n], 4)
	want := []string{"ACRE", "CARE", "RACE"}
	if !sliceEqual(got, want) {
		t.Errorf("CARE lookup after WriteToFile: got %v, want %v", got, want)
	}
}

// A failed write must not leave a partial .wmp behind: a truncated file
// would be picked up by every later load (os.Stat succeeds) and fail to
// parse forever, silently disabling the WMP.
func TestWriteToFileToUnwritableDirLeavesNothing(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root; directory permissions would not be enforced")
	}
	words := stringsToMachineWords([]string{"CAT", "ACT", "TAB", "BAT"})
	wmp, err := MakeFromWords(words, testEnglishLD(t), testBoardDim, 1)
	if err != nil {
		t.Fatalf("MakeFromWords failed: %v", err)
	}

	dir := filepath.Join(t.TempDir(), "readonly")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatalf("Mkdir failed: %v", err)
	}
	if err := os.Chmod(dir, 0555); err != nil {
		t.Fatalf("Chmod failed: %v", err)
	}
	t.Cleanup(func() { os.Chmod(dir, 0755) })

	path := filepath.Join(dir, "TEST.wmp")
	if err := wmp.WriteToFile(path); err == nil {
		t.Fatal("WriteToFile to a read-only directory unexpectedly succeeded")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("failed WriteToFile left a file at %s (stat err: %v)", path, err)
	}
	if leftovers := tempFilesIn(t, dir); len(leftovers) > 0 {
		t.Errorf("failed WriteToFile left temp files behind: %v", leftovers)
	}
}

// EnsureWMP must still hand back the WMP it just built when the data
// directory is read-only (e.g. the bot Lambda's EFS mount), and must cache
// it so the build cost is paid once per process rather than once per call.
func TestEnsureWMPUsesBuiltWMPWhenSaveFails(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root; directory permissions would not be enforced")
	}
	dp := os.Getenv("MACONDO_DATA_PATH")
	if dp == "" {
		t.Skip("MACONDO_DATA_PATH not set; skipping WMP build test")
	}
	const lexicon = "CSW24"
	realKWG := filepath.Join(dp, "lexica", "gaddag", lexicon+".kwg")
	if _, err := os.Stat(realKWG); err != nil {
		t.Skipf("%s not found at %s", lexicon, realKWG)
	}

	// A data dir whose lexica/ is read-only, but whose KWG and letter
	// distributions are the real ones.
	root := t.TempDir()
	lexica := filepath.Join(root, "lexica")
	if err := os.MkdirAll(lexica, 0755); err != nil {
		t.Fatalf("MkdirAll failed: %v", err)
	}
	if err := os.Symlink(filepath.Join(dp, "lexica", "gaddag"), filepath.Join(lexica, "gaddag")); err != nil {
		t.Fatalf("Symlink gaddag failed: %v", err)
	}
	if err := os.Symlink(filepath.Join(dp, "letterdistributions"), filepath.Join(root, "letterdistributions")); err != nil {
		t.Fatalf("Symlink letterdistributions failed: %v", err)
	}
	if err := os.Chmod(lexica, 0555); err != nil {
		t.Fatalf("Chmod failed: %v", err)
	}
	t.Cleanup(func() { os.Chmod(lexica, 0755) })

	cfg := &wglconfig.Config{DataPath: root}
	w, err := EnsureWMP(cfg, lexicon)
	if err != nil {
		t.Fatalf("EnsureWMP failed even though the built WMP is usable: %v", err)
	}
	if w == nil {
		t.Fatal("EnsureWMP returned a nil WMP")
	}
	if w.Name != lexicon {
		t.Errorf("built WMP Name = %q, want %q", w.Name, lexicon)
	}
	if _, err := os.Stat(filepath.Join(lexica, lexicon+".wmp")); !os.IsNotExist(err) {
		t.Errorf("expected no .wmp file in a read-only dir (stat err: %v)", err)
	}
	if leftovers := tempFilesIn(t, lexica); len(leftovers) > 0 {
		t.Errorf("failed save left temp files behind: %v", leftovers)
	}

	// Second call must come from the cache rather than rebuilding.
	w2, err := EnsureWMP(cfg, lexicon)
	if err != nil {
		t.Fatalf("second EnsureWMP failed: %v", err)
	}
	if w2 != w {
		t.Error("EnsureWMP rebuilt the WMP instead of returning the cached one")
	}
}
