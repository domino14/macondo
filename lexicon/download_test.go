package lexicon

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/matryer/is"

	wglconfig "github.com/domino14/word-golib/config"
)

// bucket is a stand-in for the published files. Like the real site, it
// answers a path it has no file for with a page, status 200.
type bucket struct {
	files map[string]string // name -> contents
	// lie makes the sha256 header of a file disagree with its contents.
	lie   map[string]bool
	gets  atomic.Int32
	heads atomic.Int32
}

func sum(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func (b *bucket) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		b.heads.Add(1)
	} else {
		b.gets.Add(1)
	}
	body, ok := b.files[filepath.Base(r.URL.Path)]
	if !ok {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte("<html>the site's page</html>"))
		return
	}
	sha := sum(body)
	if b.lie[filepath.Base(r.URL.Path)] {
		sha = sum("something else")
	}
	w.Header().Set("Content-Type", "binary/octet-stream")
	w.Header().Set(shaHeader, sha)
	w.Write([]byte(body))
}

func setup(t *testing.T, b *bucket) (cfg *wglconfig.Config, dir string) {
	t.Helper()
	srv := httptest.NewServer(b)
	t.Cleanup(srv.Close)
	t.Setenv(LexiconFileURLEnv, srv.URL+"/wasm")
	cfg = &wglconfig.Config{DataPath: t.TempDir()}
	dir = filepath.Join(cfg.DataPath, "lexica", "gaddag")
	return cfg, dir
}

func read(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestEnsureLexiconFile(t *testing.T) {
	is := is.New(t)
	b := &bucket{files: map[string]string{
		"NEW.kwg": "new graph", "SAME.klv2": "same leaves", "STALE.klv2": "new leaves",
		"BAD.kwg": "bad", "BAD2.kwg": "bad2",
	}, lie: map[string]bool{"BAD.kwg": true, "BAD2.kwg": true}}
	cfg, dir := setup(t, b)
	is.NoErr(os.MkdirAll(dir, 0o755))
	is.NoErr(os.WriteFile(filepath.Join(dir, "SAME.klv2"), []byte("same leaves"), 0o644))
	is.NoErr(os.WriteFile(filepath.Join(dir, "STALE.klv2"), []byte("old leaves"), 0o644))
	is.NoErr(os.WriteFile(filepath.Join(dir, "MINE.klv2"), []byte("my leaves"), 0o644))
	is.NoErr(os.WriteFile(filepath.Join(dir, "BAD2.kwg"), []byte("old graph"), 0o644))

	// missing: downloaded
	is.NoErr(EnsureLexiconFile("NEW", ".kwg", cfg))
	is.Equal(read(t, filepath.Join(dir, "NEW.kwg")), "new graph")
	is.Equal(b.gets.Load(), int32(1))

	// on disk and current: not downloaded, and not asked about twice
	is.NoErr(EnsureLexiconFile("SAME", ".klv2", cfg))
	is.Equal(b.gets.Load(), int32(1))
	heads := b.heads.Load()
	is.NoErr(EnsureLexiconFile("SAME", ".klv2", cfg))
	is.NoErr(EnsureLexiconFile("NEW", ".kwg", cfg))
	is.Equal(b.heads.Load(), heads)

	// on disk but not the published file: replaced
	is.NoErr(EnsureLexiconFile("STALE", ".klv2", cfg))
	is.Equal(read(t, filepath.Join(dir, "STALE.klv2")), "new leaves")

	// not published: a local file is left alone, a missing one is reported,
	// and the site's page is never saved as the file
	is.NoErr(EnsureLexiconFile("MINE", ".klv2", cfg))
	is.Equal(read(t, filepath.Join(dir, "MINE.klv2")), "my leaves")
	is.True(errors.Is(EnsureLexiconFile("NONE", ".klv2", cfg), ErrNotPublished))
	_, err := os.Stat(filepath.Join(dir, "NONE.klv2"))
	is.True(os.IsNotExist(err))

	// a download that is not the published file: an error with nothing on
	// disk, and the copy on disk kept when there is one
	err = EnsureLexiconFile("BAD", ".kwg", cfg)
	is.True(err != nil && !errors.Is(err, ErrNotPublished))
	_, err = os.Stat(filepath.Join(dir, "BAD.kwg"))
	is.True(os.IsNotExist(err))
	is.NoErr(EnsureLexiconFile("BAD2", ".kwg", cfg))
	is.Equal(read(t, filepath.Join(dir, "BAD2.kwg")), "old graph")

	// no temporary files are left behind
	entries, err := os.ReadDir(dir)
	is.NoErr(err)
	is.Equal(len(entries), 5)
}

func TestEnsureLexiconFileOffline(t *testing.T) {
	is := is.New(t)
	b := &bucket{}
	cfg, dir := setup(t, b)
	t.Setenv(LexiconFileURLEnv, "http://127.0.0.1:1/") // nothing listens here
	is.NoErr(os.MkdirAll(dir, 0o755))
	is.NoErr(os.WriteFile(filepath.Join(dir, "HAVE.kwg"), []byte("graph"), 0o644))

	is.NoErr(EnsureLexiconFile("HAVE", ".kwg", cfg))
	err := EnsureLexiconFile("LACK", ".kwg", cfg)
	is.True(err != nil && !errors.Is(err, ErrNotPublished))
}

func TestEnsureLexiconFileNoAutoUpdate(t *testing.T) {
	is := is.New(t)
	b := &bucket{files: map[string]string{"STALE.klv2": "new leaves", "NEW.kwg": "new graph"}}
	cfg, dir := setup(t, b)
	t.Setenv(LexiconAutoUpdateEnv, "false")
	is.NoErr(os.MkdirAll(dir, 0o755))
	is.NoErr(os.WriteFile(filepath.Join(dir, "STALE.klv2"), []byte("old leaves"), 0o644))

	is.NoErr(EnsureLexiconFile("STALE", ".klv2", cfg))
	is.Equal(read(t, filepath.Join(dir, "STALE.klv2")), "old leaves")
	is.Equal(b.heads.Load(), int32(0))
	// a missing file is still fetched
	is.NoErr(EnsureLexiconFile("NEW", ".kwg", cfg))
	is.Equal(read(t, filepath.Join(dir, "NEW.kwg")), "new graph")
}
