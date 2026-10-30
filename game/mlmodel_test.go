package game

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/config"
)

// The model for a lexicon: <base>-<suffix> when it exists, the base for
// English lexica, an error for another alphabet without one.
func TestResolveMLModel(t *testing.T) {
	is := is.New(t)
	ld := func(name string) *tilemapping.LetterDistribution {
		d, err := tilemapping.GetDistribution(DefaultConfig.WGLConfig(), name)
		is.NoErr(err)
		return d
	}
	english, german, french, spanish := ld("english"), ld("german"), ld("french"), ld("spanish")

	// A data directory whose models are macondo-tf, macondo-tf-csw and
	// macondo-tf-german (ONNX files, as the CPU path looks for them).
	dir := t.TempDir()
	for _, m := range []string{"macondo-tf", "macondo-tf-csw", "macondo-tf-german"} {
		p := filepath.Join(dir, "strategy", "default", "models", m, "1")
		is.NoErr(os.MkdirAll(p, 0o755))
		is.NoErr(os.WriteFile(filepath.Join(p, "model.onnx"), []byte("x"), 0o644))
	}
	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDataPath, dir)
	cfg.Set(config.ConfigTritonUseTriton, false)
	cfg.Set(config.ConfigMLModelsURL, "") // only what is on disk

	for _, c := range []struct {
		base, lexicon string
		ld            *tilemapping.LetterDistribution
		want          string // "" = an error
	}{
		{"macondo-tf", "NWL23", english, "macondo-tf"},
		{"macondo-tf", "CSW24", english, "macondo-tf-csw"},
		{"macondo-tf", "CSW21", english, "macondo-tf-csw"},
		{"macondo-tf-selfplay", "CSW24", english, "macondo-tf-selfplay"}, // no -csw variant: the base
		{"macondo-tf", "RD29", german, "macondo-tf-german"},
		{"macondo-tf", "FRA24", french, ""}, // no French model
		{"my-net-french", "FRA24", french, "my-net-french"},
		{"macondo-tf", "FILE2017", spanish, ""}, // no Spanish model
	} {
		got, err := ResolveMLModel(cfg, c.base, c.lexicon, c.ld)
		if c.want == "" {
			is.True(err != nil)
			continue
		}
		is.NoErr(err)
		is.Equal(got, c.want)
	}
}

// A missing model is downloaded once, downloaded again when ml-models-url
// changes, and never replaced when it was put there by hand.
func TestEnsureMLModel(t *testing.T) {
	is := is.New(t)
	var gets []string
	content := map[string]string{"/r1/macondo-tf/1/model.onnx": "r1", "/r2/macondo-tf/1/model.onnx": "r2"}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gets = append(gets, r.URL.Path)
		if c, ok := content[r.URL.Path]; ok {
			w.Write([]byte(c))
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	dir := t.TempDir()
	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDataPath, dir)
	models := filepath.Join(dir, "strategy", "default", "models")
	read := func(name string) string {
		b, err := os.ReadFile(filepath.Join(models, name, "1", "model.onnx"))
		is.NoErr(err)
		return string(b)
	}

	cfg.Set(config.ConfigMLModelsURL, srv.URL+"/r1")
	is.NoErr(ensureMLModel(cfg, "macondo-tf", "1", "NWL23"))
	is.Equal(read("macondo-tf"), "r1")
	is.NoErr(ensureMLModel(cfg, "macondo-tf", "1", "NWL23"))
	is.Equal(len(gets), 1) // already there

	cfg.Set(config.ConfigMLModelsURL, srv.URL+"/r2")
	is.NoErr(ensureMLModel(cfg, "macondo-tf", "1", "NWL23"))
	is.Equal(read("macondo-tf"), "r2") // a new revision
	is.Equal(len(gets), 2)

	err := ensureMLModel(cfg, "macondo-tf-french", "1", "FRA24")
	is.True(errors.Is(err, ErrNoMLModel)) // not published

	// A model trained here has no .source file and is left alone.
	p := filepath.Join(models, "my-net", "1")
	is.NoErr(os.MkdirAll(p, 0o755))
	is.NoErr(os.WriteFile(filepath.Join(p, "model.onnx"), []byte("mine"), 0o644))
	is.NoErr(ensureMLModel(cfg, "my-net", "1", "NWL23"))
	is.Equal(read("my-net"), "mine")

	// With the server gone, an out-of-date copy is still used.
	srv.Close()
	cfg.Set(config.ConfigMLModelsURL, srv.URL+"/r3")
	is.NoErr(ensureMLModel(cfg, "macondo-tf", "1", "NWL23"))
	is.Equal(read("macondo-tf"), "r2")
}
