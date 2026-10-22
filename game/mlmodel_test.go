package game

import (
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
