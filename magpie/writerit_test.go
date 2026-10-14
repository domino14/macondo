package magpie_test

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/magpie"
)

// TestWriteRIT builds a rack info table when MAGPIE_WRITE_RIT names an output
// file (it takes minutes and ~1.7 GB for English).
func TestWriteRIT(t *testing.T) {
	out := os.Getenv("MAGPIE_WRITE_RIT")
	if out == "" {
		t.Skip("set MAGPIE_WRITE_RIT to an output path")
	}
	dataPath := defaultConfig.GetString(config.ConfigDataPath)
	ld, err := tilemapping.GetDistribution(defaultConfig.WGLConfig(), "english")
	if err != nil {
		t.Fatal(err)
	}
	d, err := magpie.NewData(magpie.DataOptions{
		KWGPath:  filepath.Join(dataPath, "lexica", "gaddag", "NWL23.kwg"),
		KLVPath:  filepath.Join(dataPath, "lexica", "gaddag", "NWL23.klv2"),
		BuildWMP: true, Dist: ld, Layout: board.CrosswordGameBoard,
		Variant: magpie.VariantClassic, BingoBonus: 50})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	start := time.Now()
	if err := d.WriteRIT(out, 0); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s in %s", out, time.Since(start))
}
