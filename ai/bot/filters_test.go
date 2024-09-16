package bot

import (
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/config"
	"github.com/domino14/word-golib/tilemapping"
)

func TestCombinations(t *testing.T) {
	is := is.New(t)
	cfg := config.DefaultConfig()
	ld, err := tilemapping.EnglishLetterDistribution(cfg.WGLConfig())
	is.NoErr(err)

	scc := countSubCombos(ld)
	cmbs := combinations(ld, scc, []tilemapping.MachineLetter{1, 5, 8, 10}, true)
	is.Equal(cmbs, uint64(1121))
}
