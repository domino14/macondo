package testhelpers

import (
	"github.com/domino14/macondo/config"
	"github.com/domino14/word-golib/tilemapping"
)

var DefaultConfig = config.DefaultConfig()

func EnglishAlphabet() *tilemapping.TileMapping {
	ld, err := tilemapping.GetDistribution(DefaultConfig.WGLConfig(), "english")
	if err != nil {
		panic(err)
	}
	return ld.TileMapping()
}
