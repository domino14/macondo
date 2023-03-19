package tilemapping

import (
	"fmt"
	"testing"

	"github.com/domino14/macondo/config"
	"github.com/matryer/is"
)

var DefaultConfig = config.DefaultConfig()

func TestLetterDistributionScores(t *testing.T) {
	is := is.New(t)
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)

	is.Equal(ld.Score(0), 0)
	is.Equal(ld.Score(0x81), 0)
	is.Equal(ld.Score(25), 4)
	is.Equal(ld.Score(26), 10)
	is.Equal(ld.Score(8), 4)
	is.Equal(ld.Score(1), 1)
}

func TestLetterDistributionWordScore(t *testing.T) {
	is := is.New(t)
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)

	word := "CoOKIE"
	mls, err := ToMachineLetters(word, ld.TileMapping())
	fmt.Println("mls", mls)
	is.NoErr(err)
	is.Equal(ld.WordScore(mls), 11)
}
