package cgp

import (
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/testhelpers"
	"github.com/domino14/word-golib/tilemapping"
)

var DefaultConfig = config.DefaultConfig()

func TestRowToLetters(t *testing.T) {
	is := is.New(t)
	testcases := []struct {
		row    string
		parsed []tilemapping.MachineLetter
	}{
		{"15", []tilemapping.MachineLetter{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		{"10AB3", []tilemapping.MachineLetter{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0}},
		{"A3B10", []tilemapping.MachineLetter{1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		{"1A1B2C3D4", []tilemapping.MachineLetter{0, 1, 0, 2, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0}},
	}
	for _, tc := range testcases {
		parsed, err := rowToLetters(tc.row, testhelpers.EnglishAlphabet())
		is.NoErr(err)
		is.Equal(parsed, tc.parsed)
	}
}

func TestRowToLettersMultichar(t *testing.T) {
	is := is.New(t)
	catalan, err := tilemapping.NamedLetterDistribution(DefaultConfig.WGLConfig(), "catalan")
	is.NoErr(err)
	testcases := []struct {
		row    string
		parsed []tilemapping.MachineLetter
	}{
		{"15", []tilemapping.MachineLetter{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		{"10AB3", []tilemapping.MachineLetter{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0}},
		{"A3b10", []tilemapping.MachineLetter{1, 0, 0, 0, 2 | 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		{"1A1B2C3D4", []tilemapping.MachineLetter{0, 1, 0, 2, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0}},
		{"[qu]3[L·L]5MA3", []tilemapping.MachineLetter{19 | 0x80, 0, 0, 0, 13, 0, 0, 0, 0, 0, 14, 1, 0, 0, 0}},
		{"[qu]3[L·L]5MA2Ç", []tilemapping.MachineLetter{19 | 0x80, 0, 0, 0, 13, 0, 0, 0, 0, 0, 14, 1, 0, 0, 4}},
		{"[QU]3[l·l]9[NY]", []tilemapping.MachineLetter{19, 0, 0, 0, 13 | 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16}},
	}
	for _, tc := range testcases {
		parsed, err := rowToLetters(tc.row, catalan.TileMapping())
		is.NoErr(err)
		is.Equal(parsed, tc.parsed)
	}
}
