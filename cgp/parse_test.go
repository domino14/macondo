package cgp

import (
	"testing"

	"github.com/matryer/is"
)

func TestRowToLetters(t *testing.T) {
	is := is.New(t)
	testcases := []struct {
		row    string
		parsed string
	}{
		{"15", "               "},
		{"10AB3", "          AB   "},
		{"A3B10", "A   B          "},
		{"1A1B2C3D4", " A B  C   D    "},
	}
	for _, tc := range testcases {
		parsed, err := rowToLetters(tc.row)
		is.NoErr(err)
		is.Equal(parsed, tc.parsed)
	}
}
