package preendgame

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
)

func TestPermuteLeaves(t *testing.T) {
	is := is.New(t)
	letters := []tilemapping.MachineLetter{
		1, 1, 3, 5, 7, 8, 8, 10,
	}
	ret := permuteLeaves(letters, 1)
	is.Equal(ret, [][]tilemapping.MachineLetter{{1}, {1}, {3}, {5}, {7}, {8}, {8}, {10}})
}

func TestPermuteLeaves2(t *testing.T) {
	is := is.New(t)
	letters := []tilemapping.MachineLetter{
		1, 3, 5, 5,
	}
	ret := permuteLeaves(letters, 2)
	is.Equal(ret, [][]tilemapping.MachineLetter{
		{1, 3}, {3, 1}, {1, 5}, {5, 1}, {1, 5}, {5, 1},
		{3, 5}, {5, 3}, {3, 5}, {5, 3}, {5, 5}, {5, 5}})
}
