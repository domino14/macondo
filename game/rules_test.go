package game

import (
	"testing"

	"github.com/matryer/is"
)

func TestMaxCanExchange(t *testing.T) {
	is := is.New(t)
	for _, tc := range []struct {
		exchlimit int
		inbag     int
		expected  int
	}{
		{7, 84, 7},
		{7, 12, 7},
		{7, 6, 0},
		{7, 7, 7},
		{7, 1, 0},
		{1, 5, 5},
		{1, 2, 2},
		{1, 1, 1},
		{1, 0, 0},
		{1, 47, 7},
		{7, 0, 0},
	} {
		is.Equal(MaxCanExchange(tc.inbag, tc.exchlimit), tc.expected)
	}
}
