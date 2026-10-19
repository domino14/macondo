package game

import (
	"reflect"
	"testing"
)

func TestMLBatchRanges(t *testing.T) {
	for _, c := range []struct {
		n, max int
		want   [][2]int
	}{
		{0, 128, nil},
		{50, 128, [][2]int{{0, 50}}},
		{128, 128, [][2]int{{0, 128}}},
		{129, 128, [][2]int{{0, 128}, {128, 129}}},
		{300, 128, [][2]int{{0, 128}, {128, 256}, {256, 300}}},
	} {
		if got := mlBatchRanges(c.n, c.max); !reflect.DeepEqual(got, c.want) {
			t.Errorf("mlBatchRanges(%d, %d) = %v, want %v", c.n, c.max, got, c.want)
		}
	}
}
