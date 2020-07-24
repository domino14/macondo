package game

import (
	"fmt"
	"testing"

	"github.com/matryer/is"
)

func TestRandomFirst(t *testing.T) {
	is := is.New(t)
	counts := [2]int{0, 0}
	for i := 0; i < 100000; i++ {
		_, source := seededRandSource()
		selection := source.Bound(2)
		counts[selection]++
	}
	fmt.Println(counts)
	// This test should pass 97% of the time.
	is.True(counts[0] > 49700)
	is.True(counts[0] < 50300)
}
