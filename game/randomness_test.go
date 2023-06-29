package game

import (
	"fmt"
	"testing"

	"github.com/matryer/is"
	"lukechampine.com/frand"
)

func TestRandomFirst(t *testing.T) {
	is := is.New(t)
	counts := [2]int{0, 0}
	for i := 0; i < 100000; i++ {
		selection := frand.Intn(2)
		counts[selection]++
	}
	fmt.Println(counts)
	// This test should pass 99.84% of the time.
	is.True(counts[0] > 49500)
	is.True(counts[0] < 50500)
}
