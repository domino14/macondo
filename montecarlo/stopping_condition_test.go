package montecarlo

import (
	"fmt"
	"testing"

	"github.com/matryer/is"
)

func TestPassZTest(t *testing.T) {
	is := is.New(t)
	is.Equal(passTest(450, 10000, 460, 10000, Stop95), false)
	is.True(passTest(450, 10, 400, 5, Stop95))
	is.True(passTest(450, 10, 400, 5, Stop99))
	is.Equal(passTest(450, 10, 450, 5, Stop95), false)
	// 53% win chances with a stdev of 0.01 beats 50% win chances with a stdev of 0.01
	// at the 95% confidence level, but not at the 99% confidence level.
	is.True(passTest(0.53, 0.0001, 0.50, 0.0001, Stop95))
	is.Equal(passTest(0.53, 0.0001, 0.50, 0.0001, Stop99), false)
}

func TestZVal(t *testing.T) {
	is := is.New(t)
	is.Equal(zValStdev(10, 5, 10, 2), float64(0))
	// fmt.Println(zVal(89.97, 77.7924, 78.20, 379.08))
	fmt.Println(zValStdev(89.04, 6.51, 68.58, 11.81))
	is.True(false)

}
