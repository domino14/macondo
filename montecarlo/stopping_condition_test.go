package montecarlo

import (
	"testing"

	"github.com/domino14/macondo/stats"
	"github.com/matryer/is"
)

func TestWelchTest(t *testing.T) {
	z95 := stats.ZVal(95)
	z99 := stats.ZVal(99)
	z999 := stats.ZVal(99.9)
	is := is.New(t)
	is.True(welchTest(30, 1, 27.2, 1, z95))
	is.True(!welchTest(30, 1, 29.0, 1, z95))

	is.True(!welchTest(100, 5, 90, 4, z99))
	is.True(welchTest(30, 1, 25, 1, z999))
}

func TestZTest(t *testing.T) {
	z98 := stats.ZVal(98)
	z99 := stats.ZVal(99)
	is := is.New(t)
	is.True(!zTest(100, 96, 1.62, -z98, false))
	// actual significance is about 98.6%
	is.True(zTest(100, 96, 1.62, -z99, false))

	// Are we 99% sure that 0.002 < 0.005 with the given stderr? Yes
	is.True(zTest(0.005, 0.002, 0.001, -z99, true))
	// Are we 99% sure that 0.998 > 0.995 with the given stderr? Yes
	is.True(zTest(0.995, 0.998, 0.001, z99, false))
	// We are NOT 99% sure that 0.997 > 0.995 with the given stderr.
	is.True(!zTest(0.995, 0.997, 0.001, z99, false))
	is.True(!zTest(0.995, 0.990, 0.001, z99, false))
}
