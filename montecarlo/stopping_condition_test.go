package montecarlo

import (
	"testing"

	"github.com/domino14/macondo/stats"
	"github.com/matryer/is"
)

func TestWelchTest(t *testing.T) {
	is := is.New(t)
	is.True(welchTest(30, 1, 27.2, 1, stats.Z95))
	is.True(!welchTest(30, 1, 29.0, 1, stats.Z95))

	is.True(!welchTest(100, 5, 90, 4, stats.Z99))
	is.True(welchTest(30, 1, 25, 1, stats.Z999))
}

func TestZTest(t *testing.T) {
	is := is.New(t)
	is.True(!zTest(100, 96, 1.62, -stats.Z98, false))
	// actual significance is about 98.6%
	is.True(zTest(100, 96, 1.62, -stats.Z99, false))

	// Are we 99% sure that 0.002 < 0.005 with the given stderr? Yes
	is.True(zTest(0.005, 0.002, 0.001, -stats.Z99, true))
	// Are we 99% sure that 0.998 > 0.995 with the given stderr? Yes
	is.True(zTest(0.995, 0.998, 0.001, stats.Z99, false))
	// We are NOT 99% sure that 0.997 > 0.995 with the given stderr.
	is.True(!zTest(0.995, 0.997, 0.001, stats.Z99, false))
}
