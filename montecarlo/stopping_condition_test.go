package montecarlo

import (
	"testing"

	"github.com/domino14/macondo/stats"
	"github.com/matryer/is"
)

func TestPassTest(t *testing.T) {
	is := is.New(t)
	is.True(passTest(30, 1, 27.2, 1, stats.Z95))
	is.True(!passTest(30, 1, 29.0, 1, stats.Z95))

	is.True(!passTest(100, 5, 90, 4, stats.Z99))
	is.True(passTest(30, 1, 25, 1, stats.Z999))
}
