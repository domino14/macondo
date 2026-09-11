package stats

import (
	"math"
	"testing"

	"github.com/matryer/is"
)

func closeTo(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// With a handful of pairs the sign-flip distribution can be worked out by hand,
// which is the best check that the implementation counts what it should.
func TestSignFlipPValueSmallCases(t *testing.T) {
	is := is.New(t)

	// Three pairs, all favouring the same side. Of the 2^3 = 8 sign
	// assignments, only all-plus and all-minus reach |sum| = 6, so p = 2/8.
	p, exact := SignFlipPValue([]float64{1, 2, 3}, 0)
	is.True(exact)
	is.True(closeTo(p, 0.25, 1e-12))

	// One pair cannot say anything: both assignments tie the observed value.
	p, exact = SignFlipPValue([]float64{5}, 0)
	is.True(exact)
	is.True(closeTo(p, 1.0, 1e-12))

	// Nothing at all to test.
	p, _ = SignFlipPValue(nil, 0)
	is.True(closeTo(p, 1.0, 1e-12))

	// All zeros: every assignment gives the same sum of zero, so no evidence.
	p, _ = SignFlipPValue([]float64{0, 0, 0, 0}, 0)
	is.True(closeTo(p, 1.0, 1e-12))
}

// Ten pairs all pointing the same way is the strongest evidence ten pairs can
// give: 2 of 1024 assignments match it.
func TestSignFlipPValueAllSameSign(t *testing.T) {
	is := is.New(t)
	diffs := make([]float64, 10)
	for i := range diffs {
		diffs[i] = float64(i + 1)
	}
	p, exact := SignFlipPValue(diffs, 0)
	is.True(exact)
	is.True(closeTo(p, 2.0/1024.0, 1e-12))
}

// Zeros must neither add evidence nor dilute it away: padding with pairs that
// came out even leaves the p-value where it was. (Dropping them, as a signed
// rank test would, is what this guards against.)
func TestSignFlipPValueZerosAreNeutral(t *testing.T) {
	is := is.New(t)

	base, _ := SignFlipPValue([]float64{1, 2, 3}, 0)
	padded, _ := SignFlipPValue([]float64{1, 2, 3, 0, 0, 0, 0}, 0)
	is.True(closeTo(base, padded, 1e-12))
}

// Above the enumeration limit the test switches to sampling, which has to land
// close to the exact answer and stay strictly positive.
func TestSignFlipPValueSampled(t *testing.T) {
	is := is.New(t)

	// 24 pairs of equal size: the exact two-sided p-value is the chance that a
	// sum of 24 fair +-1 steps is at its extreme, 2/2^24.
	diffs := make([]float64, 24)
	for i := range diffs {
		diffs[i] = 1
	}
	p, exact := SignFlipPValue(diffs, 50000)
	is.True(!exact)
	is.True(p > 0)                  // never claims impossibility from a sample
	is.True(p <= 2.0/50001.0+1e-12) // and finds nothing more extreme

	// A sampled run is reproducible: same input, same answer.
	mixed := make([]float64, 25)
	for i := range mixed {
		mixed[i] = float64((i%7)-3) * 10
	}
	p1, _ := SignFlipPValue(mixed, 20000)
	p2, _ := SignFlipPValue(mixed, 20000)
	is.Equal(p1, p2)
}

// The sampled path has to agree with the exact one on a case where both can be
// computed, or the sampling is wrong somewhere.
func TestSignFlipSampledMatchesExact(t *testing.T) {
	is := is.New(t)

	diffs := []float64{40, -12, 33, 5, -60, 18, 7, -3, 25, 11,
		-45, 9, 0, 14, -22, 31, 6, -8, 19, 2}
	exactP, exact := SignFlipPValue(diffs, 0)
	is.True(exact)

	// Same values with a 21st zero pair, which changes nothing statistically
	// but pushes the count past the enumeration limit.
	sampled, wasExact := SignFlipPValue(append(append([]float64{}, diffs...), 0), 200000)
	is.True(!wasExact)
	is.True(closeTo(sampled, exactP, 0.01))
}

// Reference values from a t table: t = 2.064 at 95% with 24 degrees of freedom,
// and 2.776 at 95% with 4.
func TestTCriticalValue(t *testing.T) {
	is := is.New(t)
	is.True(closeTo(TCriticalValue(0.95, 25), 2.0639, 1e-3))
	is.True(closeTo(TCriticalValue(0.95, 5), 2.7764, 1e-3))
	is.True(closeTo(TCriticalValue(0.99, 25), 2.7969, 1e-3))
	// A single observation has no spread to speak of.
	is.True(math.IsNaN(TCriticalValue(0.95, 1)))
}

// A worked example: mean 10, sd 20, n = 25 gives t = 2.5 on 24 df, whose
// two-sided p-value is 0.0196.
func TestPairedTTestPValue(t *testing.T) {
	is := is.New(t)
	is.True(closeTo(PairedTTestPValue(10, 20, 25), 0.0196, 1e-3))

	// Symmetric in the direction of the effect.
	is.True(closeTo(PairedTTestPValue(-10, 20, 25), PairedTTestPValue(10, 20, 25), 1e-12))

	// Degenerate inputs: no pairs, no spread.
	is.True(closeTo(PairedTTestPValue(10, 20, 1), 1.0, 1e-12))
	is.True(closeTo(PairedTTestPValue(0, 0, 25), 1.0, 1e-12))
	is.True(closeTo(PairedTTestPValue(5, 0, 25), 0.0, 1e-12))
}

// The two tests should broadly agree when the differences are well behaved,
// which is the regime the t-test is entitled to.
func TestSignFlipAgreesWithTOnTameData(t *testing.T) {
	is := is.New(t)

	diffs := []float64{12, -5, 20, 3, -8, 15, 7, 1, -2, 9,
		11, -6, 4, 18, -1, 6, 13, 2, -4, 8}
	mean, sd := meanStdev(diffs)
	tp := PairedTTestPValue(mean, sd, len(diffs))
	sp, _ := SignFlipPValue(diffs, 0)
	is.True(closeTo(tp, sp, 0.02))
}

func meanStdev(xs []float64) (float64, float64) {
	s := &Statistic{}
	for _, x := range xs {
		s.Push(x)
	}
	return s.Mean(), s.Stdev()
}
