package stats

import (
	"math"

	"gonum.org/v1/gonum/stat/distuv"
	"lukechampine.com/frand"
)

// Tests for a matched-pairs experiment, where each observation is the
// difference between two treatments measured on the same unit -- an autoplay
// game pair being the case at hand, with one bag played twice and the bots
// swapping seats.
//
// The null hypothesis is that the two treatments are exchangeable. Swapping
// their labels negates every difference, so under the null each difference is
// as likely to be d as -d: the differences are symmetric about zero. That
// symmetry is the only thing these tests assume, and in a game-pair run the
// design guarantees it rather than the data having to support it.

// exactSignFlipLimit is the largest number of pairs whose 2^n sign assignments
// are enumerated in full. Beyond it the distribution is sampled.
const exactSignFlipLimit = 20

// defaultSignFlipSamples is how many sign assignments are drawn when there are
// too many pairs to enumerate. At 200k the p-value is resolved to about 2e-3,
// with a Monte Carlo standard error under 0.0011 anywhere near p = 0.05.
const defaultSignFlipSamples = 200000

// SignFlipPValue returns the two-sided p-value for the null hypothesis that
// diffs are symmetric about zero, by comparing the observed mean against the
// means produced by every way of flipping the signs.
//
// It assumes nothing about the shape of the distribution, which matters here:
// the differences from a game-pair run are a spike of exact zeros -- every pair
// whose halves never diverged -- sitting in a long-tailed spread of the rest,
// nothing like a normal distribution.
//
// Zeros are kept. A pair that plays out identically is a real observation of no
// difference, not a missing one, and dropping it would pull the mean away from
// zero. Flipping a zero's sign changes nothing, so they correctly neither add
// evidence nor take it away.
//
// The second return value reports whether the answer is exact (every sign
// assignment enumerated) or sampled.
func SignFlipPValue(diffs []float64, samples int) (float64, bool) {
	n := len(diffs)
	if n == 0 {
		return 1.0, true
	}
	observed := 0.0
	for _, d := range diffs {
		observed += d
	}
	observed = math.Abs(observed)
	// Sums of the same values can only differ by floating-point noise, so
	// compare with a tolerance rather than letting a tie land on the wrong side.
	tol := 1e-9 * (1 + observed)

	if n <= exactSignFlipLimit {
		atLeast := 0
		total := 1 << uint(n)
		for mask := 0; mask < total; mask++ {
			sum := 0.0
			for i, d := range diffs {
				if mask&(1<<uint(i)) != 0 {
					sum -= d
				} else {
					sum += d
				}
			}
			if math.Abs(sum) >= observed-tol {
				atLeast++
			}
		}
		// The observed assignment is one of the ones enumerated, so this needs
		// no further correction.
		return float64(atLeast) / float64(total), true
	}

	if samples <= 0 {
		samples = defaultSignFlipSamples
	}
	// A fixed seed, so the same log always yields the same p-value.
	rng := frand.NewCustom(make([]byte, 32), 1024, 12)
	bits := make([]byte, (n+7)/8)
	atLeast := 0
	for b := 0; b < samples; b++ {
		rng.Read(bits)
		sum := 0.0
		for i, d := range diffs {
			if bits[i/8]&(1<<uint(i%8)) != 0 {
				sum -= d
			} else {
				sum += d
			}
		}
		if math.Abs(sum) >= observed-tol {
			atLeast++
		}
	}
	// Counting the observed value itself keeps the test valid at any number of
	// samples, rather than letting the p-value reach an impossible zero.
	return float64(atLeast+1) / float64(samples+1), false
}

// PairedTTestPValue returns the two-sided p-value from a one-sample t-test on
// the paired differences: the null is that they are drawn from a distribution
// with mean zero. Unlike the sign-flip test it assumes the mean is normally
// distributed, which is a stretch for a handful of pairs, so it is worth having
// beside the exact answer rather than instead of it.
func PairedTTestPValue(meanDiff, stdevDiff float64, n int) float64 {
	if n <= 1 {
		return 1.0
	}
	if stdevDiff == 0 {
		// Every pair produced the same difference. Zero difference is no
		// evidence of anything; a repeated non-zero one is as certain as this
		// test can express.
		if meanDiff == 0 {
			return 1.0
		}
		return 0.0
	}
	se := stdevDiff / math.Sqrt(float64(n))
	t := math.Abs(meanDiff) / se
	dist := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: float64(n - 1)}
	return 2 * dist.Survival(t)
}

// TCriticalValue returns the two-sided critical value at the given confidence
// level for n observations -- 2.064 for 25 of them at 95%, where a normal
// approximation would say 1.96 and quietly understate the interval.
func TCriticalValue(confidence float64, n int) float64 {
	if n <= 1 {
		return math.NaN()
	}
	dist := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: float64(n - 1)}
	return dist.Quantile(1 - (1-confidence)/2)
}
