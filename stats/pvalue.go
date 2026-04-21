package stats

import "math"

// NormalCDF returns the cumulative distribution function of the standard normal distribution.
func NormalCDF(x float64) float64 {
	return 0.5 * math.Erfc(-x/math.Sqrt2)
}

// BinomialZTestPValue returns the two-sided p-value for testing whether the win
// rate equals 0.5 (null hypothesis: both players are equally strong).
// wins and total are float64 to support half-wins from draws.
func BinomialZTestPValue(wins, total float64) float64 {
	if total == 0 {
		return 1.0
	}
	p := wins / total
	se := math.Sqrt(0.25 / total) // se under H0: p=0.5
	z := math.Abs(p-0.5) / se
	return 2 * (1 - NormalCDF(z))
}

// PairedZTestPValue returns the two-sided p-value for testing whether the mean
// paired difference equals zero. Uses a z-test (appropriate for large n).
func PairedZTestPValue(meanDiff, stdevDiff float64, n int) float64 {
	if n <= 1 || stdevDiff == 0 {
		return 1.0
	}
	se := stdevDiff / math.Sqrt(float64(n))
	z := math.Abs(meanDiff) / se
	return 2 * (1 - NormalCDF(z))
}
