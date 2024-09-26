package stats

import "gonum.org/v1/gonum/stat/distuv"

// ZVal returns the two-tailed Z-value associated with a specific confidence interval.
// The interval is a number from 0 to 100 percent.
func ZVal(confidenceInterval float64) float64 {
	dist := distuv.Normal{
		Mu:    0,
		Sigma: 1,
	}
	area := (1 + (confidenceInterval / 100)) / 2
	zValue := dist.Quantile(area)
	return zValue
}
