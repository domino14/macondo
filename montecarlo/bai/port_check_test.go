package bai

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// TestPortCheck prints how Run behaves on arms with known normal
// distributions: the scenarios of a matching C driver for MAGPIE's bai(),
// whose numbers these are compared with (see the comment on the results
// below). It runs only when BAI_PORT_CHECK is set.
//
// MAGPIE (d4442999), 2,000 seeds each, mean samples to stop (± standard
// error), delta 0.05 then 0.01:
//
//	a10 min 50:  10,846 ± 136   14,397 ± 158
//	b15 min 50:  10,860 ± 148   14,284 ± 170
//	c30 min 50:  40,377 ± 493   52,980 ± 566
//	a10 min 500: 13,467 ± 130   16,679 ± 154
//	b15 min 500: 16,579 ± 143   19,748 ± 166
//	c30 min 500: 50,030 ± 486   62,152 ± 560
//
// with one wrong pick in 2,000 for a10 and for b15 at delta 0.05, minimum
// 50, and none elsewhere.
func TestPortCheck(t *testing.T) {
	if os.Getenv("BAI_PORT_CHECK") == "" {
		t.Skip("BAI_PORT_CHECK not set")
	}
	type arm struct{ mean, variance float64 }
	var a, b, c []arm
	for i := 0; i < 10; i++ {
		a = append(a, arm{0.50 - 0.02*float64(i), 0.09})
	}
	for i := 0; i < 15; i++ {
		m := 0.45 - 0.01*float64(i)
		if i == 0 {
			m = 0.55
		} else if i == 1 {
			m = 0.53
		}
		b = append(b, arm{m, 0.10})
	}
	for i := 0; i < 30; i++ {
		c = append(c, arm{0.60 - 0.01*float64(i), 0.08})
	}
	const seeds = 2000
	for _, delta := range []float64{0.05, 0.01} {
		for _, minimum := range []int{50, 500} {
			for _, sc := range []struct {
				name string
				arms []arm
			}{{"a10", a}, {"b15", b}, {"c30", c}} {
				var sum, sum2 float64
				wrong, threshold := 0, 0
				for seed := 1; seed <= seeds; seed++ {
					rng := rand.New(rand.NewSource(int64(seed)))
					res, err := Run(context.Background(), len(sc.arms), Options{
						Delta: delta, SampleMinimum: minimum, SampleLimit: len(sc.arms) * 5000, Threads: 1,
					}, func(_, arm int, _ uint64) (float64, error) {
						return sc.arms[arm].mean + math.Sqrt(sc.arms[arm].variance)*rng.NormFloat64(), nil
					})
					if err != nil {
						t.Fatal(err)
					}
					n := float64(res.Samples)
					sum += n
					sum2 += n * n
					if res.Best != 0 {
						wrong++
					}
					if res.Status == StatusThreshold {
						threshold++
					}
				}
				mean := sum / seeds
				sd := math.Sqrt(sum2/seeds - mean*mean)
				fmt.Printf("BAIPORT %s delta=%g min=%d seeds=%d mean_samples=%.1f se=%.1f wrong=%.4f threshold=%.4f\n",
					sc.name, delta, minimum, seeds, mean, sd/math.Sqrt(seeds), float64(wrong)/seeds, float64(threshold)/seeds)
			}
		}
	}
}
