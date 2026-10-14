package bai

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"
)

// gaussian draws arm's ordinal-th sample from N(means[arm], sd^2), seeded by
// (arm, ordinal) so a run is reproducible whatever the threads do. With
// paired set the noise depends on the ordinal only, as common random numbers.
func gaussian(means []float64, sd float64, paired bool, seed uint64) Sampler {
	return func(_ int, arm int, ordinal uint64) (float64, error) {
		key := uint64(arm) + 1
		if paired {
			key = 0
		}
		r := rand.New(rand.NewPCG(seed^ordinal*0x9E3779B97F4A7C15, key))
		return means[arm] + sd*r.NormFloat64(), nil
	}
}

func TestFindsBestArm(t *testing.T) {
	means := []float64{0.40, 0.45, 0.55, 0.50, 0.30}
	wrong := 0
	for trial := uint64(0); trial < 40; trial++ {
		res, err := Run(context.Background(), len(means), Options{
			Delta: 0.05, SampleMinimum: 10, SampleLimit: 200000, Threads: 4,
		}, gaussian(means, 0.3, false, trial))
		if err != nil {
			t.Fatal(err)
		}
		if res.Status != StatusThreshold {
			t.Fatalf("trial %d stopped on %v after %d samples", trial, res.Status, res.Samples)
		}
		if res.Best != 2 {
			wrong++
		}
		// Top-two spends its samples on the contenders, not the worst arm.
		if res.Counts[4] >= res.Counts[2] {
			t.Errorf("trial %d: worst arm got %d samples, best %d", trial, res.Counts[4], res.Counts[2])
		}
	}
	// delta = 0.05 allows 5% wrong; the threshold is conservative.
	if wrong > 4 {
		t.Fatalf("picked the wrong arm %d times in 40", wrong)
	}
}

// Shared luck makes a gap provable with far fewer samples.
func TestPairedStopsSooner(t *testing.T) {
	means := []float64{0.50, 0.52}
	samples := func(paired bool) int {
		total := 0
		for trial := uint64(0); trial < 10; trial++ {
			res, err := Run(context.Background(), 2, Options{
				Delta: 0.01, SampleMinimum: 10, SampleLimit: 1000000, Threads: 2, Paired: paired,
			}, gaussian(means, 0.3, paired, trial))
			if err != nil {
				t.Fatal(err)
			}
			if res.Best != 1 {
				t.Fatalf("paired=%v trial %d picked arm %d", paired, trial, res.Best)
			}
			total += res.Samples
		}
		return total
	}
	unpaired, paired := samples(false), samples(true)
	if paired*10 > unpaired {
		t.Fatalf("paired took %d samples, unpaired %d", paired, unpaired)
	}
}

func TestSimilarArmsNeedNotBeBeaten(t *testing.T) {
	// Arms 0 and 1 are the same play; only arm 2 has to be ruled out.
	means := []float64{0.6, 0.6, 0.3}
	res, err := Run(context.Background(), 3, Options{
		Delta: 0.01, SampleMinimum: 10, SampleLimit: 1000000, Threads: 1,
		Similar: func(a, b int) bool { return a+b == 1 },
	}, gaussian(means, 0.2, false, 1))
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusThreshold || res.Best == 2 || res.Samples > 2000 {
		t.Fatalf("got %+v", res)
	}
}

func TestCutoffAndLimit(t *testing.T) {
	res, err := Run(context.Background(), 3, Options{
		Delta: 0.01, SampleMinimum: 5, Cutoff: 0.01, Threads: 1,
	}, gaussian([]float64{0.999, 0.999, 0.998}, 0.0001, false, 1))
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusCutoff {
		t.Fatalf("decided game stopped on %v", res.Status)
	}
	res, err = Run(context.Background(), 2, Options{
		Delta: 0.01, SampleMinimum: 5, SampleLimit: 500, Threads: 3,
	}, gaussian([]float64{0.5, 0.5}, 0.3, false, 1))
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusSampleLimit || res.Samples != 500 {
		t.Fatalf("tied arms: %v after %d samples", res.Status, res.Samples)
	}
}

// The result does not depend on how many threads ran it.
func TestThreadsDoNotChangeTheResult(t *testing.T) {
	means := []float64{0.40, 0.45, 0.47, 0.30}
	var first Result
	for i, threads := range []int{1, 2, 7, 16} {
		res, err := Run(context.Background(), len(means), Options{
			Delta: 0.01, SampleMinimum: 20, SampleLimit: 100000, Threads: threads, Paired: true,
		}, gaussian(means, 0.3, true, 99))
		if err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			first = res
			continue
		}
		if res.Samples != first.Samples || res.Best != first.Best {
			t.Fatalf("%d threads: %d samples best %d; 1 thread: %d best %d",
				threads, res.Samples, res.Best, first.Samples, first.Best)
		}
		for a := range means {
			if math.Float64bits(res.Means[a]) != math.Float64bits(first.Means[a]) {
				t.Fatalf("%d threads: arm %d mean differs", threads, a)
			}
		}
	}
}

func TestCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	n := 0
	res, err := Run(ctx, 2, Options{Delta: 0.01, Threads: 1}, func(_, arm int, _ uint64) (float64, error) {
		n++
		if n == 100 {
			cancel()
		}
		return 0.5, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusCanceled {
		t.Fatalf("got %v", res.Status)
	}
}
