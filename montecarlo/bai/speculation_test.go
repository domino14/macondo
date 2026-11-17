package bai

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"
	"time"
)

// slow makes a sampler take long enough that threads run out of slots to
// claim, as they do when a sample is many rollouts.
func slow(sample Sampler) Sampler {
	return func(worker, arm int, ordinal uint64) (float64, error) {
		time.Sleep(20 * time.Microsecond)
		return sample(worker, arm, ordinal)
	}
}

// speculating is opts with a held sampler made from sample, whose commits
// are counted in commits.
func speculating(opts Options, sample Sampler, commits *atomic.Int64) Options {
	opts.Held = func(worker, arm int, ordinal uint64) (float64, func(), error) {
		v, err := sample(worker, arm, ordinal)
		return v, func() { commits.Add(1) }, err
	}
	return opts
}

func sameResult(t *testing.T, what string, got, want Result) {
	t.Helper()
	if got.Samples != want.Samples || got.Best != want.Best || got.Status != want.Status {
		t.Fatalf("%s: %d samples best %d %v; want %d best %d %v", what,
			got.Samples, got.Best, got.Status, want.Samples, want.Best, want.Status)
	}
	for a := range want.Means {
		if math.Float64bits(got.Means[a]) != math.Float64bits(want.Means[a]) || got.Counts[a] != want.Counts[a] {
			t.Fatalf("%s: arm %d differs", what, a)
		}
	}
}

// With small rounds and many threads the workers speculate, rounds take
// their samples, and the result is the one-thread result.
func TestSpeculationDoesNotChangeTheResult(t *testing.T) {
	means := []float64{0.44, 0.45, 0.47, 0.40, 0.44, 0.46}
	for _, roundSize := range []int{1, 3, 32} {
		opts := Options{Delta: 0.01, SampleMinimum: 4, SampleLimit: 6000, RoundSize: roundSize}
		opts.Threads = 1
		want, err := Run(context.Background(), len(means), opts, gaussian(means, 0.3, false, 7))
		if err != nil {
			t.Fatal(err)
		}
		for _, threads := range []int{2, 5, 16} {
			opts.Threads = threads
			var commits atomic.Int64
			sample := slow(gaussian(means, 0.3, false, 7))
			r := NewRunner(len(means), opts)
			got, err := r.Run(context.Background(), speculating(opts, sample, &commits), nil)
			if err != nil {
				t.Fatal(err)
			}
			sameResult(t, "one call", got, want)
			if roundSize == 3 && threads == 16 {
				if r.specDrawn == 0 || r.specUsed == 0 {
					t.Fatalf("round size 3, 16 threads: %d speculations, %d used", r.specDrawn, r.specUsed)
				}
				t.Logf("round size 3, 16 threads: %d samples, %d speculations, %d used",
					got.Samples, r.specDrawn, r.specUsed)
			}
			// Every sample folded is committed, and no other.
			if r.specUsed > r.specDrawn || int(commits.Load()) != got.Samples {
				t.Fatalf("%d speculations used of %d drawn; %d samples, %d committed",
					r.specUsed, r.specDrawn, got.Samples, commits.Load())
			}
		}
	}
}

// A run split into calls by sample limits, with speculation carried from
// one call to the next, is the run done in one.
func TestSpeculationAcrossCalls(t *testing.T) {
	means := []float64{0.44, 0.45, 0.47, 0.40, 0.46}
	opts := Options{Delta: 0.01, SampleMinimum: 4, SampleLimit: 5000, RoundSize: 3, Threads: 1}
	want, err := Run(context.Background(), len(means), opts, gaussian(means, 0.3, false, 11))
	if err != nil {
		t.Fatal(err)
	}
	opts.Threads = 12
	r := NewRunner(len(means), opts)
	var got Result
	var commits atomic.Int64
	for limit := 37; ; limit += 37 {
		o := opts
		o.SampleLimit = min(limit, opts.SampleLimit)
		sample := slow(gaussian(means, 0.3, false, 11))
		got, err = r.Run(context.Background(), speculating(o, sample, &commits), nil)
		if err == nil && int(commits.Load()) != got.Samples {
			t.Fatalf("%d samples, %d committed", got.Samples, commits.Load())
		}
		if err != nil {
			t.Fatal(err)
		}
		if got.Status != StatusSampleLimit || o.SampleLimit == opts.SampleLimit {
			break
		}
	}
	sameResult(t, "split", got, want)
}

// A sampler that fails now and then, whether on a claimed sample or a
// speculative one, stops the call with its error, and the run carries on to
// the same result.
func TestSpeculationSamplerError(t *testing.T) {
	means := []float64{0.44, 0.45, 0.47, 0.40, 0.46}
	opts := Options{Delta: 0.01, SampleMinimum: 4, SampleLimit: 3000, RoundSize: 3, Threads: 1}
	want, err := Run(context.Background(), len(means), opts, gaussian(means, 0.3, false, 13))
	if err != nil {
		t.Fatal(err)
	}
	opts.Threads = 12
	r := NewRunner(len(means), opts)
	boom := errors.New("boom")
	var calls atomic.Int64
	good := slow(gaussian(means, 0.3, false, 13))
	failing := func(worker, arm int, ordinal uint64) (float64, error) {
		if calls.Add(1)%401 == 0 {
			return 0, boom
		}
		return good(worker, arm, ordinal)
	}
	failures := 0
	var commits atomic.Int64
	for {
		got, err := r.Run(context.Background(), speculating(opts, failing, &commits), nil)
		if err != nil {
			if !errors.Is(err, boom) {
				t.Fatal(err)
			}
			failures++
			continue
		}
		sameResult(t, "after errors", got, want)
		if int(commits.Load()) != got.Samples {
			t.Fatalf("%d samples, %d committed", got.Samples, commits.Load())
		}
		break
	}
	if failures == 0 {
		t.Fatal("the sampler never failed")
	}
}
