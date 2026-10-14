// Package bai is best-arm identification for simulations: which of several
// plays has the highest mean, with a stopping rule that says when the sample
// is big enough to tell. It is a port of MAGPIE's bai.h (the top-two IDS
// sampling rule with the GK16 threshold, from Jourdan et al., "Top Two
// Algorithms Revisited"), with MAGPIE's paired statistics for arms whose
// samples share their luck. MAGPIE's scheduler is replaced by a simpler one
// built on goroutines: samples run in rounds, every round's allocation is
// computed from the statistics of the rounds before it, and rounds fold in
// slot order, so the result does not depend on thread timing.
package bai

import (
	"context"
	"errors"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// Status is why a run stopped.
type Status int

const (
	StatusNone Status = iota
	// StatusThreshold: every other arm is similar to the best or provably
	// worse.
	StatusThreshold
	// StatusCutoff: the best arm's mean is at an extreme (a decided game).
	StatusCutoff
	// StatusSampleLimit: the sample budget is spent.
	StatusSampleLimit
	// StatusTimeout: the time limit passed.
	StatusTimeout
	// StatusCanceled: the context was canceled.
	StatusCanceled
)

func (s Status) String() string {
	switch s {
	case StatusThreshold:
		return "threshold"
	case StatusCutoff:
		return "cutoff"
	case StatusSampleLimit:
		return "sample-limit"
	case StatusTimeout:
		return "timeout"
	case StatusCanceled:
		return "canceled"
	}
	return "none"
}

const (
	// fixedPointShift scales sample values into int64 accumulators, so that
	// sums do not depend on the order samples are added in.
	fixedPointShift = 30
	fixedPointScale = int64(1) << fixedPointShift
	// maxSampleMagnitude bounds a sample, and so its square, well inside the
	// accumulators.
	maxSampleMagnitude = 1 << 10
	minimumVariance    = 1e-10
	// maxPairedOrdinals is how many ordinals the paired statistics keep. MAGPIE
	// found the stopping test stops as early with 128 as with all of them.
	maxPairedOrdinals = 1024
	// DefaultRoundSize is how many samples a round holds.
	DefaultRoundSize = 32
)

// Options configure a run.
type Options struct {
	// Delta is the error probability the stopping threshold allows (1 - the
	// confidence: 0.01 for 99%). Zero means no threshold: the run stops only
	// on the sample limit, the cutoff, all arms being similar, the time limit
	// or the context.
	Delta float64
	// SampleMinimum is how many samples every arm gets, round-robin, before
	// any adaptive sampling or stopping.
	SampleMinimum int
	// SampleLimit is the most samples, over all arms; 0 for no limit.
	SampleLimit int
	// TimeLimit stops the run; 0 for none.
	TimeLimit time.Duration
	// Cutoff stops the run once the best arm's mean is at most Cutoff or at
	// least 1 - Cutoff (a win probability at an extreme). Zero disables it.
	Cutoff float64
	// Threads is how many samples run at once.
	Threads int
	// RoundSize is how many samples a round holds; DefaultRoundSize if 0.
	RoundSize int
	// Similar reports whether two arms are interchangeable: an arm similar to
	// the best never needs to be beaten. Nil means none are.
	Similar func(a, b int) bool
	// Paired says arms' samples at the same ordinal share their luck (common
	// random numbers), so the stopping test may use their covariance.
	Paired bool
}

// Sampler draws arm's ordinal-th sample on worker (0 <= worker < Threads).
// It may be called concurrently for different workers. A sample with a
// shared seed per ordinal is what makes Options.Paired valid.
type Sampler func(worker, arm int, ordinal uint64) (float64, error)

// Result is the outcome of a run.
type Result struct {
	Status Status
	// Best is the arm with the highest mean (lowest index on a tie).
	Best int
	// Samples is how many samples were folded, over all arms.
	Samples int
	// Means and Counts are per arm.
	Means  []float64
	Counts []int
}

type armDatum struct {
	numSamples   uint64
	numScheduled uint64
	sum          int64
	sumSquares   int64
	mean         float64
	variance     float64
}

var errSampleRange = errors.New("bai: sample out of range")

func quantize(v float64) int64 {
	return int64(math.Round(v * float64(fixedPointScale)))
}

func (a *armDatum) add(v float64) error {
	if math.IsNaN(v) || math.Abs(v) > maxSampleMagnitude {
		return errSampleRange
	}
	a.numSamples++
	a.sum += quantize(v)
	a.sumSquares += quantize(v * v)
	inv := 1.0 / float64(fixedPointScale)
	n := float64(a.numSamples)
	a.mean = float64(a.sum) * inv / n
	a.variance = float64(a.sumSquares)*inv/n - a.mean*a.mean
	if a.variance < minimumVariance {
		a.variance = minimumVariance
	}
	return nil
}

// altLambda is the mean that best explains both arms under the alternative
// that they are equal, weighting each by its sample count.
func altLambda(mu1, sigma21, w1, mua, sigma2a, wa float64) float64 {
	if w1 == 0 {
		return mua
	}
	if wa == 0 || mu1 == mua {
		return mu1
	}
	x := wa / w1
	return (sigma2a*mu1 + x*sigma21*mua) / (sigma2a + x*sigma21)
}

func klGaussian(mu, sigma2, lambda float64) float64 {
	d := mu - lambda
	return 0.5 * d * d / sigma2
}

// zWithCounts is the GLR statistic for "astar beats a", weighting the arms
// by the given counts.
func zWithCounts(astar *armDatum, na float64, a *armDatum, nb float64) float64 {
	l := altLambda(astar.mean, astar.variance, na, a.mean, a.variance, nb)
	return na*klGaussian(astar.mean, astar.variance, l) + nb*klGaussian(a.mean, a.variance, l)
}

// state is a run's statistics.
type state struct {
	opts  Options
	arms  []armDatum
	total uint64
	astar int

	// Paired statistics over the first maxPairedOrdinals ordinals of each arm.
	pairedValues  []float64 // [arm*maxPairedOrdinals + ordinal]
	prefix        []int64   // [arm*(maxPairedOrdinals+1) + n]: sum of the first n
	prefixSquares []int64
	cross         []int64 // [low*numArms + high]: sum over common ordinals of products
}

func newState(numArms int, opts Options) *state {
	s := &state{opts: opts, arms: make([]armDatum, numArms), astar: -1}
	if opts.Paired {
		s.pairedValues = make([]float64, numArms*maxPairedOrdinals)
		s.prefix = make([]int64, numArms*(maxPairedOrdinals+1))
		s.prefixSquares = make([]int64, numArms*(maxPairedOrdinals+1))
		s.cross = make([]int64, numArms*numArms)
	}
	return s
}

func (s *state) similar(a, b int) bool {
	return s.opts.Similar != nil && s.opts.Similar(a, b)
}

// record folds arm's sample at ordinal into the paired statistics. Each arm's
// samples arrive in ordinal order, so each pair's product at an ordinal is
// added once, by whichever arm folds it second.
func (s *state) record(arm int, ordinal uint64, v float64) {
	if s.pairedValues == nil || ordinal >= maxPairedOrdinals {
		return
	}
	n := len(s.arms)
	s.pairedValues[arm*maxPairedOrdinals+int(ordinal)] = v
	base := arm * (maxPairedOrdinals + 1)
	s.prefix[base+int(ordinal)+1] = s.prefix[base+int(ordinal)] + quantize(v)
	s.prefixSquares[base+int(ordinal)+1] = s.prefixSquares[base+int(ordinal)] + quantize(v*v)
	for other := range s.arms {
		if other == arm || s.arms[other].numSamples <= ordinal {
			continue
		}
		lo, hi := min(arm, other), max(arm, other)
		s.cross[lo*n+hi] += quantize(v * s.pairedValues[other*maxPairedOrdinals+int(ordinal)])
	}
}

func (s *state) prefixSum(p []int64, arm int, pairs uint64) float64 {
	return float64(p[arm*(maxPairedOrdinals+1)+int(pairs)]) / float64(fixedPointScale)
}

// pairedZ is the Z statistic for "astar beats arm" when the arms' samples at
// an ordinal share a seed:
//
//	Var(mean_a - mean_b) = v_a/n_a + v_b/n_b - 2 c m / (n_a n_b)
//
// with v_a, v_b and the covariance c all from the first m common ordinals.
// When that is not positive it falls back to zWithCounts, which ignores the
// shared luck and so never overstates the evidence.
func (s *state) pairedZ(astarIdx, arm int) float64 {
	astar, a := &s.arms[astarIdx], &s.arms[arm]
	na, nb := float64(astar.numSamples), float64(a.numSamples)
	common := min(astar.numSamples, a.numSamples)
	pairs := min(common, maxPairedOrdinals)
	variance := 0.0
	if pairs > 1 {
		p := float64(pairs)
		ma := s.prefixSum(s.prefix, astarIdx, pairs) / p
		mb := s.prefixSum(s.prefix, arm, pairs) / p
		va := s.prefixSum(s.prefixSquares, astarIdx, pairs)/p - ma*ma
		vb := s.prefixSum(s.prefixSquares, arm, pairs)/p - mb*mb
		lo, hi := min(astarIdx, arm), max(astarIdx, arm)
		crossSum := float64(s.cross[lo*len(s.arms)+hi]) / float64(fixedPointScale)
		cov := crossSum/p - ma*mb
		variance = va/na + vb/nb - 2*cov*float64(common)/(na*nb)
	}
	if !(variance > 0) {
		return zWithCounts(astar, na, a, nb)
	}
	gap := astar.mean - a.mean
	return gap * gap / (2 * variance)
}

func (s *state) updateAstar() float64 {
	s.astar = -1
	best := 0.0
	for i := range s.arms {
		if s.arms[i].numSamples == 0 {
			continue
		}
		if s.astar < 0 || s.arms[i].mean > best {
			s.astar, best = i, s.arms[i].mean
		}
	}
	return best
}

// thresholdReached checks every arm other than astar is similar to it or has
// a Z statistic above the GK16 threshold. With no threshold only the first
// can hold.
func (s *state) thresholdReached() bool {
	beta := math.Log((math.Log(float64(s.total)) + 1) / s.opts.Delta)
	astar := &s.arms[s.astar]
	for i := range s.arms {
		if i == s.astar || s.similar(s.astar, i) {
			continue
		}
		if s.opts.Delta <= 0 {
			return false
		}
		var z float64
		if s.pairedValues != nil {
			z = s.pairedZ(s.astar, i)
		} else {
			z = zWithCounts(astar, float64(astar.numSamples), &s.arms[i], float64(s.arms[i].numSamples))
		}
		if !(z > beta) {
			return false
		}
	}
	return true
}

// challenger is the top-two challenger: the arm minimizing Z + log(N)
// against astar, weighting arms by the samples scheduled to them. astar if
// every other arm is similar to it.
func (s *state) challenger() int {
	astar := &s.arms[s.astar]
	na := float64(astar.numScheduled)
	best, bestValue := -1, 0.0
	for i := range s.arms {
		if i == s.astar || s.similar(s.astar, i) {
			continue
		}
		n := float64(s.arms[i].numScheduled)
		v := zWithCounts(astar, na, &s.arms[i], n) + math.Log(n)
		if best < 0 || v < bestValue {
			best, bestValue = i, v
		}
	}
	if best < 0 {
		return s.astar
	}
	return best
}

// slot is one sample of a round.
type slot struct {
	arm     int
	ordinal uint64
	value   float64
	err     error
}

// scheduleTopTwo splits n samples between astar and the challenger by the
// top-two IDS proportion, rounded (the stratified form of MAGPIE's rule).
func (s *state) scheduleTopTwo(n int, out []slot) []slot {
	c := s.challenger()
	a, b := &s.arms[s.astar], &s.arms[c]
	na, nb := float64(a.numScheduled), float64(b.numScheduled)
	theta := (na*a.mean + nb*b.mean) / (na + nb)
	num := na * klGaussian(a.mean, a.variance, theta)
	den := num + nb*klGaussian(b.mean, b.variance, theta)
	share := 0.0
	if den > 0 {
		share = min(max(num/den, 0), 1)
	}
	astarCount := min(max(int(math.Round(share*float64(n))), 0), n)
	for i := 0; i < n; i++ {
		arm := c
		if i < astarCount {
			arm = s.astar
		}
		out = append(out, slot{arm: arm, ordinal: s.arms[arm].numScheduled})
		s.arms[arm].numScheduled++
	}
	return out
}

// Runner is a best-arm identification run that can be continued: each Run
// call picks up the statistics, schedule and ordinals where the last left off.
type Runner struct {
	s           *state
	numArms     int
	nextInitial int
	status      Status
	// pending is the rest of a round a call's sample limit cut short; the
	// next call draws it first, so rounds are laid out the same however a
	// run is split into calls.
	pending []slot
}

// NewRunner starts a run over numArms arms. Options.Similar and
// Options.Paired are fixed for the run; the rest are read on each Run call.
func NewRunner(numArms int, opts Options) *Runner {
	return &Runner{s: newState(numArms, opts), numArms: numArms}
}

// Samples is how many samples have been folded so far.
func (r *Runner) Samples() int { return int(r.s.total) }

// Run finds the best of numArms arms.
func Run(ctx context.Context, numArms int, opts Options, sample Sampler) (Result, error) {
	if numArms < 1 {
		return Result{}, errors.New("bai: no arms")
	}
	return NewRunner(numArms, opts).Run(ctx, opts, sample)
}

// Run samples until a stop condition, continuing from earlier calls.
// opts.SampleLimit counts every sample of the run, earlier calls' included.
// With one thread the samples are drawn on the calling goroutine.
func (r *Runner) Run(ctx context.Context, opts Options, sample Sampler) (Result, error) {
	numArms := r.numArms
	s := r.s
	opts.Similar, opts.Paired = s.opts.Similar, s.opts.Paired
	if opts.Threads < 1 {
		opts.Threads = 1
	}
	if opts.RoundSize < 1 {
		opts.RoundSize = DefaultRoundSize
	}
	if opts.SampleMinimum < 1 {
		opts.SampleMinimum = 1
	}
	s.opts = opts
	var deadline time.Time
	if opts.TimeLimit > 0 {
		deadline = time.Now().Add(opts.TimeLimit)
	}
	initialLimit := numArms * opts.SampleMinimum
	var workers *pool
	if opts.Threads > 1 {
		workers = newPool(ctx, opts.Threads, sample)
		defer workers.close()
	}
	status := StatusNone
	var round []slot
	for status == StatusNone {
		round = round[:0]
		budget := opts.RoundSize
		if opts.SampleLimit > 0 {
			budget = opts.SampleLimit - int(s.total)
		}
		if budget <= 0 {
			status = StatusSampleLimit
			break
		}
		if ctx.Err() != nil {
			status = StatusCanceled
			break
		}
		if len(r.pending) > 0 {
			round = append(round, r.pending...)
			r.pending = r.pending[:0]
		} else {
			// The initial phase lays out every arm's first SampleMinimum
			// samples iteration-major: consecutive slots are the same ordinal
			// across arms.
			for r.nextInitial < initialLimit && len(round) < opts.RoundSize {
				arm := r.nextInitial % numArms
				round = append(round, slot{arm: arm, ordinal: s.arms[arm].numScheduled})
				s.arms[arm].numScheduled++
				r.nextInitial++
			}
			if len(round) == 0 {
				if numArms == 1 {
					status = StatusThreshold
					break
				}
				round = s.scheduleTopTwo(opts.RoundSize, round)
			}
		}
		if len(round) > budget {
			r.pending = append(r.pending, round[budget:]...)
			round = round[:budget]
		}
		var err error
		if workers != nil {
			err = workers.run(ctx, round)
		} else {
			err = runInline(ctx, round, sample)
		}
		if err != nil {
			// The round is drawn again next call.
			r.pending = append(append([]slot(nil), round...), r.pending...)
			if ctx.Err() != nil {
				status = StatusCanceled
				break
			}
			return Result{}, err
		}
		// Fold in slot order.
		for i := range round {
			sl := &round[i]
			ordinal := s.arms[sl.arm].numSamples
			if err := s.arms[sl.arm].add(sl.value); err != nil {
				return Result{}, err
			}
			s.record(sl.arm, ordinal, sl.value)
		}
		s.total += uint64(len(round))
		best := s.updateAstar()
		switch {
		case ctx.Err() != nil:
			status = StatusCanceled
		case !deadline.IsZero() && time.Now().After(deadline):
			status = StatusTimeout
		case int(s.total) < initialLimit:
		case opts.Cutoff > 0 && (best <= opts.Cutoff || best >= 1-opts.Cutoff):
			status = StatusCutoff
		case s.thresholdReached():
			status = StatusThreshold
		case opts.SampleLimit > 0 && int(s.total) >= opts.SampleLimit:
			status = StatusSampleLimit
		}
	}
	r.status = status
	res := Result{Status: status, Best: s.astar, Samples: int(s.total),
		Means: make([]float64, numArms), Counts: make([]int, numArms)}
	for i := range s.arms {
		res.Means[i] = s.arms[i].mean
		res.Counts[i] = int(s.arms[i].numSamples)
	}
	if res.Best < 0 {
		res.Best = 0
	}
	return res, nil
}

// runInline draws a round's samples on the calling goroutine.
func runInline(ctx context.Context, round []slot, sample Sampler) error {
	for i := range round {
		if err := ctx.Err(); err != nil {
			return err
		}
		v, err := sample(0, round[i].arm, round[i].ordinal)
		if err != nil {
			return err
		}
		round[i].value = v
	}
	return nil
}

// pool runs rounds' samples on persistent workers, each locked to its OS
// thread for the run so that per-thread caches (MAGPIE's move generator's)
// stay warm.
type pool struct {
	work chan []slot
	done sync.WaitGroup
	quit chan struct{}
	next atomic.Int64
	errs []error
}

func newPool(ctx context.Context, threads int, sample Sampler) *pool {
	p := &pool{work: make(chan []slot), quit: make(chan struct{}), errs: make([]error, threads)}
	for w := 0; w < threads; w++ {
		go func(worker int) {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			for {
				select {
				case <-p.quit:
					return
				case round := <-p.work:
					for ctx.Err() == nil {
						i := int(p.next.Add(1) - 1)
						if i >= len(round) {
							break
						}
						v, err := sample(worker, round[i].arm, round[i].ordinal)
						if err != nil {
							p.errs[worker] = err
							break
						}
						round[i].value = v
					}
					p.done.Done()
				}
			}
		}(w)
	}
	return p
}

// run draws every slot of round on all the workers.
func (p *pool) run(ctx context.Context, round []slot) error {
	p.next.Store(0)
	workers := len(p.errs)
	p.done.Add(workers)
	for w := 0; w < workers; w++ {
		p.work <- round
	}
	p.done.Wait()
	for i, err := range p.errs {
		if err != nil {
			p.errs[i] = nil
			return err
		}
	}
	return ctx.Err()
}

func (p *pool) close() { close(p.quit) }
