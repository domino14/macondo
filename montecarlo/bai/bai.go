// Package bai is best-arm identification for simulations: which of several
// plays has the highest mean, with a stopping rule that says when the sample
// is big enough to tell. It is a port of MAGPIE's bai.h (the top-two IDS
// sampling rule with the GK16 threshold, from Jourdan et al., "Top Two
// Algorithms Revisited"), with MAGPIE's paired statistics for arms whose
// samples share their luck. Samples are scheduled as MAGPIE schedules them:
// in rounds, each laid out from the statistics committed when an earlier
// round folded, with several rounds live at once so that threads do not wait
// for one another, and folded in slot order, so the result does not depend
// on thread timing or on the number of threads.
package bai

import (
	"context"
	"errors"
	"math"
	"sync"
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
	// Round j is laid out when round j - lag - 1 folds. The rounds of the
	// initial phase need no statistics, so they run far ahead: a slow sample
	// in the oldest round does not keep the other threads from the rounds
	// behind it.
	roundRobinLag = 32
	// Adaptive rounds read statistics, and every round of lag makes them
	// staler. MAGPIE measured two rounds to choose as well as none.
	adaptiveLag = 2
	// roundRobinClaimBatch is how many consecutive slots of a round-robin
	// round a worker takes at once. They are the same ordinal of consecutive
	// arms, so samples that share their luck run on one thread, whose caches
	// that keeps warm. Adaptive rounds are claimed a slot at a time.
	roundRobinClaimBatch = 8
)

// Options configure a run.
type Options struct {
	// Delta is the error probability the stopping threshold allows (1 - the
	// confidence: 0.01 for 99%). Zero means no threshold: the run stops only
	// on the sample limit, the cutoff, all arms being similar, the time limit
	// or the context.
	Delta float64
	// SampleMinimum is how many samples every arm gets, round-robin, before
	// any adaptive sampling or stopping. It is fixed by a run's first call.
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
	// RoundSize is how many samples a round holds; DefaultRoundSize if 0. It
	// is fixed by a run's first call.
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
	s       *state
	numArms int
	status  Status

	mu   sync.Mutex
	cond *sync.Cond
	// started is set by the first Run call, which fixes the round size and
	// the length of the initial phase.
	started      bool
	roundSize    int
	initialLimit uint64
	// initialPhase is true until the round holding the last sample of the
	// initial phase has folded. Rounds laid out before then are round-robin.
	initialPhase bool
	// requested is how many samples have been laid out in rounds.
	requested uint64
	// live are the rounds laid out and not yet folded, oldest first.
	live            []*round
	roundsScheduled uint64
	roundsFolded    uint64
	// claimFrom is the first live round that may still have a free slot.
	claimFrom int
}

const (
	slotFree = iota
	// slotTaken: a worker is drawing it.
	slotTaken
	// slotDone: its value is in the round.
	slotDone
)

// round is a run of samples laid out together and folded together.
type round struct {
	slots []slot
	state []uint8
	// start is how many samples were laid out before this round.
	start uint64
	// Claims scan forward from claimed; every slot before it is taken or done.
	claimed int
	// folded is how many of its slots are folded already: more than none only
	// when a sample limit ended a call partway through the round.
	folded     int
	claimBatch int
	// lastInitial marks the round holding the last sample of the initial phase.
	lastInitial bool
}

// call is the state of one Run call.
type call struct {
	ctx      context.Context
	opts     Options
	limit    uint64
	deadline time.Time
	status   Status
	stopped  bool
	err      error
	// inFlight is how many workers are drawing samples, waiting how many are
	// waiting for work.
	inFlight int
	waiting  int
}

// NewRunner starts a run over numArms arms. Options.Similar and
// Options.Paired are fixed for the run, Options.SampleMinimum and
// Options.RoundSize by the first Run call; the rest are read on each call.
func NewRunner(numArms int, opts Options) *Runner {
	r := &Runner{s: newState(numArms, opts), numArms: numArms, initialPhase: true}
	r.cond = sync.NewCond(&r.mu)
	return r
}

// Samples is how many samples have been folded so far.
func (r *Runner) Samples() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return int(r.s.total)
}

// Run finds the best of numArms arms.
func Run(ctx context.Context, numArms int, opts Options, sample Sampler) (Result, error) {
	if numArms < 1 {
		return Result{}, errors.New("bai: no arms")
	}
	return NewRunner(numArms, opts).Run(ctx, opts, sample)
}

// scheduleRounds lays out every round the folds so far allow. The caller
// holds the lock. Since rounds fold in order, each is laid out from the
// statistics of one fixed fold, whatever the threads' timing.
func (r *Runner) scheduleRounds() {
	for {
		lag := uint64(adaptiveLag)
		if r.requested < r.initialLimit {
			lag = roundRobinLag
		}
		if r.roundsScheduled > r.roundsFolded+lag {
			return
		}
		r.scheduleRound()
	}
}

// scheduleRound lays out the next round: round-robin over the arms until the
// initial phase has folded, and by the top-two rule after that.
func (r *Runner) scheduleRound() {
	s := r.s
	n := r.roundSize
	// A round never straddles the end of the initial phase.
	if r.requested < r.initialLimit && r.initialLimit-r.requested < uint64(n) {
		n = int(r.initialLimit - r.requested)
	}
	rd := &round{start: r.requested, slots: make([]slot, 0, n), state: make([]uint8, n), claimBatch: 1}
	if r.initialPhase {
		// Iteration-major: consecutive slots are the same ordinal across
		// consecutive arms.
		for i := 0; i < n; i++ {
			arm := int((r.requested + uint64(i)) % uint64(r.numArms))
			rd.slots = append(rd.slots, slot{arm: arm, ordinal: s.arms[arm].numScheduled})
			s.arms[arm].numScheduled++
		}
		rd.claimBatch = roundRobinClaimBatch
	} else {
		rd.slots = s.scheduleTopTwo(n, rd.slots)
	}
	if r.requested < r.initialLimit && r.requested+uint64(n) >= r.initialLimit {
		rd.lastInitial = true
	}
	r.requested += uint64(n)
	r.roundsScheduled++
	r.live = append(r.live, rd)
}

// claim hands out the next free slots of the oldest round that has some: up
// to the round's claim batch, and none past the call's sample limit. The
// caller holds the lock.
func (r *Runner) claim(c *call) (*round, int, int) {
	for r.claimFrom < len(r.live) {
		rd := r.live[r.claimFrom]
		for rd.claimed < len(rd.slots) && rd.state[rd.claimed] != slotFree {
			rd.claimed++
		}
		if rd.claimed == len(rd.slots) {
			r.claimFrom++
			continue
		}
		first, n := rd.claimed, 0
		for n < rd.claimBatch && first+n < len(rd.slots) && rd.state[first+n] == slotFree {
			if c.limit > 0 && rd.start+uint64(first+n) >= c.limit {
				break
			}
			rd.state[first+n] = slotTaken
			n++
		}
		rd.claimed += n
		return rd, first, n
	}
	return nil, 0, 0
}

// fold commits every round that is complete, oldest first, checks the stop
// conditions after each and lays out the rounds the fold allows. A round the
// sample limit cuts short is folded up to the limit. The caller holds the
// lock.
func (r *Runner) fold(c *call) error {
	s := r.s
	for !c.stopped && len(r.live) > 0 {
		rd := r.live[0]
		n := len(rd.slots) - rd.folded
		partial := false
		if c.limit > 0 && c.limit-s.total < uint64(n) {
			n, partial = int(c.limit-s.total), true
		}
		for i := rd.folded; i < rd.folded+n; i++ {
			if rd.state[i] != slotDone {
				return nil
			}
		}
		// In slot order, so that the statistics are a function of the samples
		// and not of the order they finished in.
		for i := rd.folded; i < rd.folded+n; i++ {
			sl := &rd.slots[i]
			ordinal := s.arms[sl.arm].numSamples
			if err := s.arms[sl.arm].add(sl.value); err != nil {
				return err
			}
			s.record(sl.arm, ordinal, sl.value)
		}
		s.total += uint64(n)
		rd.folded += n
		best := s.updateAstar()
		if partial {
			// The rest of the round is drawn by the next call, which then
			// checks the other conditions as a run in one call would.
			c.status, c.stopped = StatusSampleLimit, true
			return nil
		}
		r.live = r.live[1:]
		r.roundsFolded++
		if r.claimFrom > 0 {
			r.claimFrom--
		}
		if rd.lastInitial {
			r.initialPhase = false
		}
		switch {
		case c.ctx.Err() != nil:
			c.status = StatusCanceled
		case !c.deadline.IsZero() && time.Now().After(c.deadline):
			c.status = StatusTimeout
		case r.initialPhase:
		case c.opts.Cutoff > 0 && (best <= c.opts.Cutoff || best >= 1-c.opts.Cutoff):
			c.status = StatusCutoff
		case s.thresholdReached():
			c.status = StatusThreshold
		case c.limit > 0 && s.total >= c.limit:
			c.status = StatusSampleLimit
		}
		if c.status != StatusNone {
			c.stopped = true
			return nil
		}
		r.scheduleRounds()
	}
	return nil
}

// work claims slots, draws them and folds, until the call stops.
func (r *Runner) work(c *call, worker int, sample Sampler) {
	var values [roundRobinClaimBatch]float64
	r.mu.Lock()
	defer r.mu.Unlock()
	for !c.stopped {
		rd, first, n := r.claim(c)
		if n == 0 {
			if c.inFlight == 0 {
				// Nothing is left to draw within the sample limit.
				c.status, c.stopped = StatusSampleLimit, true
				break
			}
			c.waiting++
			r.cond.Wait()
			c.waiting--
			continue
		}
		c.inFlight++
		r.mu.Unlock()
		var err error
		for i := 0; i < n && err == nil; i++ {
			values[i], err = sample(worker, rd.slots[first+i].arm, rd.slots[first+i].ordinal)
		}
		r.mu.Lock()
		c.inFlight--
		if err != nil {
			// The slots go back, to be drawn by a later call.
			for i := 0; i < n; i++ {
				rd.state[first+i] = slotFree
			}
			rd.claimed, r.claimFrom = 0, 0
			if !c.stopped {
				c.err, c.stopped = err, true
			}
			break
		}
		// A sample is a function of its arm and ordinal, so one that finishes
		// after the call has stopped is still good for the next call.
		for i := 0; i < n; i++ {
			rd.slots[first+i].value = values[i]
			rd.state[first+i] = slotDone
		}
		if !c.stopped {
			if err := r.fold(c); err != nil {
				c.err, c.stopped = err, true
			}
		}
		if !c.stopped {
			// Between folds too, so that a slow round cannot hold up the clock.
			switch {
			case c.ctx.Err() != nil:
				c.status, c.stopped = StatusCanceled, true
			case !c.deadline.IsZero() && time.Now().After(c.deadline):
				c.status, c.stopped = StatusTimeout, true
			}
		}
		if c.waiting > 0 {
			r.cond.Broadcast()
		}
	}
	if c.waiting > 0 {
		r.cond.Broadcast()
	}
}

// Run samples until a stop condition, continuing from earlier calls.
// opts.SampleLimit counts every sample of the run, earlier calls' included.
// With one thread the samples are drawn on the calling goroutine. The
// samples drawn, and so the result, are the same for any number of threads
// and however the run is split into calls by sample limits.
func (r *Runner) Run(ctx context.Context, opts Options, sample Sampler) (Result, error) {
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
	c := &call{ctx: ctx, opts: opts}
	if opts.SampleLimit > 0 {
		c.limit = uint64(opts.SampleLimit)
	}
	if opts.TimeLimit > 0 {
		c.deadline = time.Now().Add(opts.TimeLimit)
	}
	r.mu.Lock()
	s.opts = opts
	if !r.started {
		r.started = true
		r.roundSize = opts.RoundSize
		r.initialLimit = uint64(r.numArms) * uint64(opts.SampleMinimum)
	}
	switch {
	case ctx.Err() != nil:
		c.status, c.stopped = StatusCanceled, true
	case c.limit > 0 && s.total >= c.limit:
		c.status, c.stopped = StatusSampleLimit, true
	default:
		r.scheduleRounds()
		// A call that the last one's limit cut short mid-round may have
		// complete rounds to fold before it draws anything.
		if err := r.fold(c); err != nil {
			c.err, c.stopped = err, true
		}
	}
	r.mu.Unlock()
	if !c.stopped {
		if opts.Threads == 1 {
			r.work(c, 0, sample)
		} else {
			var wg sync.WaitGroup
			for w := 0; w < opts.Threads; w++ {
				wg.Add(1)
				go func(worker int) {
					defer wg.Done()
					r.work(c, worker, sample)
				}(w)
			}
			wg.Wait()
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if c.err != nil {
		if ctx.Err() == nil {
			return Result{}, c.err
		}
		c.status = StatusCanceled
	}
	r.status = c.status
	res := Result{Status: c.status, Best: s.astar, Samples: int(s.total),
		Means: make([]float64, r.numArms), Counts: make([]int, r.numArms)}
	for i := range s.arms {
		res.Means[i] = s.arms[i].mean
		res.Counts[i] = int(s.arms[i].numSamples)
	}
	if res.Best < 0 {
		res.Best = 0
	}
	return res, nil
}
