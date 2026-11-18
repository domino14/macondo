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
	// StatusBestArmLimit: the best arm has all the samples it may have.
	StatusBestArmLimit
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
	case StatusBestArmLimit:
		return "best-arm-limit"
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
	// A worker with nothing to claim draws ahead the samples the next rounds
	// are expected to hold (see speculation). speculationSlots is how many
	// slots ahead it looks, speculationsPerThread how many such samples are
	// held for each thread.
	speculationSlots      = 8 * DefaultRoundSize
	speculationsPerThread = 16
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
	// BestArmLimit stops the run once the best arm has this many samples;
	// 0 for no limit. Arms the run has not told apart by then are close: the
	// samples it takes to tell two arms apart go as one over the square of
	// the gap between them.
	BestArmLimit int
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
	// Held, if set, draws the samples in place of Run's sampler, and lets a
	// thread that has nothing to draw draw ahead the samples the schedule is
	// expected to ask for next (see speculation).
	Held HeldSampler
}

// Sampler draws arm's ordinal-th sample on worker (0 <= worker < Threads).
// It may be called concurrently for different workers. A sample with a
// shared seed per ordinal is what makes Options.Paired valid.
type Sampler func(worker, arm int, ordinal uint64) (float64, error)

// HeldSampler draws a sample as Sampler does, but the run may never use the
// sample: the run can stop before the sample's round is folded, and a sample
// drawn ahead of the schedule may never be asked for. Whatever else drawing
// a sample does that should happen only for samples the run uses (tallies,
// logs) the sampler holds back and does in commit, which is called when the
// sample is folded: at most once, in the order samples are folded, which is
// the same on any number of threads, and possibly from another goroutine. A
// nil commit means there is nothing to hold back.
type HeldSampler func(worker, arm int, ordinal uint64) (value float64, commit func(), err error)

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
	// commit is the held sampler's, called when the slot is folded.
	commit func()
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
	// initialLastRound is the number of the round holding the last sample of
	// the initial phase, once it is laid out.
	initialLastRound uint64
	initialLaidOut   bool

	// specs is the speculation cache and specIndex finds an entry by its
	// sample. Empty when the run has one thread.
	specs     []speculation
	specIndex map[specKey]int
	// expected are the slots the next rounds are expected to hold, which
	// change only when a round is laid out or folded: epoch counts those
	// events, and the expectation is worked out again when it has moved.
	// covered marks the expected slots that need no speculation: one is held
	// already, or the slot is past the sample limit. Claims take the others
	// in order from specCursor.
	epoch      uint64
	specEpoch  uint64
	specFresh  bool
	specValid  bool
	specCursor int
	expected   []slot
	covered    []bool
	// specDrawn counts the speculative samples started, specUsed those a
	// round took.
	specDrawn, specUsed int
}

const (
	specEmpty = iota
	specRunning
	specDone
)

type specKey struct {
	arm     int
	ordinal uint64
}

// speculation is a sample drawn ahead of the schedule by a worker that had
// nothing to claim. A sample is a function of its arm and ordinal, so it is
// the same whenever and on whichever thread it is drawn. A worker that would
// otherwise wait therefore draws the samples the next rounds will most
// likely hold. When a round is laid out, a slot whose sample is here is
// filled in and never claimed; one still being drawn is adopted, and the
// worker drawing it completes the slot. Samples no round ever asks for are
// dropped. Which samples are speculated on depends on thread timing; a
// slot's value never does, and so neither does the result.
type speculation struct {
	state   uint8
	arm     int
	ordinal uint64
	value   float64
	// adopted is set when a round laid out this sample's slot while it was
	// being drawn; round and slot say where.
	adopted bool
	round   *round
	slot    int
	// expected says it is one of the currently expected slots.
	expected bool
	// commit is the sampler's, handed to the slot that takes the sample.
	commit func()
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
		r.initialLastRound, r.initialLaidOut = r.roundsScheduled, true
	}
	for i := range rd.slots {
		r.matchSpeculation(rd, i)
	}
	r.requested += uint64(n)
	r.roundsScheduled++
	r.epoch++
	r.live = append(r.live, rd)
}

// findSpeculation is the cache entry holding arm's sample at ordinal, or -1.
func (r *Runner) findSpeculation(arm int, ordinal uint64) int {
	if i, ok := r.specIndex[specKey{arm, ordinal}]; ok {
		return i
	}
	return -1
}

// releaseSpeculation empties a cache entry.
func (r *Runner) releaseSpeculation(i int) {
	sp := &r.specs[i]
	delete(r.specIndex, specKey{sp.arm, sp.ordinal})
	*sp = speculation{}
}

// matchSpeculation hands the speculative sample for slot i of rd, a round
// being laid out, to that slot if there is one: a finished sample fills the
// slot in, one being drawn adopts it. Either way no worker will claim it.
func (r *Runner) matchSpeculation(rd *round, i int) {
	if len(r.specs) == 0 {
		return
	}
	si := r.findSpeculation(rd.slots[i].arm, rd.slots[i].ordinal)
	if si < 0 {
		return
	}
	sp := &r.specs[si]
	if sp.state == specDone {
		rd.slots[i].value, rd.slots[i].commit = sp.value, sp.commit
		rd.state[i] = slotDone
		r.releaseSpeculation(si)
		r.specUsed++
		return
	}
	rd.state[i] = slotTaken
	sp.adopted, sp.round, sp.slot = true, rd, i
}

// nextRoundIsRoundRobin says whether the next round to be laid out will be
// one of the initial phase's: whether the round whose fold lays it out folds
// before the initial phase's last round has.
func (r *Runner) nextRoundIsRoundRobin() bool {
	if !r.initialPhase {
		return false
	}
	if !r.initialLaidOut || r.roundsScheduled < adaptiveLag+1 {
		return true
	}
	return r.roundsScheduled-adaptiveLag-1 < r.initialLastRound
}

// expectSlots works out the slots the next rounds are expected to hold:
// exactly, while they are round-robin, and otherwise as the top-two rule
// would lay them out, round after round, from the statistics committed so
// far. It reports whether there is anything sound to predict from.
func (r *Runner) expectSlots() bool {
	s := r.s
	r.expected = r.expected[:0]
	if r.nextRoundIsRoundRobin() {
		counts := make(map[int]uint64)
		for i := 0; i < speculationSlots; i++ {
			arm := int((r.requested + uint64(i)) % uint64(r.numArms))
			r.expected = append(r.expected, slot{arm: arm, ordinal: s.arms[arm].numScheduled + counts[arm]})
			counts[arm]++
		}
		return true
	}
	// The top-two rule reads every arm's mean.
	if s.astar < 0 || s.total < uint64(r.numArms) {
		return false
	}
	// Each predicted round counts the ones predicted before it, as laying
	// them out will; the counts are put back afterwards.
	for len(r.expected) < speculationSlots {
		n := min(r.roundSize, speculationSlots-len(r.expected))
		r.expected = s.scheduleTopTwo(n, r.expected)
	}
	for i := range r.expected {
		s.arms[r.expected[i].arm].numScheduled--
	}
	return true
}

// refreshSpeculation works the expected slots out again, and which of them
// the cache already holds.
func (r *Runner) refreshSpeculation(c *call) {
	r.specEpoch, r.specFresh = r.epoch, true
	r.specCursor = 0
	r.specValid = r.expectSlots()
	for i := range r.specs {
		r.specs[i].expected = false
	}
	if !r.specValid {
		return
	}
	// Slots past the sample limit will not be drawn by this call.
	remaining := uint64(speculationSlots)
	if c.limit > 0 {
		remaining = 0
		if c.limit > r.requested {
			remaining = c.limit - r.requested
		}
	}
	r.covered = r.covered[:0]
	for i := range r.expected {
		covered := uint64(i) >= remaining
		if si := r.findSpeculation(r.expected[i].arm, r.expected[i].ordinal); si >= 0 {
			r.specs[si].expected = true
			covered = true
		}
		r.covered = append(r.covered, covered)
	}
}

// claimSpeculation reserves a cache entry for the first expected slot nobody
// has drawn or is drawing, and returns its index, or -1 if there is no such
// slot or no entry to put it in. A finished entry that is no longer expected
// is used again. The caller holds the lock.
func (r *Runner) claimSpeculation(c *call) int {
	if len(r.specs) == 0 || c.stopped || c.opts.Held == nil {
		return -1
	}
	if !r.specFresh || r.specEpoch != r.epoch {
		r.refreshSpeculation(c)
	}
	if !r.specValid {
		return -1
	}
	for r.specCursor < len(r.expected) && r.covered[r.specCursor] {
		r.specCursor++
	}
	if r.specCursor == len(r.expected) {
		return -1
	}
	entry := -1
	for i := range r.specs {
		if r.specs[i].state == specEmpty {
			entry = i
			break
		}
		if r.specs[i].state == specDone && !r.specs[i].expected && entry < 0 {
			entry = i
		}
	}
	if entry < 0 {
		return -1
	}
	if r.specs[entry].state != specEmpty {
		r.releaseSpeculation(entry)
	}
	e := r.expected[r.specCursor]
	r.specs[entry] = speculation{state: specRunning, arm: e.arm, ordinal: e.ordinal, expected: true}
	r.specIndex[specKey{e.arm, e.ordinal}] = entry
	r.covered[r.specCursor] = true
	r.specCursor++
	r.specDrawn++
	return entry
}

// finishSpeculation stores a speculative sample that has been drawn or, if a
// round adopted it meanwhile, completes that round's slot with it and folds.
// The caller holds the lock.
func (r *Runner) finishSpeculation(c *call, si int, value float64, commit func(), err error) {
	sp := r.specs[si]
	if err != nil {
		if sp.adopted {
			// The slot goes back, to be drawn by a later call.
			sp.round.state[sp.slot] = slotFree
			sp.round.claimed, r.claimFrom = 0, 0
		}
		r.releaseSpeculation(si)
		if !c.stopped {
			c.err, c.stopped = err, true
		}
		return
	}
	if !sp.adopted {
		r.specs[si].value, r.specs[si].state, r.specs[si].commit = value, specDone, commit
		return
	}
	// As with a claimed sample, one that finishes after the call has stopped
	// is still good for the next call.
	sp.round.slots[sp.slot].value, sp.round.slots[sp.slot].commit = value, commit
	sp.round.state[sp.slot] = slotDone
	r.releaseSpeculation(si)
	r.specUsed++
	if !c.stopped {
		if err := r.fold(c); err != nil {
			c.err, c.stopped = err, true
		}
	}
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
			if sl.commit != nil {
				sl.commit()
				sl.commit = nil
			}
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
		r.epoch++
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
		case s.bestArmLimitReached(c.opts.BestArmLimit):
			c.status = StatusBestArmLimit
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
	var commits [roundRobinClaimBatch]func()
	r.mu.Lock()
	defer r.mu.Unlock()
	for !c.stopped {
		rd, first, n := r.claim(c)
		if n == 0 {
			if si := r.claimSpeculation(c); si >= 0 {
				arm, ordinal := r.specs[si].arm, r.specs[si].ordinal
				c.inFlight++
				r.mu.Unlock()
				v, commit, err := c.opts.Held(worker, arm, ordinal)
				r.mu.Lock()
				c.inFlight--
				r.finishSpeculation(c, si, v, commit, err)
				r.checkClock(c)
				if c.waiting > 0 {
					r.cond.Broadcast()
				}
				continue
			}
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
			sl := &rd.slots[first+i]
			if c.opts.Held != nil {
				values[i], commits[i], err = c.opts.Held(worker, sl.arm, sl.ordinal)
			} else {
				values[i], commits[i] = 0, nil
				values[i], err = sample(worker, sl.arm, sl.ordinal)
			}
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
			rd.slots[first+i].value, rd.slots[first+i].commit = values[i], commits[i]
			rd.state[first+i] = slotDone
		}
		if !c.stopped {
			if err := r.fold(c); err != nil {
				c.err, c.stopped = err, true
			}
		}
		r.checkClock(c)
		if c.waiting > 0 {
			r.cond.Broadcast()
		}
	}
	if c.waiting > 0 {
		r.cond.Broadcast()
	}
}

// bestArmLimitReached reports whether the best arm has limit samples.
func (s *state) bestArmLimitReached(limit int) bool {
	return limit > 0 && s.astar >= 0 && s.arms[s.astar].numSamples >= uint64(limit)
}

// checkClock stops the call if its context is done or its time is up. It is
// checked between folds too, so that a slow round cannot hold up the clock.
func (r *Runner) checkClock(c *call) {
	if c.stopped {
		return
	}
	switch {
	case c.ctx.Err() != nil:
		c.status, c.stopped = StatusCanceled, true
	case !c.deadline.IsZero() && time.Now().After(c.deadline):
		c.status, c.stopped = StatusTimeout, true
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
	// The sample limit, which bounds what is worth speculating on, is the
	// call's.
	r.specFresh = false
	if want := min(speculationsPerThread*opts.Threads, speculationSlots); opts.Held != nil && opts.Threads > 1 && len(r.specs) < want {
		r.specs = append(r.specs, make([]speculation, want-len(r.specs))...)
		if r.specIndex == nil {
			r.specIndex = make(map[specKey]int)
		}
	}
	switch {
	case ctx.Err() != nil:
		c.status, c.stopped = StatusCanceled, true
	case c.limit > 0 && s.total >= c.limit:
		c.status, c.stopped = StatusSampleLimit, true
	case !r.initialPhase && s.bestArmLimitReached(opts.BestArmLimit):
		c.status, c.stopped = StatusBestArmLimit, true
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
