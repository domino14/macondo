package montecarlo

import (
	"fmt"
	"math"
	"sync"
)

// paired_stop.go: a stopping rule that drops plays by their paired
// differences with the leader.
//
// The sim runs as it does under the older rule: an iteration deals the
// opponent one rack and rolls out every play still in the sim against it.
// The rollouts of an iteration therefore share their luck, and the
// difference between two plays' results in an iteration is far less noisy
// than either result. The older rule compares each play's own confidence
// interval with the leader's; this one tests the mean of a play's
// differences with the leader, and can tell two plays apart in a fraction of
// the rollouts.
//
// Unlike best-arm identification (bai_sim.go) it keeps the rollouts of an
// iteration together, so the move generators' memory of best plays by rack
// (movegen.SetBestMoveCarry) keeps working, and threads need no rounds.

const (
	// DefaultPairedMinIterations is how many iterations run before any play
	// is dropped. The test estimates a variance, and with fewer iterations
	// it drops the best play too often.
	DefaultPairedMinIterations = 128
	// pairedCheckInterval is how many iterations pass between tests.
	pairedCheckInterval = 16
	// DefaultPairedCapPerPlay is the budget of a sim, in rollouts per play
	// on average.
	DefaultPairedCapPerPlay = 1500
)

type pairedState struct {
	on           bool
	minIters     int
	capPerPlay   int
	spreadWeight float64

	mu sync.Mutex
	// arms are the sim's plays when it started; a play's arm is its index.
	arms    []*SimmedPlay
	similar []bool
	// sum[i] is the sum of arm i's rewards and cross[i*k+j], i <= j, the sum
	// of the products of arms i and j's rewards of the same iteration.
	sum   []float64
	cross []float64
	// beaten marks a play the test would have dropped but which the caller
	// asked to keep: it goes on being rolled out, and does not hold the sim
	// open.
	beaten []bool
	// iters is the iterations folded in, rollouts their rollouts.
	iters    int
	rollouts int
	stopped  bool
	status   string
	winner   *SimmedPlay

	// By thread, the iteration in progress: the arms rolled out and their
	// rewards.
	threadArms    [][]int
	threadRewards [][]float64
}

// SetPairedStop switches the paired stopping rule on or off for sims that
// have a stopping condition. Best-arm identification, if it is on too,
// takes precedence. Call it before PrepareSim.
func (s *Simmer) SetPairedStop(on bool) {
	s.paired.on = on
}

// SetPairedMinIterations sets how many iterations run before any play is
// dropped; 0 for the default.
func (s *Simmer) SetPairedMinIterations(n int) {
	s.paired.minIters = n
}

// SetPairedCapPerPlay sets the most rollouts the sim makes, as an average
// per candidate play; 0 for the default.
func (s *Simmer) SetPairedCapPerPlay(n int) {
	s.paired.capPerPlay = n
}

// SetPairedSpreadWeight sets the weight of the spread term in the reward
// the rule compares plays by (see baiReward); a negative weight means none,
// 0 the default.
func (s *Simmer) SetPairedSpreadWeight(w float64) {
	s.paired.spreadWeight = w
}

// PairedStatus is why the last paired run stopped: "threshold" when one
// play was left, "sample-limit" when the budget ran out, "" when it was
// stopped from outside or has not stopped.
func (s *Simmer) PairedStatus() string {
	s.paired.mu.Lock()
	defer s.paired.mu.Unlock()
	return s.paired.status
}

// PairedRollouts is the number of rollouts the paired run has made.
func (s *Simmer) PairedRollouts() int {
	s.paired.mu.Lock()
	defer s.paired.mu.Unlock()
	return s.paired.rollouts
}

func (s *Simmer) usesPaired() bool {
	return s.paired.on && !s.usesBAI() && s.autostopper.stoppingCondition != StopNone
}

func (s *Simmer) resetPaired() {
	p := &s.paired
	p.mu.Lock()
	defer p.mu.Unlock()
	p.arms, p.winner, p.status, p.stopped = nil, nil, "", false
	p.iters, p.rollouts = 0, 0
}

// startPaired fixes the sim's arms. It runs on the first simulate after
// PrepareSim.
func (s *Simmer) startPaired() {
	p := &s.paired
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.arms == nil {
		s.simmedPlays.RLock()
		p.arms = append([]*SimmedPlay(nil), s.simmedPlays.plays...)
		s.simmedPlays.RUnlock()
		k := len(p.arms)
		keys := make([]string, k)
		for i, a := range p.arms {
			a.arm = i
			keys[i] = similarPlayKey(a.play)
		}
		p.similar = make([]bool, k*k)
		for i := 0; i < k; i++ {
			for j := i + 1; j < k; j++ {
				// A play the caller asked to keep has to be beaten on its own.
				if p.arms[i].unignorable || p.arms[j].unignorable {
					continue
				}
				if keys[i] == keys[j] {
					p.similar[i*k+j], p.similar[j*k+i] = true, true
				}
			}
		}
		p.sum = make([]float64, k)
		p.cross = make([]float64, k*k)
		p.beaten = make([]bool, k)
	}
	// A sim that is continued gets a new budget, as it does under the other
	// rules.
	p.stopped, p.status = false, ""
	p.rollouts = 0
	k := len(p.arms)
	if len(p.threadArms) < s.threads {
		p.threadArms = make([][]int, s.threads)
		p.threadRewards = make([][]float64, s.threads)
	}
	for t := 0; t < s.threads; t++ {
		p.threadArms[t] = make([]int, 0, k)
		p.threadRewards[t] = make([]float64, 0, k)
	}
}

// pairedNote records a rollout of the iteration thread is running.
func (s *Simmer) pairedNote(thread int, play *SimmedPlay, win, eq float64) {
	p := &s.paired
	w := p.spreadWeight
	if w == 0 {
		w = DefaultBAISpreadWeight
	}
	p.threadArms[thread] = append(p.threadArms[thread], play.arm)
	p.threadRewards[thread] = append(p.threadRewards[thread],
		baiReward(win, eq+float64(s.initialSpread), w))
}

// pairedFold counts the iteration thread has just finished and, when a test
// is due, runs it. It reports whether the sim should stop, and, to the
// thread that stopped it, why.
func (s *Simmer) pairedFold(thread int) (bool, string) {
	p := &s.paired
	arms, rewards := p.threadArms[thread], p.threadRewards[thread]
	p.threadArms[thread], p.threadRewards[thread] = arms[:0], rewards[:0]

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.stopped {
		// Another thread ended the sim while this iteration ran; it is not
		// counted, so that the plays' tallies stay those the sim stopped on.
		return true, ""
	}
	k := len(p.arms)
	for a, i := range arms {
		xi := rewards[a]
		p.sum[i] += xi
		for b := a; b < len(arms); b++ {
			j := arms[b]
			p.cross[min(i, j)*k+max(i, j)] += xi * rewards[b]
		}
	}
	p.iters++
	p.rollouts += len(arms)

	minIters := p.minIters
	if minIters <= 0 {
		minIters = DefaultPairedMinIterations
	}
	if p.iters >= minIters && p.iters%pairedCheckInterval == 0 {
		s.pairedTest()
	}
	if !p.stopped {
		capPer := p.capPerPlay
		if capPer <= 0 {
			capPer = DefaultPairedCapPerPlay
		}
		if p.rollouts >= capPer*k {
			p.stopped, p.status = true, "sample-limit"
			p.winner = s.pairedLeader()
		}
	}
	return p.stopped, p.status
}

// pairedFinish names the sim's pick when the sim was ended from outside
// (the clock, the caller's iteration limit, the user).
func (s *Simmer) pairedFinish() {
	p := &s.paired
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.winner == nil && p.iters > 0 {
		p.winner = s.pairedLeader()
	}
	p.stopped = true
}

// pairedLeader is the play still in the sim with the best mean reward. The
// caller holds the lock.
func (s *Simmer) pairedLeader() *SimmedPlay {
	p := &s.paired
	var lead *SimmedPlay
	for _, a := range p.arms {
		if a.ignore.Load() || a.trimmed.Load() {
			continue
		}
		if lead == nil || p.sum[a.arm] > p.sum[lead.arm] {
			lead = a
		}
	}
	return lead
}

// pairedTest drops every play whose mean difference with the leader is
// above zero by the generalized likelihood ratio threshold best-arm
// identification uses, log((log n + 1) / delta). The caller holds the lock.
func (s *Simmer) pairedTest() {
	p := &s.paired
	k := len(p.arms)
	n := float64(p.iters)
	threshold := math.Log((math.Log(n) + 1) / baiDelta(s.autostopper.stoppingCondition))
	lead := s.pairedLeader()
	if lead == nil {
		p.stopped, p.status = true, "threshold"
		return
	}
	l := lead.arm
	open := 0
	for _, a := range p.arms {
		i := a.arm
		if i == l || a.ignore.Load() || a.trimmed.Load() {
			continue
		}
		lo, hi := min(i, l), max(i, l)
		d := (p.sum[l] - p.sum[i]) / n
		variance := (p.cross[l*k+l] + p.cross[i*k+i] - 2*p.cross[lo*k+hi] - n*d*d) / (n - 1)
		beaten := false
		switch {
		case p.similar[l*k+i]:
			// The same tiles in the same place for the same score: the
			// leader stands in for it.
			beaten = true
		case variance <= 1e-12:
			// The rollouts cannot tell it from the leader.
			beaten = true
		default:
			beaten = n*d*d/(2*variance) > threshold
		}
		switch {
		case !beaten:
			open++
		case a.unignorable:
			p.beaten[i] = true
		default:
			a.ignore.Store(true)
		}
		if !beaten && p.beaten[i] {
			// Kept at the caller's request and already beaten once.
			open--
		}
	}
	if open == 0 {
		p.stopped, p.status = true, "threshold"
		p.winner = lead
	}
}

// StopRule is how a sim with a stopping condition chooses what to roll out
// and when to stop.
type StopRule int

const (
	// StopRuleClassic rolls out every play still in the sim once an
	// iteration and drops a play when its confidence interval is below the
	// leader's.
	StopRuleClassic StopRule = iota
	// StopRuleBAI is best-arm identification (bai_sim.go).
	StopRuleBAI
	// StopRulePaired is the classic rule's sampling with a test on paired
	// differences (this file).
	StopRulePaired
)

// ParseStopRule reads a rule's name: classic, bai or paired.
func ParseStopRule(name string) (StopRule, error) {
	switch name {
	case "classic":
		return StopRuleClassic, nil
	case "bai":
		return StopRuleBAI, nil
	case "paired":
		return StopRulePaired, nil
	}
	return StopRuleClassic, fmt.Errorf("stop rule %q is not one of classic, bai, paired", name)
}

// SetStopRule picks the rule for sims that have a stopping condition. Call
// it before PrepareSim.
func (s *Simmer) SetStopRule(r StopRule) {
	s.bai.on = r == StopRuleBAI
	s.paired.on = r == StopRulePaired
}

// SetStopCapPerPlay sets the most rollouts a best-arm or paired sim makes,
// as an average per candidate play; 0 for the rule's default.
func (s *Simmer) SetStopCapPerPlay(n int) {
	s.bai.capPerPlay = n
	s.paired.capPerPlay = n
}
