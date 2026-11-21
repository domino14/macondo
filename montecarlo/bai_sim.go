package montecarlo

import (
	"context"
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand/v2"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"
	"gopkg.in/yaml.v3"

	"github.com/domino14/macondo/montecarlo/bai"
	"github.com/domino14/macondo/move"
)

// A sim can be run as a best-arm identification problem, as MAGPIE runs its
// sims: every candidate play is an arm and a sample of an arm is one
// rollout of that play. The sim then chooses which plays to roll out --
// mostly the leader and its closest challenger -- instead of rolling every
// surviving play out once an iteration, and stops when the leader has been
// told apart from every other play (see the bai package).
//
// Samples are drawn on common random numbers: a play's n-th rollout deals
// the opponent the rack, and the bag the draws, that every other play's
// n-th rollout gets, so the stopping test can use how the plays' results
// move together.
//
// It is used when it has been switched on (SetBAI) and the sim has a
// stopping condition; a sim without one rolls every play out a fixed
// number of times as before.

const (
	// DefaultBAIMinSamples is how many rollouts every play gets before any
	// play is preferred or the sim may stop.
	DefaultBAIMinSamples = 128
	// DefaultBAICapPerPlay is the budget of a sim, in rollouts per play on
	// average. With 100 plays, at 2 to 7 plies, the plays picked stop
	// getting measurably better somewhere between 1,500 and 2,500.
	DefaultBAICapPerPlay = 1500
	// DefaultBAISpreadWeight is the weight of the spread term in the reward
	// (the win probability's weight is 1).
	DefaultBAISpreadWeight = 0.1
	// baiSpreadScale is the spread the reward's sigmoid is scaled by.
	baiSpreadScale = 100.0
	// baiCutoff stops the sim once the best play's reward is within it of 0
	// or 1: a decided game (MAGPIE's default).
	baiCutoff = 0.005 / 100
	// baiMinCycles is the fewest samples a play gets in a bag-cycled sim, so
	// that its variance is never estimated from one.
	baiMinCycles = 2
)

// baiState is what a sim keeps for best-arm identification.
type baiState struct {
	on           bool
	minSamples   int
	spreadWeight float64
	// capPerPlay is the most rollouts the sim makes, as an average per
	// play; 0 for the stopping rule's own cap on iterations.
	capPerPlay int
	// bestPlayCap stops the sim once the leading play has been rolled out
	// this many times; 0 for no such stop.
	bestPlayCap int

	// bagCycle makes a sample a whole cycle of rollouts (see SetBAIBagCycle),
	// rotations how many rollouts that is in this sim.
	bagCycle  bool
	rotations int

	runner *bai.Runner
	arms   []*SimmedPlay
	// toppedUp is, for each arm, how many of its first samples are in its
	// statistics already because the sim topped it up (baiTopUp); the
	// schedule's own samples at those ordinals are not counted again.
	toppedUp []uint64
	// folded counts the samples folded so far, for the sim's running
	// iteration count.
	folded atomic.Uint64
	// rngs are the threads' random sources (baiRNG).
	rngs []baiRNGPair
	// topUpStart and topUpRewards are the rewards of each arm's top-up
	// samples: those of its ordinals from topUpStart on.
	topUpStart   []uint64
	topUpRewards [][]float64
	seed         [32]byte
	// winner is the best arm when the last run stopped.
	winner *SimmedPlay
	status bai.Status
	logMu  sync.Mutex
}

// SetBAI switches best-arm identification on or off for sims that have a
// stopping condition. Call it before PrepareSim.
func (s *Simmer) SetBAI(on bool) {
	s.bai.on = on
}

// SetBAIMinSamples sets how many rollouts every play gets before the sim
// chooses among them; 0 for the default.
func (s *Simmer) SetBAIMinSamples(n int) {
	s.bai.minSamples = n
}

// SetBAISpreadWeight sets the weight of the spread term in the reward the
// sim ranks plays by while it runs, win probability having weight 1; a
// negative weight means none. 0 is the default.
func (s *Simmer) SetBAISpreadWeight(w float64) {
	s.bai.spreadWeight = w
}

// SetBAICapPerPlay sets the most rollouts the sim makes, as an average per
// candidate play; 0 for the stopping rule's cap on iterations.
func (s *Simmer) SetBAICapPerPlay(n int) {
	s.bai.capPerPlay = n
}

// SetBAIBestPlayCap stops a best-arm sim once the play in the lead has been
// rolled out n times, whether or not it has been told apart from the rest;
// 0 for no such stop. Plays that many rollouts have not told apart are
// close: the rollouts it takes to tell two plays apart go as one over the
// square of the gap between them, so the last fraction of a point of win
// probability costs most of a long sim. The sim's budget (SetBAICapPerPlay)
// still applies.
func (s *Simmer) SetBAIBestPlayCap(n int) {
	s.bai.bestPlayCap = n
}

// SetBAIBagCycle makes each sample of a best-arm sim a cycle of rollouts
// rather than one: the unseen tiles are shuffled once, and the play is
// rolled out once for each way of moving that order round by a rack, so
// that every unseen tile starts on the opponent's rack once. A sample is the
// mean of its rollouts. The opponent's racks of a cycle are spread evenly
// over the unseen tiles, which takes much of the luck of the draw out of a
// sample (MAGPIE's bag cycling). Budgets and minimums still count rollouts.
// It does nothing when the opponent's rack is known or inferred. Call it
// before PrepareSim.
func (s *Simmer) SetBAIBagCycle(on bool) {
	s.bai.bagCycle = on
}

// BAIStatus is why the last best-arm identification run stopped.
func (s *Simmer) BAIStatus() string {
	return s.bai.status.String()
}

// usesBAI reports whether a run is a best-arm identification run.
func (s *Simmer) usesBAI() bool {
	return s.bai.on && s.autostopper.stoppingCondition != StopNone
}

// baiReward is the number a rollout is worth to the sim: its win
// probability, blended with a sigmoid of where the spread ended up. The
// spread term ranks the plays of a decided game, where every win
// probability is the same, and breaks near-ties in favor of spread.
func baiReward(winProb, finalSpread, spreadWeight float64) float64 {
	if spreadWeight <= 0 {
		return winProb
	}
	x := finalSpread / baiSpreadScale
	var sig float64
	if x >= 0 {
		sig = 1 / (1 + math.Exp(-x))
	} else {
		e := math.Exp(x)
		sig = e / (1 + e)
	}
	return (winProb + spreadWeight*sig) / (1 + spreadWeight)
}

func baiDelta(sc StoppingCondition) float64 {
	switch sc {
	case Stop90:
		return 0.10
	case Stop95:
		return 0.05
	case Stop98:
		return 0.02
	case Stop99:
		return 0.01
	case Stop999:
		return 0.001
	}
	return 0
}

// similarPlayKey is equal for two plays the stopping rules count as the
// same: plays from the same square in the same direction that use the same
// tiles. A play like that never has to be told apart from the leader.
func similarPlayKey(m *move.Move) string {
	r, c, v := m.CoordsAndVertical()
	tiles := append([]byte(nil), m.Tiles().ToByteArr()...)
	sort.Slice(tiles, func(i, j int) bool { return tiles[i] < tiles[j] })
	key := make([]byte, 0, 8+len(tiles))
	key = append(key, byte(m.Action()), byte(r), byte(c), byte(m.TilesPlayed()))
	if v {
		key = append(key, 1)
	} else {
		key = append(key, 0)
	}
	return string(append(key, tiles...))
}

// startBAI fixes the sim's arms and the seed its samples are drawn from. It
// runs on the first simulate after PrepareSim.
func (s *Simmer) startBAI() {
	b := &s.bai
	s.simmedPlays.RLock()
	b.arms = b.arms[:0]
	for _, p := range s.simmedPlays.plays {
		if !p.ignore.Load() {
			b.arms = append(b.arms, p)
		}
	}
	s.simmedPlays.RUnlock()
	n := len(b.arms)
	b.folded.Store(0)
	b.toppedUp = make([]uint64, n)
	b.topUpStart = make([]uint64, n)
	b.topUpRewards = make([][]float64, n)
	keys := make([]string, n)
	for i, a := range b.arms {
		keys[i] = similarPlayKey(a.play)
	}
	similar := make([]bool, n*n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			// A play the caller asked to keep has to be beaten on its own.
			if b.arms[i].unignorable || b.arms[j].unignorable {
				continue
			}
			if keys[i] == keys[j] {
				similar[i*n+j], similar[j*n+i] = true, true
			}
		}
	}
	b.runner = bai.NewRunner(n, bai.Options{
		Similar: func(x, y int) bool { return similar[x*n+y] },
		Paired:  true,
	})
	b.rotations = 1
	if b.bagCycle && len(s.knownOppRack) == 0 && s.inferenceMode == InferenceOff {
		b.rotations = s.origGame.BagCycleRotations(1 - s.origGame.PlayerOnTurn())
	}
	seed := s.origGame.Seed()
	if seed == ([32]byte{}) {
		crand.Read(b.seed[:])
	} else {
		var buf [40]byte
		copy(buf[:32], seed[:])
		binary.BigEndian.PutUint64(buf[32:], uint64(s.origGame.Turn()))
		b.seed = sha256.Sum256(buf[:])
	}
}

// baiRNG is the random source for sample number ordinal, set as thread's.
// A thread has two and takes them in turn, starting each again from the
// sample's seed: the bag still holds the last sample's source and draws from
// it while the old rack is put back, before it is given this one, and those
// draws must not come out of this sample's stream.
func (s *Simmer) baiRNG(thread int, ordinal uint64) *rand.Rand {
	var buf [40]byte
	copy(buf[:32], s.bai.seed[:])
	binary.BigEndian.PutUint64(buf[32:], ordinal)
	h := sha256.Sum256(buf[:])
	pair := &s.bai.rngs[thread]
	pair.next ^= 1
	if pair.rngs[pair.next] == nil {
		pair.srcs[pair.next] = rand.NewChaCha8(h)
		pair.rngs[pair.next] = rand.New(pair.srcs[pair.next])
	} else {
		pair.srcs[pair.next].Seed(h)
	}
	s.rngs[thread] = pair.rngs[pair.next]
	return pair.rngs[pair.next]
}

// baiRNGPair is a thread's two sources, the generators they draw from, and
// which was handed out last.
type baiRNGPair struct {
	srcs [2]*rand.ChaCha8
	rngs [2]*rand.Rand
	next int
}

// simBAI runs the sim until its stopping rule, its budget or the context
// says to stop.
func (s *Simmer) simBAI(ctx context.Context, run simRun) {
	logger := zerolog.Ctx(ctx)
	b := &s.bai
	if b.runner == nil {
		s.startBAI()
	}
	numArms := len(b.arms)
	threads := s.threads
	if run.inline || threads < 1 {
		threads = 1
	}
	if len(s.rngs) < threads {
		s.rngs = make([]*rand.Rand, threads)
	}
	if len(b.rngs) < threads {
		b.rngs = make([]baiRNGPair, threads)
	}
	minimum := b.minSamples
	if minimum <= 0 {
		minimum = DefaultBAIMinSamples
	}
	// A sample is rot rollouts; the minimum, the budget and the round size
	// count rollouts, and are turned into samples here.
	rot := b.rotations
	samplesFor := func(rollouts int) int { return (rollouts + rot - 1) / rot }
	opts := bai.Options{
		Delta:         baiDelta(s.autostopper.stoppingCondition),
		SampleMinimum: samplesFor(minimum),
		Cutoff:        baiCutoff,
		Threads:       threads,
		TimeLimit:     s.timeLimit,
		RoundSize:     max((bai.DefaultRoundSize+rot/2)/rot, 1),
	}
	if b.bestPlayCap > 0 {
		opts.BestArmLimit = samplesFor(b.bestPlayCap)
	}
	if rot > 1 && opts.SampleMinimum < baiMinCycles {
		opts.SampleMinimum = baiMinCycles
	}
	// An iteration is one rollout of every play.
	switch {
	case run.maxIters > 0:
		opts.SampleLimit = b.runner.Samples() + samplesFor(run.maxIters*numArms)
	case b.capPerPlay > 0:
		opts.SampleLimit = samplesFor(b.capPerPlay * numArms)
	default:
		opts.SampleLimit = samplesFor(DefaultBAICapPerPlay * numArms)
	}
	// A sample's rollouts go into its play's statistics and the log when the
	// sample is folded, not when it is drawn: samples of rounds the sim stops
	// before folding, and samples a thread with nothing to draw drew ahead of
	// the schedule in vain, leave no trace, and the statistics are the same on
	// any number of threads.
	opts.Held = func(worker, arm int, ordinal uint64) (float64, bai.Held, error) {
		held := baiHeldPool.Get().(*baiHeld)
		held.reset(b.arms[arm])
		held.sim, held.arm, held.ordinal = s, arm, ordinal
		v, err := s.baiSample(worker, b.arms[arm], ordinal, held)
		return v, held, err
	}
	tstart := time.Now()
	res, err := b.runner.Run(ctx, opts, nil)
	s.iterationCount.Store(uint64(b.runner.Samples() * rot / max(numArms, 1)))
	if err != nil {
		logger.Err(err).Msg("error simming; stopping")
		return
	}
	b.status = res.Status
	b.winner = b.arms[res.Best]
	if res.Status != bai.StatusCanceled {
		var deadline time.Time
		if s.timeLimit > 0 {
			deadline = tstart.Add(s.timeLimit)
		}
		if n := s.baiTopUp(ctx, res, threads, deadline); n > 0 {
			s.event(logger, run.quiet).Int("rollouts", n*rot).Msg("sim-topped-up-kept-plays")
		}
	}
	s.event(logger, run.quiet).Str("status", res.Status.String()).Int("rollouts", res.Samples*rot).
		Str("best", b.winner.play.ShortDescription()).Msg("sim-stopped")
}

// baiTopUp gives every play the caller asked to keep (AvoidPruningMoves) as
// many samples as the most sampled play has, once the sim has stopped, so
// that its numbers are as good as the leaders' (as MAGPIE's
// sim_unpruned_to_winner does, which tops them up to the winner's count).
// The samples are the play's next ones, on the same luck as the winner's at
// the same ordinals. They go into the play's statistics only: the stopping
// rule's own count of the play is left alone, and when the sim is continued
// and the rule draws those samples again they are not counted twice. It
// returns how many samples it drew; the context or the deadline cut it short.
// A kept play that ends up ahead of the winner becomes the winner.
func (s *Simmer) baiTopUp(ctx context.Context, res bai.Result, threads int, deadline time.Time) int {
	b := &s.bai
	// The most any play has: nearly always the winner's count, but a
	// challenger that led for most of the sim can have more.
	target := uint64(0)
	for _, c := range res.Counts {
		target = max(target, uint64(c))
	}
	type job struct {
		arm     int
		ordinal uint64
	}
	// Ordinal by ordinal across the kept plays, so that a top-up cut short
	// leaves each of them a whole prefix of its samples.
	var jobs []job
	next := make([]uint64, len(b.arms))
	open := 0
	for i, a := range b.arms {
		next[i] = max(uint64(res.Counts[i]), b.toppedUp[i])
		if a.unignorable && next[i] < target {
			open++
		} else {
			next[i] = target
		}
	}
	for open > 0 {
		for i := range b.arms {
			if next[i] >= target {
				continue
			}
			jobs = append(jobs, job{i, next[i]})
			if next[i]++; next[i] == target {
				open--
			}
		}
	}
	for i := range b.arms {
		if c := uint64(res.Counts[i]); c >= b.toppedUp[i] {
			// The schedule has caught up with the last top-up.
			b.topUpStart[i], b.topUpRewards[i] = c, b.topUpRewards[i][:0]
		}
	}
	// A kept play that its top-up samples put ahead of the winner, by the
	// reward the sim ran on, is the winner.
	defer func() {
		best, bestMean := res.Best, res.Means[res.Best]
		for i, a := range b.arms {
			if !a.unignorable || len(b.topUpRewards[i]) == 0 {
				continue
			}
			sum, n := res.Means[i]*float64(res.Counts[i]), float64(res.Counts[i])
			for k, v := range b.topUpRewards[i] {
				if b.topUpStart[i]+uint64(k) >= uint64(res.Counts[i]) {
					sum, n = sum+v, n+1
				}
			}
			if mean := sum / n; mean > bestMean {
				best, bestMean = i, mean
			}
		}
		b.winner = b.arms[best]
	}()
	if len(jobs) == 0 {
		return 0
	}
	rewards := make([]float64, len(jobs))
	helds := make([]*baiHeld, len(jobs))
	errs := make([]error, len(jobs))
	var claimed atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < threads; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for {
				if ctx.Err() != nil || (!deadline.IsZero() && time.Now().After(deadline)) {
					return
				}
				j := int(claimed.Add(1)) - 1
				if j >= len(jobs) {
					return
				}
				held := baiHeldPool.Get().(*baiHeld)
				held.reset(b.arms[jobs[j].arm])
				rewards[j], errs[j] = s.baiSample(worker, b.arms[jobs[j].arm], jobs[j].ordinal, held)
				helds[j] = held
			}
		}(w)
	}
	wg.Wait()
	// In the jobs' order, whatever thread drew them, so that the statistics
	// do not depend on the number of threads.
	done := 0
	for j, held := range helds {
		if held == nil || errs[j] != nil {
			if errs[j] != nil {
				zerolog.Ctx(ctx).Err(errs[j]).Msg("error topping up a kept play")
			}
			break
		}
		s.baiCommit(held)
		baiHeldPool.Put(held)
		b.toppedUp[jobs[j].arm] = jobs[j].ordinal + 1
		b.topUpRewards[jobs[j].arm] = append(b.topUpRewards[jobs[j].arm], rewards[j])
		done++
	}
	return done
}

// baiHeld is what a sample holds back until the sim folds it: its rollouts'
// statistics and its log entry.
type baiHeld struct {
	tally rolloutTally
	log   []byte
	// The sample it belongs to, for a sample the runner commits.
	sim     *Simmer
	arm     int
	ordinal uint64
}

// Commit is the runner's call when the sample is folded. A sample the top-up
// drew already is in the statistics.
func (h *baiHeld) Commit() {
	b := &h.sim.bai
	if h.ordinal >= b.toppedUp[h.arm] {
		h.sim.baiCommit(h)
	}
	// Kept up as the sim runs, for whoever asks how far along it is: an
	// iteration is one rollout of every play.
	n := b.folded.Add(1)
	h.sim.iterationCount.Store(n * uint64(max(b.rotations, 1)) / uint64(max(len(b.arms), 1)))
	h.Discard()
}

// Discard is the runner's call for a sample it will not use.
func (h *baiHeld) Discard() {
	h.sim = nil
	baiHeldPool.Put(h)
}

var baiHeldPool = sync.Pool{New: func() any { return new(baiHeld) }}

func (h *baiHeld) reset(play *SimmedPlay) {
	t := &h.tally
	t.play = play
	t.plies, t.scores, t.bingos = t.plies[:0], t.scores[:0], t.bingos[:0]
	t.equities, t.leftovers, t.wins = t.equities[:0], t.leftovers[:0], t.wins[:0]
	h.log = h.log[:0]
}

// baiCommit adds a held sample to its play's statistics and to the log.
func (s *Simmer) baiCommit(held *baiHeld) {
	held.tally.commit()
	if len(held.log) > 0 {
		s.bai.logMu.Lock()
		defer s.bai.logMu.Unlock()
		s.logStream.Write(held.log)
	}
}

// baiSample draws sample number ordinal of play on thread's game copy and
// returns its reward: deal the opponent the sample's rack, roll the play
// out, and put the game back. With held, the play's statistics and the log
// are left alone and what would go to them is put in held.
func (s *Simmer) baiSample(thread int, play *SimmedPlay, ordinal uint64, held *baiHeld) (float64, error) {
	var tally *rolloutTally
	if held != nil {
		tally = &held.tally
	}
	g := s.gameCopies[thread]
	opp := (s.initialPlayer + 1) % g.NumPlayers()
	w := s.bai.spreadWeight
	if w == 0 {
		w = DefaultBAISpreadWeight
	}
	if s.bai.rotations > 1 {
		// A cycle: the same shuffle of the unseen tiles, moved round by a
		// rack for each rollout. The sample is the rollouts' mean.
		sum := 0.0
		for rotation := 0; rotation < s.bai.rotations; rotation++ {
			rng := s.baiRNG(thread, ordinal)
			if err := g.SetRandomRackCycled(opp, rng, rotation); err != nil {
				return 0, err
			}
			var logIter *LogIteration
			if s.logStream != nil && rotation == 0 {
				logIter = &LogIteration{Iteration: int(ordinal), Thread: thread}
			}
			win, eq := s.rollout(g, thread, s.maxPlies, play, logIter, tally)
			if logIter != nil {
				if err := s.baiWriteLog(logIter, held); err != nil {
					return 0, err
				}
			}
			sum += baiReward(win, eq+float64(s.initialSpread), w)
		}
		return sum / float64(s.bai.rotations), nil
	}
	// The opponent's rack, when it is inferred, is drawn from the sample's
	// source too.
	rng := s.baiRNG(thread, ordinal)
	rackToSet, err := s.oppRackFor(thread)
	if err != nil {
		return 0, err
	}
	if err := g.SetRandomRackSeeded(opp, rackToSet, rng); err != nil {
		return 0, err
	}
	var logIter *LogIteration
	if s.logStream != nil {
		logIter = &LogIteration{Iteration: int(ordinal), Thread: thread}
		if len(rackToSet) > 0 {
			logIter.InferredRack = tilemapping.MachineWord(rackToSet).UserVisible(g.Alphabet())
		}
	}
	win, eq := s.rollout(g, thread, s.maxPlies, play, logIter, tally)
	if logIter != nil {
		if err := s.baiWriteLog(logIter, held); err != nil {
			return 0, err
		}
	}
	return baiReward(win, eq+float64(s.initialSpread), w), nil
}

// baiWriteLog writes one sample's log entry: JSON lines for a heat map (up
// to MaxHeatMapIterations of each play's samples), YAML otherwise. With
// held, the entry is kept there instead.
func (s *Simmer) baiWriteLog(logIter *LogIteration, held *baiHeld) error {
	var out []byte
	var err error
	if s.collectHeatMap {
		if logIter.Iteration >= MaxHeatMapIterations {
			return nil
		}
		out, err = json.Marshal(logIter)
		out = append(out, '\n')
	} else {
		out, err = yaml.Marshal([]LogIteration{*logIter})
	}
	if err != nil {
		return err
	}
	if held != nil {
		held.log = append(held.log, out...)
		return nil
	}
	s.bai.logMu.Lock()
	defer s.bai.logMu.Unlock()
	_, err = s.logStream.Write(out)
	return err
}
