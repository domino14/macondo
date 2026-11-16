package montecarlo

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"math"
	"sort"
	"sync"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"
	"gopkg.in/yaml.v3"
	"lukechampine.com/frand"

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
	// DefaultBAISpreadWeight is the weight of the spread term in the reward
	// (the win probability's weight is 1).
	DefaultBAISpreadWeight = 0.1
	// baiSpreadScale is the spread the reward's sigmoid is scaled by.
	baiSpreadScale = 100.0
	// baiCutoff stops the sim once the best play's reward is within it of 0
	// or 1: a decided game (MAGPIE's default).
	baiCutoff = 0.005 / 100
)

// baiState is what a sim keeps for best-arm identification.
type baiState struct {
	on           bool
	minSamples   int
	spreadWeight float64
	// capPerPlay is the most rollouts the sim makes, as an average per
	// play; 0 for the stopping rule's own cap on iterations.
	capPerPlay int

	runner *bai.Runner
	arms   []*SimmedPlay
	seed   [32]byte
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
	seed := s.origGame.Seed()
	if seed == ([32]byte{}) {
		frand.Read(b.seed[:])
	} else {
		var buf [40]byte
		copy(buf[:32], seed[:])
		binary.BigEndian.PutUint64(buf[32:], uint64(s.origGame.Turn()))
		b.seed = sha256.Sum256(buf[:])
	}
}

// baiRNG is the random source for sample number ordinal.
func (s *Simmer) baiRNG(ordinal uint64) *frand.RNG {
	var buf [40]byte
	copy(buf[:32], s.bai.seed[:])
	binary.BigEndian.PutUint64(buf[32:], ordinal)
	h := sha256.Sum256(buf[:])
	return frand.NewCustom(h[:], 0, 0)
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
		s.rngs = make([]*frand.RNG, threads)
	}
	minimum := b.minSamples
	if minimum <= 0 {
		minimum = DefaultBAIMinSamples
	}
	opts := bai.Options{
		Delta:         baiDelta(s.autostopper.stoppingCondition),
		SampleMinimum: minimum,
		Cutoff:        baiCutoff,
		Threads:       threads,
		TimeLimit:     s.timeLimit,
		// A round is laid out before any of it is drawn, so it has to be
		// big enough to keep every thread busy.
		RoundSize: max(bai.DefaultRoundSize, 2*threads),
	}
	// An iteration is one rollout of every play.
	switch {
	case run.maxIters > 0:
		opts.SampleLimit = b.runner.Samples() + run.maxIters*numArms
	case b.capPerPlay > 0:
		opts.SampleLimit = b.capPerPlay * numArms
	default:
		opts.SampleLimit = (s.autostopper.iterationsCutoff + s.maxPlies*s.autostopper.perPlyStopScaling) * numArms
	}
	res, err := b.runner.Run(ctx, opts, func(worker, arm int, ordinal uint64) (float64, error) {
		return s.baiSample(worker, b.arms[arm], ordinal)
	})
	s.iterationCount.Store(uint64(b.runner.Samples() / max(numArms, 1)))
	if err != nil {
		logger.Err(err).Msg("error simming; stopping")
		return
	}
	b.status = res.Status
	b.winner = b.arms[res.Best]
	s.event(logger, run.quiet).Str("status", res.Status.String()).Int("rollouts", res.Samples).
		Str("best", b.winner.play.ShortDescription()).Msg("sim-stopped")
}

// baiSample draws sample number ordinal of play on thread's game copy and
// returns its reward: deal the opponent the sample's rack, roll the play
// out, and put the game back.
func (s *Simmer) baiSample(thread int, play *SimmedPlay, ordinal uint64) (float64, error) {
	g := s.gameCopies[thread]
	opp := (s.initialPlayer + 1) % g.NumPlayers()
	rng := s.baiRNG(ordinal)
	// The opponent's rack, when it is inferred, is drawn from the sample's
	// source too.
	s.rngs[thread] = rng
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
	win, eq := s.rollout(g, thread, s.maxPlies, play, logIter)
	if logIter != nil {
		if err := s.baiWriteLog(logIter); err != nil {
			return 0, err
		}
	}
	w := s.bai.spreadWeight
	if w == 0 {
		w = DefaultBAISpreadWeight
	}
	return baiReward(win, eq+float64(s.initialSpread), w), nil
}

// baiWriteLog writes one sample's log entry: JSON lines for a heat map (up
// to MaxHeatMapIterations of each play's samples), YAML otherwise.
func (s *Simmer) baiWriteLog(logIter *LogIteration) error {
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
	s.bai.logMu.Lock()
	defer s.bai.logMu.Unlock()
	_, err = s.logStream.Write(out)
	return err
}
