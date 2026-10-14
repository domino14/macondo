package montecarlo

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"math"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"
	"gopkg.in/yaml.v3"
	"lukechampine.com/frand"

	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo/bai"
	"github.com/domino14/macondo/move"
)

// A sim is a best-arm identification problem, set up as MAGPIE sets up its
// sims: every candidate play is an arm, and a sample of an arm is one rollout
// of that play. Samples are drawn on common random numbers: an arm's n-th
// sample deals the opponent the rack, and the bag the order, that every other
// arm's n-th sample gets, so the stopping rule can use how the plays' results
// covary. The reward is MAGPIE's utility, a blend of the rollout's win
// probability and its final spread, whose spread term also ranks the plays of
// a decided game.

const (
	// DefaultMinSamplesPerPlay is how many rollouts every play gets before
	// any is preferred or the sim may stop (MAGPIE's -minplayiterations).
	DefaultMinSamplesPerPlay = 500
	// utilityCutoff stops the sim once the best play's utility is within it
	// of 0 or 1 (MAGPIE's default cutoff, 0.005%).
	utilityCutoff = 0.005 / 100
	// MAGPIE's default utility: weights for win probability and for the
	// spread term, and the spread the sigmoid is scaled by.
	utilityWinWeight    = 1.0
	utilitySpreadWeight = 0.5
	utilitySpreadScale  = 100.0
)

// simUtility is MAGPIE's sim_utility_blend: win probability blended with a
// sigmoid of the rollout's final spread.
func simUtility(winProb float64, spread int) float64 {
	x := float64(spread) / utilitySpreadScale
	var sig float64
	if x >= 0 {
		sig = 1 / (1 + math.Exp(-x))
	} else {
		e := math.Exp(x)
		sig = e / (1 + e)
	}
	total := utilityWinWeight + utilitySpreadWeight
	return utilityWinWeight/total*winProb + utilitySpreadWeight/total*sig
}

// deltaFor is the error probability a stopping condition allows (1 - its
// confidence, as MAGPIE takes it); 0 means no stopping threshold.
func deltaFor(sc StoppingCondition) float64 {
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

// startBAI fixes the sim's arms and derives the seed its samples are drawn
// from. It runs on the first simulate after PrepareSim (and after any
// AvoidPruningMoves, which can add arms).
func (s *Simmer) startBAI() {
	s.simmedPlays.RLock()
	s.arms = append(s.arms[:0], s.simmedPlays.plays...)
	s.simmedPlays.RUnlock()
	n := len(s.arms)
	keys := make([]similarityKey, n)
	for i, a := range s.arms {
		keys[i] = similarityKeyFor(a.play)
	}
	similar := make([]bool, n*n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			// A play the caller asked to keep has to be beaten on its own.
			if s.arms[i].unignorable || s.arms[j].unignorable {
				continue
			}
			if keys[i] == keys[j] {
				similar[i*n+j], similar[j*n+i] = true, true
			}
		}
	}
	s.baiRunner = bai.NewRunner(n, bai.Options{
		Similar: func(a, b int) bool { return similar[a*n+b] },
		Paired:  true,
	})

	seed := s.origGame.Seed()
	if seed == ([32]byte{}) {
		frand.Read(s.simSeed[:])
	} else {
		var buf [40]byte
		copy(buf[:32], seed[:])
		binary.BigEndian.PutUint64(buf[32:], uint64(s.origGame.Turn()))
		s.simSeed = sha256.Sum256(buf[:])
	}
}

// rngFor is the random source for sample number ordinal.
func (s *Simmer) rngFor(ordinal uint64) *frand.RNG {
	var buf [40]byte
	copy(buf[:32], s.simSeed[:])
	binary.BigEndian.PutUint64(buf[32:], ordinal)
	h := sha256.Sum256(buf[:])
	return frand.NewCustom(h[:], 0, 0)
}

// simBAI runs the sim until its stopping rule, its iteration budget or the
// context says to stop.
func (s *Simmer) simBAI(ctx context.Context, run simRun) {
	logger := zerolog.Ctx(ctx)
	if s.baiRunner == nil {
		s.startBAI()
	}
	numArms := len(s.arms)
	threads := s.threads
	if run.inline || threads < 1 {
		threads = 1
	}
	if len(s.rngs) < threads {
		s.rngs = make([]*frand.RNG, threads)
	}
	sc := s.autostopper.stoppingCondition
	opts := bai.Options{
		Delta:         deltaFor(sc),
		SampleMinimum: s.minSamplesPerPlay,
		Cutoff:        utilityCutoff,
		Threads:       threads,
	}
	if opts.SampleMinimum <= 0 {
		opts.SampleMinimum = DefaultMinSamplesPerPlay
	}
	// The budget counts samples; an iteration is one sample of every play.
	switch {
	case run.maxIters > 0:
		opts.SampleLimit = s.baiRunner.Samples() + run.maxIters*numArms
	case sc != StopNone:
		opts.SampleLimit = (s.autostopper.iterationsCutoff + s.maxPlies*s.autostopper.perPlyStopScaling) * numArms
	}
	res, err := s.baiRunner.Run(ctx, opts, func(worker, arm int, ordinal uint64) (float64, error) {
		return s.sample(ctx, worker, s.arms[arm], ordinal)
	})
	s.iterationCount.Store(uint64(s.baiRunner.Samples() / numArms))
	if err != nil {
		logger.Err(err).Msg("error simming; stopping")
		return
	}
	s.event(logger, run.quiet).Str("status", res.Status.String()).Int("samples", res.Samples).
		Msg("sim-stopped")
}

// sample draws one rollout of play: deals the opponent the sample's rack,
// plays play and then the sim's plies, records the stats, and puts the game
// back. It returns the rollout's utility.
func (s *Simmer) sample(ctx context.Context, thread int, play *SimmedPlay, ordinal uint64) (float64, error) {
	g := s.gameCopies[thread]
	rng := s.rngFor(ordinal)
	s.rngs[thread] = rng
	opp := (s.initialPlayer + 1) % g.NumPlayers()

	rackToSet, err := s.oppRackFor(thread)
	if err != nil {
		return 0, err
	}
	if err := g.DealRandomRackSeeded(opp, rackToSet, rng); err != nil {
		return 0, err
	}
	var logIter *LogIteration
	if s.logStream != nil {
		logIter = &LogIteration{Iteration: int(ordinal), Thread: thread}
		if len(rackToSet) > 0 {
			logIter.InferredRack = tilemapping.MachineWord(rackToSet).UserVisible(g.Alphabet())
		}
	}
	utility := s.rollout(g, thread, play, logIter)
	if logIter != nil {
		if err := s.writeIteration(ctx, logIter); err != nil {
			return 0, err
		}
	}
	return utility, nil
}

// oppRackFor picks the tiles the opponent is known or inferred to hold for
// this sample (nil for a fully random rack), from the thread's sample source.
func (s *Simmer) oppRackFor(thread int) ([]tilemapping.MachineLetter, error) {
	rackToSet := s.knownOppRack
	if s.inferenceMode == InferenceCompletePosterior {
		// The inferences cover every feasible leave with Bayesian posterior
		// weights, so sample from them every time. There is no uncovered
		// leave space to fall back to random draws for.
		if len(s.inferenceCDF) == 0 {
			return nil, nil
		}
		return s.weightedInferredDrawRacks(thread)
	}
	if s.inferenceMode == InferenceOff {
		return rackToSet, nil
	}
	// Legacy partial-sample modes (currently only exchange inference
	// produces these): the inferred racks cover an unknown fraction of the
	// leave space, so a sigmoid of how many there are is the probability of
	// sampling from them rather than drawing a fully random rack.
	const maxAlpha = 0.95
	const midpoint = 10.0
	const sigmoidScale = 5.0
	alpha := maxAlpha / (1 + math.Exp(-(float64(len(s.inferences))-midpoint)/sigmoidScale))
	if s.float64For(thread) >= alpha {
		return nil, nil
	}
	if s.inferenceMode == InferenceWeightedRandomTiles {
		return s.weightedInferredDrawTiles(thread)
	}
	return s.weightedInferredDrawRacks(thread)
}

// rollout plays play and then the sim's plies on g, records the play's
// stats, and resets g. It returns the rollout's utility.
func (s *Simmer) rollout(g *game.Game, thread int, simmedPlay *SimmedPlay, logIter *LogIteration) float64 {
	var logPlay LogPlay
	if logIter != nil {
		logPlay = LogPlay{
			Play:  g.Board().MoveDescriptionWithPlaythrough(simmedPlay.play),
			Leave: simmedPlay.play.LeaveString(),
			Rack:  simmedPlay.play.FullRack(),
			Bingo: simmedPlay.play.BingoPlayed(),
			Pts:   simmedPlay.play.Score()}
	}
	// equity of the leftover tiles at the end of the sim
	leftover := 0.0
	plies := s.maxPlies
	// Only the play itself is backed up; the plies after it are undone by
	// resetting to that one backup.
	g.SetBackupMode(game.SimulationMode)
	g.PlayMove(simmedPlay.play, false, 0)
	s.nodeCount.Add(1)
	g.SetBackupMode(game.NoBackup)
	for ply := 0; ply < plies; ply++ {
		onTurn := g.PlayerOnTurn()
		if g.Playing() != pb.PlayState_PLAYING {
			break
		}
		bestPlay := s.bestStaticTurn(onTurn, thread)
		// On the last ply no further move generation happens, so cross-set
		// recalculation would be discarded by the reset unused.
		if ply == plies-1 {
			g.PlayMoveNoCrossSet(bestPlay, false, 0)
		} else {
			g.PlayMove(bestPlay, false, 0)
		}
		s.nodeCount.Add(1)
		var plyChild LogPlay
		if logIter != nil {
			plyChild = LogPlay{
				Play:  g.Board().MoveDescriptionWithPlaythrough(bestPlay),
				Leave: bestPlay.LeaveString(),
				Rack:  bestPlay.FullRack(),
				Pts:   bestPlay.Score(),
				Bingo: bestPlay.BingoPlayed()}
		}
		if ply == plies-2 || ply == plies-1 {
			// It's either OUR last turn or OPP's last turn: count the
			// leftover tiles' equity.
			thisLeftover := s.leaveValues.LeaveValue(bestPlay.Leave())
			plyChild.Leftover = thisLeftover
			if onTurn == s.initialPlayer {
				leftover += thisLeftover
			} else {
				leftover -= thisLeftover
			}
		}
		if logIter != nil {
			logPlay.Plies = append(logPlay.Plies, plyChild)
		}
		simmedPlay.addScoreStat(bestPlay, ply)
	}
	spread := g.SpreadFor(s.initialPlayer)
	simmedPlay.addEquityStat(s.initialSpread, spread, leftover)
	onTurn := g.PlayerOnTurn()
	winProb := simmedPlay.addWinPctStat(
		spread,
		leftover,
		g.Playing() == pb.PlayState_GAME_OVER,
		s.winPcts,
		g.Bag().TilesRemaining(),
		int(g.RackFor(onTurn).NumTiles()),
		int(g.RackFor(1-onTurn).NumTiles()),
		plies%2 == 0,
	)
	g.ResetToFirstState()
	utility := simUtility(winProb, spread)
	simmedPlay.addUtilityStat(utility)
	if logIter != nil {
		logPlay.WinRatio = winProb
		logIter.Plays = append(logIter.Plays, logPlay)
	}
	return utility
}

// writeIteration writes one sample's log entry: JSON lines for a heat map
// (up to MaxHeatMapIterations of them), YAML otherwise.
func (s *Simmer) writeIteration(ctx context.Context, logIter *LogIteration) error {
	logger := zerolog.Ctx(ctx)
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
		logger.Error().Err(err).Msg("marshalling log")
		return err
	}
	s.logMu.Lock()
	defer s.logMu.Unlock()
	_, err = s.logStream.Write(out)
	return err
}

// similarityKey identifies plays MAGPIE counts as the same for stopping
// (move_get_similarity_key): same kind, square, direction and length, the same
// rack tiles used (a blank whatever it stands for), and the same score.
type similarityKey struct {
	action           move.MoveType
	row, col, length int
	vertical         bool
	score            int
	tiles            [tilemapping.MaxAlphabetSize]uint8
}

func similarityKeyFor(m *move.Move) similarityKey {
	k := similarityKey{action: m.Action(), length: len(m.Tiles()), score: m.Score()}
	if m.Action() == move.MoveTypePlay {
		k.row, k.col, k.vertical = m.CoordsAndVertical()
	}
	if m.Action() == move.MoveTypePlay || m.Action() == move.MoveTypeExchange {
		for _, t := range m.Tiles() {
			if t == 0 && m.Action() == move.MoveTypePlay {
				continue // played through
			}
			k.tiles[t.IntrinsicTileIdx()]++
		}
	}
	return k
}

// sampleEveryPlay draws one sample of every play at sample number ordinal,
// on thread's game copy: what a sim iteration was before the sim chose which
// plays to sample. For tests and benchmarks.
func (s *Simmer) sampleEveryPlay(ctx context.Context, thread int, ordinal uint64) error {
	if s.baiRunner == nil {
		s.startBAI()
	}
	if len(s.rngs) <= thread {
		s.rngs = append(s.rngs, make([]*frand.RNG, thread+1-len(s.rngs))...)
	}
	for _, arm := range s.arms {
		if _, err := s.sample(ctx, thread, arm, ordinal); err != nil {
			return err
		}
	}
	return nil
}
