package rangefinder

import (
	"context"
	"errors"
	"time"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/simplesimmer"
	"github.com/domino14/macondo/move"
)

// MiniSimMeasurement is one mini-sim of a candidate leave: every simmed play's
// log-odds of winning, and which of them is the move the opponent made.
type MiniSimMeasurement struct {
	LogOdds   []float64
	Samples   []int
	TargetIdx int
	Elapsed   time.Duration
}

// MeasureLeaveMiniSim runs the mini-sim that measures leave, as
// evaluateLeaves does, but seeded by seed, capped at iters rollouts a play,
// over the top nMoves static plays (plus the opponent's), and with every play
// given the same iters rollouts when equal is set. For experiments on
// mini-sim noise; PrepareFinder must have run, and it uses thread 0.
func (r *RangeFinder) MeasureLeaveMiniSim(leave []tilemapping.MachineLetter, seed [32]byte,
	iters, nMoves int, equal bool) (MiniSimMeasurement, error) {

	if !r.readyToInfer || len(r.gameCopies) == 0 {
		return MiniSimMeasurement{}, errors.New("PrepareFinder first")
	}
	gc := r.gameCopies[0]
	opp := gc.PlayerOnTurn()
	simmer := r.aiplayers[0].(*simplesimmer.SimpleSimmer)
	fullRack := append(append([]tilemapping.MachineLetter{}, r.lastOppMoveRackTiles...), leave...)
	if _, err := gc.SetRandomRack(opp, fullRack); err != nil {
		return MiniSimMeasurement{}, err
	}
	gc.SeedBag(seed)
	lastOppMove := &move.Move{}
	lastOppMove.CopyFrom(r.lastOppMove)
	lastOppMove.SetLeave(leave)

	simmer.SetMaxIters(iters)
	simmer.SetEqualAllocation(equal)
	started := time.Now()
	if _, err := simmer.GenAndSim(context.Background(), nMoves, lastOppMove); err != nil {
		return MiniSimMeasurement{}, err
	}
	m := MiniSimMeasurement{Elapsed: time.Since(started), TargetIdx: -1}
	for i, sp := range simmer.BestPlays().PlaysNoLock() {
		m.LogOdds = append(m.LogOdds, Logit(sp.WinProb()))
		m.Samples = append(m.Samples, sp.WinProbIterations())
		if movesAreTheSame(sp.Move(), lastOppMove, gc.Board()) {
			m.TargetIdx = i
		}
	}
	return m, nil
}
