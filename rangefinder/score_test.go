package rangefinder

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/game"
)

// scoreTrueLeave grades the posterior against the leave the opponent really
// held, which is what a self-play experiment gets to do.
func scoreTrueLeave(t *testing.T, g *game.Game, budget int) (LeaveScore, []tilemapping.MachineLetter) {
	t.Helper()
	rf := &RangeFinder{}
	rf.Init(g, defaultSimCalculators("NWL18"), DefaultConfig)
	rf.SetThreads(1)
	rf.SetBudget(budget)
	if err := rf.PrepareFinder(g.RackFor(g.PlayerOnTurn()).TilesOn()); err != nil {
		t.Skipf("nothing to infer here: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if err := rf.Infer(ctx); err != nil {
		t.Fatal(err)
	}
	trueLeave, err := game.ExtractLastOppLeave(g)
	if err != nil {
		t.Skipf("no true leave available: %v", err)
	}
	return rf.ScoreLeave(trueLeave), trueLeave
}

func TestScoreLeaveGradesTheTruth(t *testing.T) {
	is := is.New(t)
	g := seededMidGame(t, 6)
	score, trueLeave := scoreTrueLeave(t, g, 40)

	is.True(score.Leaves > 0)
	is.True(score.InPosterior) // the true leave must be feasible
	is.True(score.Posterior > 0 && score.Posterior <= 1)
	is.True(score.Prior > 0 && score.Prior <= 1)
	is.True(score.Rank >= 1 && score.Rank <= score.Leaves)

	// LiftBits is exactly the log2 ratio, which is the whole point of it.
	is.True(math.Abs(score.LiftBits-math.Log2(score.Posterior/score.Prior)) < 1e-12)
	is.True(len(trueLeave) > 0)
}

// The posterior is a probability distribution, so the shares it reports have to
// add to one across every leave it covers.
func TestScoreLeavePosteriorsSumToOne(t *testing.T) {
	is := is.New(t)
	g := seededMidGame(t, 6)

	rf := &RangeFinder{}
	rf.Init(g, defaultSimCalculators("NWL18"), DefaultConfig)
	rf.SetThreads(1)
	rf.SetBudget(30)
	if err := rf.PrepareFinder(g.RackFor(g.PlayerOnTurn()).TilesOn()); err != nil {
		t.Skipf("nothing to infer here: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	is.NoErr(rf.Infer(ctx))

	total := 0.0
	for _, ir := range rf.Inferences().InferredRacks {
		total += rf.ScoreLeave(ir.Leave).Posterior
	}
	is.True(math.Abs(total-1.0) < 1e-9)
}

// A leave the posterior cannot contain scores as absent rather than erroring:
// "inference ruled the truth out" is a result worth logging, not a failure.
func TestScoreLeaveHandlesAbsentLeaves(t *testing.T) {
	is := is.New(t)
	g := seededMidGame(t, 6)

	rf := &RangeFinder{}
	rf.Init(g, defaultSimCalculators("NWL18"), DefaultConfig)
	rf.SetThreads(1)
	rf.SetBudget(20)
	if err := rf.PrepareFinder(g.RackFor(g.PlayerOnTurn()).TilesOn()); err != nil {
		t.Skipf("nothing to infer here: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	is.NoErr(rf.Infer(ctx))

	// A tile index past the end of the unseen pool cannot be part of any leave.
	score := rf.ScoreLeave([]tilemapping.MachineLetter{200})
	is.True(!score.InPosterior)
	is.True(score.Posterior == 0)
	is.True(score.Rank == 0)
	is.True(math.IsNaN(score.LiftBits))

	// An empty rangefinder grades nothing at all.
	empty := &RangeFinder{}
	blank := empty.ScoreLeave([]tilemapping.MachineLetter{1})
	is.True(!blank.InPosterior)
	is.True(blank.Leaves == 0)
}
