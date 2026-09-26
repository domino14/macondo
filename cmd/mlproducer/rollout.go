// rollout.go - bootstrapped labels from sampled rollouts.
//
// Instead of one real continuation looked up in the win-percentage table,
// a position is labeled by N rollouts of K plies of static best play with
// the value net at the leaf:
//
//  1. draw the mover's replacement tiles and the opponent's rack at random
//     from the unseen pool,
//  2. play K plies, each side's best static play,
//  3. score the leaf with the net, from the leaf mover's side, negated when
//     the leaf mover is the opponent,
//
// and average the N results. A rollout that ends the game contributes the
// real result instead. The spread label is built the same way: the real
// spread change over the K plies plus the net's predicted change at the
// leaf.
package main

import (
	"fmt"
	"math"

	"github.com/domino14/word-golib/tilemapping"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/triton"
)

// rolloutLabel is what the labeler produces for one position.
type rolloutLabel struct {
	value  float32 // mean leaf value from the mover's side, in [-1, 1]
	spread float32 // mean spread change from the mover's side, in points
}

// leafScorer scores leaf positions; the Triton client in production, a stub
// in tests.
type leafScorer interface {
	Infer(planes, scalars []float32, n int) (*triton.ModelOutputs, error)
}

// RolloutLabeler labels positions by rollouts; one per worker (it keeps
// batch buffers), sharing the Triton client and calculators.
type RolloutLabeler struct {
	plies    int
	rollouts int
	client   leafScorer
	calcs    []equity.EquityCalculator
	eqCalc   *equity.ExhaustiveLeaveCalculator

	planes  []float32
	scalars []float32
}

// RolloutShared is what every worker's labeler shares.
type RolloutShared struct {
	Client leafScorer
	Calcs  []equity.EquityCalculator
}

// NewRolloutShared connects to Triton (from the MACONDO_TRITON_* settings)
// and builds the static-play equity calculators.
func NewRolloutShared(cfg *config.Config, lexicon string) (*RolloutShared, error) {
	client, err := triton.NewTritonClient(
		cfg.GetString(config.ConfigTritonURL),
		cfg.GetString(config.ConfigTritonModelName),
		cfg.GetString(config.ConfigTritonModelVersion))
	if err != nil {
		return nil, err
	}
	client.SetOutputs([]string{"value", "spread"})
	calc, err := equity.NewCombinedStaticCalculator(lexicon, cfg, "", "")
	if err != nil {
		return nil, err
	}
	return &RolloutShared{Client: client, Calcs: []equity.EquityCalculator{calc}}, nil
}

func NewRolloutLabeler(shared *RolloutShared, plies, rollouts int,
	eqCalc *equity.ExhaustiveLeaveCalculator) *RolloutLabeler {
	return &RolloutLabeler{
		plies:    plies,
		rollouts: rollouts,
		client:   shared.Client,
		calcs:    shared.Calcs,
		eqCalc:   eqCalc,
		planes:   make([]float32, 0, rollouts*game.NN_N_PLANES),
		scalars:  make([]float32, 0, rollouts*game.NN_N_SCAL),
	}
}

// StackLength is the game state stack a labeled game needs.
func (rl *RolloutLabeler) StackLength() int { return rl.plies + 2 }

type leaf struct {
	sign  float32 // +1 if the leaf mover is the original mover, else -1
	delta float32 // real spread change for the mover over the plies played
}

// Label labels the position the game is in: just after `m` by `mover`,
// mover's rack holding only the leave, opponent's rack thrown into the
// bag, mover on turn. The game is returned in that same state. ok is
// false if the position cannot be labeled (game already over).
func (rl *RolloutLabeler) Label(g *game.Game, ai aiturnplayer.AITurnPlayer, mover int,
	m *move.Move, history []*move.Move) (rolloutLabel, bool, error) {

	if g.Playing() != pb.PlayState_PLAYING {
		return rolloutLabel{}, false, nil
	}
	opp := 1 - mover
	leave := m.Leave()
	exchange := m.Action() == move.MoveTypeExchange
	spreadNow := float32(g.SpreadFor(mover))

	rl.planes = rl.planes[:0]
	rl.scalars = rl.scalars[:0]
	leaves := make([]leaf, 0, rl.rollouts)
	var valueSum, spreadSum float64
	rolloutMoves := make([]*move.Move, 0, rl.plies)
	// history for the leaf's feature vector: everything before the leaf move
	lastMoves := make([]*move.Move, 0, len(history)+1+rl.plies)

	g.SetBackupMode(game.SimulationMode)
	defer func() {
		// Leave the game exactly as we found it: mover on turn holding only
		// the leave, everything else in the bag.
		g.SetBackupMode(game.NoBackup)
		g.ThrowRacksIn()
		rack := tilemapping.NewRack(g.Alphabet())
		rack.Set(leave)
		if err := g.SetRackForOnly(mover, rack); err != nil {
			panic(fmt.Sprintf("rollout: restoring the mover's leave: %v", err))
		}
		g.SetPlayerOnTurn(mover)
	}()

	for i := 0; i < rl.rollouts; i++ {
		// Fresh racks. The mover draws before their exchanged tiles (if any)
		// go back in the bag, as in a real game.
		g.ThrowRacksIn()
		if exchange {
			if err := g.Bag().RemoveTiles(m.Tiles()); err != nil {
				return rolloutLabel{}, false, fmt.Errorf("rollout: remove exchanged tiles: %w", err)
			}
		}
		if _, err := g.SetRandomRack(mover, leave); err != nil {
			return rolloutLabel{}, false, fmt.Errorf("rollout: mover draw: %w", err)
		}
		if exchange {
			g.Bag().PutBack(m.Tiles())
		}
		if _, err := g.SetRandomRack(opp, nil); err != nil {
			return rolloutLabel{}, false, fmt.Errorf("rollout: opponent draw: %w", err)
		}
		g.SetPlayerOnTurn(opp)

		rolloutMoves = rolloutMoves[:0]
		for p := 0; p < rl.plies && g.Playing() == pb.PlayState_PLAYING; p++ {
			onturn := g.PlayerOnTurn()
			best := aiturnplayer.GenBestStaticTurn(g, ai, onturn)
			if err := g.PlayMove(best, false, 0); err != nil {
				return rolloutLabel{}, false, fmt.Errorf("rollout: play %s: %w", best.ShortDescription(), err)
			}
			// The generator hands back its one reusable "winner" object, so
			// keep a copy or every ply would alias the last one.
			kept := &move.Move{}
			kept.CopyFrom(best)
			rolloutMoves = append(rolloutMoves, kept)
		}

		if g.Playing() != pb.PlayState_PLAYING {
			// Real result. In simulation mode a play-out goes straight to
			// GAME_OVER with the end-of-game bonus applied.
			spread := float32(g.SpreadFor(mover))
			switch {
			case spread > 0:
				valueSum += 1
			case spread < 0:
				valueSum -= 1
			}
			spreadSum += float64(spread - spreadNow)
		} else {
			last := rolloutMoves[len(rolloutMoves)-1]
			leafMover := 1 - g.PlayerOnTurn()
			// Same convention as a training position: leaf mover on turn,
			// holding only their leave, everything else unseen.
			g.ThrowRacksIn()
			g.SetPlayerOnTurn(leafMover)
			rack := tilemapping.NewRack(g.Alphabet())
			rack.Set(last.Leave())
			if err := g.SetRackForOnly(leafMover, rack); err != nil {
				return rolloutLabel{}, false, fmt.Errorf("rollout: leaf rack: %w", err)
			}
			lastMoves = append(lastMoves[:0], history...)
			lastMoves = append(lastMoves, m)
			lastMoves = append(lastMoves, rolloutMoves[:len(rolloutMoves)-1]...)
			vec, err := g.BuildMLVector(last, rl.eqCalc.LeaveValue(last.Leave()), lastMoves)
			if err != nil {
				return rolloutLabel{}, false, fmt.Errorf("rollout: leaf vector: %w", err)
			}
			rl.planes = append(rl.planes, (*vec)[:game.NN_N_PLANES]...)
			rl.scalars = append(rl.scalars, (*vec)[game.NN_N_PLANES:]...)
			game.MLVectorPool.Put(vec)
			sign := float32(1)
			if leafMover != mover {
				sign = -1
			}
			leaves = append(leaves, leaf{sign: sign, delta: float32(g.SpreadFor(mover)) - spreadNow})
		}
		g.ResetToFirstState()
	}

	if len(leaves) > 0 {
		out, err := rl.client.Infer(rl.planes, rl.scalars, len(leaves))
		if err != nil {
			return rolloutLabel{}, false, fmt.Errorf("rollout: infer: %w", err)
		}
		if len(out.Value) != len(leaves) || len(out.Spread) != len(leaves) {
			return rolloutLabel{}, false, fmt.Errorf("rollout: got %d values, %d spreads for %d leaves",
				len(out.Value), len(out.Spread), len(leaves))
		}
		for i, lf := range leaves {
			valueSum += float64(lf.sign * out.Value[i])
			// The spread head is tanh(x/130); invert it, clamped away from +-1.
			y := float32(math.Max(-0.999, math.Min(0.999, float64(out.Spread[i]))))
			pred := game.InverseScaleScoreWithTanh(y, 0, 130)
			spreadSum += float64(lf.delta + lf.sign*pred)
		}
	}
	n := float64(rl.rollouts)
	return rolloutLabel{
		value:  float32(valueSum / n),
		spread: float32(spreadSum / n),
	}, true, nil
}

// eqCalcsForTest builds the static-play calculators without a Triton
// connection.
func (ga *GameAssembler) eqCalcsForTest() []equity.EquityCalculator {
	calc, err := equity.NewCombinedStaticCalculator("NWL23", DefaultConfig, "", "")
	if err != nil {
		panic(err)
	}
	return []equity.EquityCalculator{calc}
}
