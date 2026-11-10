package bot

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/proto"

	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/rangefinder"
)

// carryPower is the lab knob for two-turn reads: MACONDO_CARRY_POWER > 0
// reweights this turn's read by the read of the opponent's play before
// (rangefinder.CarryFrom), its likelihood raised to this power. 0 (unset) is
// off. MACONDO_CARRY_MINBAG limits it to positions with at least that many
// tiles in the bag.
func carryPower() (power float64, minBag int) {
	power, _ = strconv.ParseFloat(os.Getenv("MACONDO_CARRY_POWER"), 64)
	minBag, _ = strconv.Atoi(os.Getenv("MACONDO_CARRY_MINBAG"))
	return power, minBag
}

// carryRead returns the bot's complete posterior reweighted by its read of
// the opponent's previous play, or ok false when the knob is off or there is
// no usable previous read (the opponent's first play, a previous exchange or
// bingo, a read that failed).
func (p *BotTurnPlayer) carryRead() (racks []montecarlo.InferredRack, ok bool) {
	power, minBag := carryPower()
	if power <= 0 || p.Game.Bag().TilesRemaining() < minBag {
		return nil, false
	}
	prev, ok := p.previousRead()
	if !ok {
		return nil, false
	}
	racks, ok = p.inferencer.CarryFrom(prev.rf, power)
	if ok && os.Getenv("MACONDO_CARRY_DEBUG") != "" {
		p.logCarry(p.inferencer.Inferences().InferredRacks, racks)
	}
	return racks, ok
}

// logCarry prints, for lab runs where the game holds the opponent's true
// rack, how many bits the carry moved the true leave's posterior share.
func (p *BotTurnPlayer) logCarry(before, after []montecarlo.InferredRack) {
	leave, err := game.ExtractLastOppLeave(p.Game)
	if err != nil {
		return
	}
	sorted := func(l []tilemapping.MachineLetter) string {
		c := slices.Clone(l)
		slices.Sort(c)
		return tilemapping.MachineWord(c).UserVisible(p.Game.Alphabet())
	}
	key := sorted(leave)
	share := func(rs []montecarlo.InferredRack) float64 {
		tot, hit := 0.0, 0.0
		for _, r := range rs {
			tot += r.Weight
			if sorted(r.Leave) == key {
				hit += r.Weight
			}
		}
		return hit / tot
	}
	b, a := share(before), share(after)
	fmt.Fprintf(os.Stderr, "carry bag=%d k=%d true=%s before=%.6g after=%.6g bits=%+.3f\n",
		p.Game.Bag().TilesRemaining(), len(leave), key, b, a, math.Log2(a/b))
}

// A cachedRead is the bot's read of one opponent play, kept for the next
// turn's carry. play is the history event read and turn the bot's turn
// number then, so it is used only for that play in that game.
type cachedRead struct {
	rf   *rangefinder.RangeFinder
	g    *game.Game
	play *pb.GameEvent
	turn int
}

// cacheRead keeps this turn's read for the next turn's carry.
func (p *BotTurnPlayer) cacheRead() {
	if power, _ := carryPower(); power <= 0 {
		return
	}
	evts := p.Game.History().Events[:p.Game.Turn()]
	k := len(evts) - 1
	for k >= 0 && evts[k].Type == pb.GameEvent_CHALLENGE_BONUS {
		k--
	}
	if k < 0 {
		return
	}
	p.prevRead = &cachedRead{rf: p.inferencer.Snapshot(), g: p.Game, play: evts[k], turn: p.Game.Turn()}
}

// previousRead returns the bot's read of the opponent's play before last:
// the one it kept from its previous turn in this game, or, for a bot new to
// the game (as in screens, which decide each position with a fresh bot), one
// rebuilt from the game as it stood then, right after that play. The rebuilt
// game takes this game's seed, so the read is the one the bot made back then.
func (p *BotTurnPlayer) previousRead() (*cachedRead, bool) {
	evts := p.Game.History().Events[:p.Game.Turn()]
	if len(evts) == 0 {
		return nil, false
	}
	opp := evts[len(evts)-1].PlayerIndex
	k := len(evts) - 1
	for k >= 0 && evts[k].PlayerIndex == opp {
		k--
	}
	// evts[k] is our last move; our turn began after the opponent's
	// event before it.
	for k >= 0 && evts[k].PlayerIndex != opp {
		k--
	}
	if k < 0 {
		return nil, false
	}
	turn := k + 1
	for k >= 0 && evts[k].PlayerIndex == opp && evts[k].Type == pb.GameEvent_CHALLENGE_BONUS {
		k--
	}
	if k < 0 || evts[k].Type != pb.GameEvent_TILE_PLACEMENT_MOVE {
		return nil, false
	}
	if c := p.prevRead; c != nil && c.g == p.Game {
		// A bot that played this game kept its last read. If it is not of
		// this play, the bot did not read that play (an exchange or bingo
		// before it), and neither would a rebuild.
		if c.play == evts[k] && c.turn == turn {
			carryCacheHits.Add(1)
			return c, true
		}
		return nil, false
	}
	carryCacheMisses.Add(1)
	if os.Getenv("MACONDO_CARRY_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "carry-rebuild turn=%d\n", turn)
	}
	myRack, err := tilemapping.ToMachineLetters(evts[turn].Rack, p.Game.Alphabet())
	if err != nil {
		return nil, false
	}
	hist := proto.Clone(p.Game.History()).(*pb.GameHistory)
	g, err := game.NewFromHistory(hist, p.Game.Rules(), turn)
	if err != nil {
		log.Debug().Err(err).Msg("carry-rebuild")
		return nil, false
	}
	g.SeedBag(p.Game.Seed())
	rf := &rangefinder.RangeFinder{}
	p.setupInferencer(rf, g)
	if err := rf.PrepareFinder(myRack); err != nil {
		return nil, false
	}
	ctx, cancel := context.Background(), context.CancelFunc(func() {})
	if p.cfg.InferenceBudget <= 0 {
		secs := 20
		if p.cfg.InferenceTimeSecs > 0 {
			secs = p.cfg.InferenceTimeSecs
		}
		ctx, cancel = context.WithTimeout(ctx, time.Duration(secs)*time.Second)
	}
	defer cancel()
	if err := rf.Infer(ctx); err != nil {
		log.Debug().Err(err).Msg("carry-infer")
	}
	return &cachedRead{rf: rf}, true
}

// carryCacheHits and carryCacheMisses count how often the carry found its
// previous read kept or had to rebuild it, for lab logs.
var carryCacheHits, carryCacheMisses atomic.Int64

// CarryCacheStats returns the carry's kept and rebuilt previous reads.
func CarryCacheStats() (hits, misses int64) {
	return carryCacheHits.Load(), carryCacheMisses.Load()
}
