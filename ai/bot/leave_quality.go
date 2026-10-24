package bot

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
)

// Leave-quality reweighting (experimental). After a short play the opponent
// most likely kept a strong leave, and how strong depends on how many tiles
// they played and for how many points. The complete posterior is multiplied
// by exp(scale * beta(tiles played, score) * (V(L) - max V)), where beta comes
// from a table fitted by maximum likelihood on true leaves in turn logs
// (per tiles played: [score_lo, score_hi, beta] bins).
//
// Configured by environment, read once per process:
//
//	MACONDO_LQ_TABLE      path of the beta table (unset = off)
//	MACONDO_LQ_SCALE      multiplier of beta (default 1)
//	MACONDO_LQ_MAXPLAYED  only after placements of at most this many tiles (default 2)
//	MACONDO_LQ_MINPLAYED  ... and at least this many (default 1)
//	MACONDO_LQ_MAXBAG     only while the bag holds at most this many tiles (default 100)
//	MACONDO_LQ_PURE_SIGNED  rule-only reads keep a negative beta (default: clipped at 0)
type lqConfig struct {
	table     map[string][][3]float64
	scale     float64
	maxPlayed int
	minPlayed int
	maxBag    int
	signed    bool
}

func envInt(name string, def int) int {
	s := os.Getenv(name)
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		log.Fatal().Err(err).Msg(name)
	}
	return n
}

// applies says whether the rule covers a placement of this many tiles with
// this many tiles in the bag.
func (c *lqConfig) applies(played, bag int) bool {
	return played >= c.minPlayed && played <= c.maxPlayed && bag <= c.maxBag
}

var (
	lqOnce sync.Once
	lqCfg  *lqConfig
)

func leaveQualityConfig() *lqConfig {
	lqOnce.Do(func() {
		path := os.Getenv("MACONDO_LQ_TABLE")
		if path == "" {
			return
		}
		b, err := os.ReadFile(path)
		if err != nil {
			log.Fatal().Err(err).Msg("MACONDO_LQ_TABLE")
		}
		c := &lqConfig{scale: 1, maxPlayed: 2}
		if err := json.Unmarshal(b, &c.table); err != nil {
			log.Fatal().Err(err).Msg("MACONDO_LQ_TABLE")
		}
		if s := os.Getenv("MACONDO_LQ_SCALE"); s != "" {
			if c.scale, err = strconv.ParseFloat(s, 64); err != nil {
				log.Fatal().Err(err).Msg("MACONDO_LQ_SCALE")
			}
		}
		if s := os.Getenv("MACONDO_LQ_MAXPLAYED"); s != "" {
			if c.maxPlayed, err = strconv.Atoi(s); err != nil {
				log.Fatal().Err(err).Msg("MACONDO_LQ_MAXPLAYED")
			}
		}
		c.minPlayed = envInt("MACONDO_LQ_MINPLAYED", 1)
		c.maxBag = envInt("MACONDO_LQ_MAXBAG", 100)
		c.signed = os.Getenv("MACONDO_LQ_PURE_SIGNED") != ""
		log.Info().Str("table", path).Float64("scale", c.scale).Int("maxPlayed", c.maxPlayed).
			Int("minPlayed", c.minPlayed).Int("maxBag", c.maxBag).Bool("signed", c.signed).
			Msg("leave-quality reweighting on")
		lqCfg = c
	})
	return lqCfg
}

func (c *lqConfig) beta(played, score int) (float64, bool) {
	bins := c.table[strconv.Itoa(played)]
	if len(bins) == 0 {
		return 0, false
	}
	for _, b := range bins {
		if float64(score) <= b[1] {
			return b[2], true
		}
	}
	return bins[len(bins)-1][2], true
}

// lastOppPlacement returns the number of tiles the opponent placed and the
// score of their last move, when it was a tile placement.
func lastOppPlacement(g *game.Game) (played, score int, ok bool) {
	evts := g.History().Events[:min(g.Turn(), len(g.History().Events))]
	if len(evts) == 0 {
		return 0, 0, false
	}
	opp := evts[len(evts)-1].PlayerIndex
	for i := len(evts) - 1; i >= 0 && evts[i].PlayerIndex == opp; i-- {
		switch evts[i].Type {
		case pb.GameEvent_CHALLENGE_BONUS:
			continue
		case pb.GameEvent_TILE_PLACEMENT_MOVE:
			return len([]rune(strings.ReplaceAll(evts[i].PlayedTiles, ".", ""))),
				int(evts[i].Score), true
		}
		return 0, 0, false
	}
	return 0, 0, false
}

// leaveQualityReweight returns racks reweighted by the leave-quality rule, or
// racks unchanged when the rule is off or does not apply.
func leaveQualityReweight(g *game.Game, racks []montecarlo.InferredRack,
	value func(tilemapping.MachineWord) float64) []montecarlo.InferredRack {

	c := leaveQualityConfig()
	if c == nil || value == nil || len(racks) == 0 {
		return racks
	}
	played, score, ok := lastOppPlacement(g)
	if !ok || !c.applies(played, g.Bag().TilesRemaining()) {
		return racks
	}
	beta, ok := c.beta(played, score)
	if !ok {
		return racks
	}
	b := c.scale * beta
	log.Info().Int("played", played).Int("score", score).Float64("beta", b).Msg("leave-quality reweight")
	vs := make([]float64, len(racks))
	top := math.Inf(-1)
	for i, r := range racks {
		l := slices.Clone(r.Leave)
		slices.Sort(l)
		vs[i] = value(tilemapping.MachineWord(l))
		top = max(top, vs[i])
	}
	out := make([]montecarlo.InferredRack, len(racks))
	for i, r := range racks {
		out[i] = montecarlo.InferredRack{Leave: r.Leave, Weight: r.Weight * math.Exp(b*(vs[i]-top))}
	}
	return out
}

// leaveValuer returns the static leave value function of the first calculator
// that has one, or nil.
func leaveValuer(calcs []equity.EquityCalculator) func(tilemapping.MachineWord) float64 {
	type valuer interface {
		LeaveValue(leave tilemapping.MachineWord) float64
	}
	for _, c := range calcs {
		if lv, ok := c.(valuer); ok {
			return lv.LeaveValue
		}
	}
	return nil
}

// Rule-only reads (experimental): a bot that runs no inference can still
// sample the opponent's leave from the tile-count prior times the
// leave-quality factor, over every leave the unseen pool allows. On when
// MACONDO_LQ_PURE_MAXPLAYED > 0 (placements of at most that many tiles), with
// the table and scale above, for the player nicknamed MACONDO_LQ_PURE_PLAYER
// (default p1) only, so a match against the same bot type is one-sided.
var (
	lqPureOnce   sync.Once
	lqPureMax    int
	lqPurePlayer string
)

// LeaveQualityPureOn says whether rule-only reads are configured; autoplay
// then keeps game history, which they read the opponent's last play from.
func LeaveQualityPureOn() bool { return leaveQualityPureMax() > 0 }

func leaveQualityPureMax() int {
	lqPureOnce.Do(func() {
		if s := os.Getenv("MACONDO_LQ_PURE_MAXPLAYED"); s != "" {
			n, err := strconv.Atoi(s)
			if err != nil {
				log.Fatal().Err(err).Msg("MACONDO_LQ_PURE_MAXPLAYED")
			}
			lqPureMax = n
			lqPurePlayer = os.Getenv("MACONDO_LQ_PURE_PLAYER")
			if lqPurePlayer == "" {
				lqPurePlayer = "p1"
			}
			log.Info().Int("maxPlayed", n).Str("player", lqPurePlayer).Msg("rule-only leave reads on")
		}
	})
	return lqPureMax
}

// leaveQualityPure returns the rule's read of the opponent's leave and its
// length, or ok false when the rule is off or does not apply.
func leaveQualityPure(g *game.Game, value func(tilemapping.MachineWord) float64) (
	[]montecarlo.InferredRack, int, bool) {

	maxPlayed := leaveQualityPureMax()
	c := leaveQualityConfig()
	if maxPlayed <= 0 || c == nil || value == nil || g.Bag().TilesRemaining() == 0 {
		return nil, 0, false
	}
	if g.History().Players[g.PlayerOnTurn()].Nickname != lqPurePlayer {
		return nil, 0, false
	}
	evts := g.History().Events[:min(g.Turn(), len(g.History().Events))]
	if len(evts) == 0 {
		return nil, 0, false
	}
	played, score, ok := lastOppPlacement(g)
	if !ok || played > maxPlayed || played < c.minPlayed || g.Bag().TilesRemaining() > c.maxBag {
		return nil, 0, false
	}
	// The opponent's rack before the play: logged in the event when known,
	// else a full rack.
	rackLen := 7
	for i := len(evts) - 1; i >= 0; i-- {
		if evts[i].Type == pb.GameEvent_TILE_PLACEMENT_MOVE {
			if n := len([]rune(evts[i].Rack)); n > 0 {
				rackLen = n
			}
			break
		}
	}
	k := rackLen - played
	beta, ok := c.beta(played, score)
	if !ok || k <= 0 {
		return nil, 0, false
	}
	b := c.scale * max(beta, 0)
	if c.signed {
		b = c.scale * beta
	}

	// Unseen from the mover's side: the bag and the opponent's rack.
	opp := g.NextPlayer()
	pool := make([]uint8, g.Alphabet().NumLetters()+1)
	for _, t := range g.Bag().Peek() {
		pool[t]++
	}
	for _, t := range g.RackFor(opp).TilesOn() {
		pool[t]++
	}
	leaves := enumerateLeaves(pool, k)
	if len(leaves) == 0 {
		return nil, 0, false
	}
	vs := make([]float64, len(leaves))
	top := math.Inf(-1)
	for i, l := range leaves {
		vs[i] = value(tilemapping.MachineWord(l)) // enumerated in sorted order
		top = max(top, vs[i])
	}
	out := make([]montecarlo.InferredRack, len(leaves))
	for i, l := range leaves {
		out[i] = montecarlo.InferredRack{Leave: l,
			Weight: multisetWays(l, pool) * math.Exp(b*(vs[i]-top))}
	}
	log.Info().Int("played", played).Int("score", score).Float64("beta", b).
		Int("leaves", len(leaves)).Msg("rule-only leave read")
	return out, k, true
}

// multisetWays is the number of ways to draw leave from pool: the
// hypergeometric prior up to a constant.
func multisetWays(leave []tilemapping.MachineLetter, pool []uint8) float64 {
	w := 1.0
	for i := 0; i < len(leave); {
		j := i
		for j < len(leave) && leave[j] == leave[i] {
			j++
		}
		n, r := int(pool[leave[i]]), j-i
		for x := 0; x < r; x++ {
			w *= float64(n-x) / float64(x+1)
		}
		i = j
	}
	return w
}

// enumerateLeaves lists every multiset of k tiles the pool allows, each in
// sorted tile order.
func enumerateLeaves(pool []uint8, k int) [][]tilemapping.MachineLetter {
	var out [][]tilemapping.MachineLetter
	buf := make([]tilemapping.MachineLetter, 0, k)
	var rec func(minIdx, remaining int)
	rec = func(minIdx, remaining int) {
		if remaining == 0 {
			out = append(out, slices.Clone(buf))
			return
		}
		for i := minIdx; i < len(pool); i++ {
			if pool[i] == 0 {
				continue
			}
			prev := len(buf)
			for c := 1; c <= int(pool[i]) && c <= remaining; c++ {
				buf = append(buf, tilemapping.MachineLetter(i))
				rec(i+1, remaining-c)
			}
			buf = buf[:prev]
		}
	}
	rec(0, k)
	return out
}
