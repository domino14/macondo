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
type lqConfig struct {
	table     map[string][][3]float64
	scale     float64
	maxPlayed int
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
		log.Info().Str("table", path).Float64("scale", c.scale).Int("maxPlayed", c.maxPlayed).
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
	evts := g.History().Events[:g.Turn()]
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
	if !ok || played > c.maxPlayed {
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
