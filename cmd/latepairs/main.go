// latepairs plays late-game positions out twice, with two bots swapping
// seats, to compare them only where they differ. Positions come from logged
// autoplay runs: in each logged game, the first turn that sees between
// -minbag and -maxbag tiles in the bag with the score within -margin. Each
// position is rebuilt from the log with the game's own seed, both racks as
// the logged game had them, and paired draws from there on, so the two
// playouts of a pair deal each seat the same tiles. Bot A is the variant
// (its sim solves the endgames its rollouts reach, set by the
// MACONDO_SIM_EG* environment), bot B the control.
//
//	latepairs -turns a.txt,b.txt -n 2500 -out pairs.jsonl
package main

import (
	"context"
	"encoding/base64"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/turnplayer"
)

type position struct {
	src, gid, first string
	rows            [][]string
	i               int // row index of the first move played out
	bag, spread     int
}

type playout struct {
	Spread int      `json:"spread"` // final spread for the position's mover
	Moves  []string `json:"moves"`
	Secs   float64  `json:"secs"`
	Err    string   `json:"err,omitempty"`
}

type record struct {
	Key    string `json:"key"` // src|gameID|first|row
	Bag    int    `json:"bag"`
	Spread int    `json:"spread"` // mover's spread at the position
	Rack   string `json:"rack"`
	// AMover: bot A sits in the mover's seat; BMover: bot B does.
	AMover playout `json:"aMover"`
	BMover playout `json:"bMover"`
	// AScore is bot A's result over the pair: 1 for two wins, 0.5 for a win
	// each (or two ties), 0 for two losses.
	AScore float64 `json:"aScore"`
	Err    string  `json:"err,omitempty"`
}

func atoi(s string) int { n, _ := strconv.Atoi(s); return n }

func readLog(path string) (map[string][][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	byGame := map[string][][]string{}
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if rec[0] == "playerID" || len(rec) < 12 {
			continue
		}
		byGame[rec[1]] = append(byGame[rec[1]], rec)
	}
	return byGame, nil
}

func seedOf(gid string) ([32]byte, error) {
	var seed [32]byte
	b, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(gid, "seed:"))
	if err != nil || len(b) != 32 {
		return seed, fmt.Errorf("game ID %q is not a seed", gid)
	}
	copy(seed[:], b)
	return seed, nil
}

// rebuild replays a logged game to just before row upto and seats both racks
// as the logged game had them there. It returns the mover's seat.
func rebuild(rules *game.GameRules, p position, upto int) (*turnplayer.BaseTurnPlayer, int, error) {
	seed, err := seedOf(p.gid)
	if err != nil {
		return nil, 0, err
	}
	players := []*pb.PlayerInfo{
		{Nickname: p.rows[0][0], RealName: p.rows[0][0]},
		{Nickname: p.rows[1][0], RealName: p.rows[1][0]},
	}
	g, err := turnplayer.BaseTurnPlayerFromRules(&turnplayer.GameOptions{
		BoardLayoutName: board.CrosswordGameLayout, Variant: game.VarClassic}, players, rules)
	if err != nil {
		return nil, 0, err
	}
	g.SetPairedBagMode(true)
	g.SeedBag(seed)
	g.StartGame()
	g.SetUidFromSeed(seed)
	for j := 0; j < upto; j++ {
		if err := automatic.PlayLoggedRow(g, p.rows[j], true); err != nil {
			return nil, 0, fmt.Errorf("replay row %d: %v", j+1, err)
		}
	}
	row := p.rows[upto]
	me := 0
	if g.History().Players[1].Nickname == row[0] {
		me = 1
	}
	// The mover's rack from this row, the opponent's from their next row
	// (nothing they hold depends on this move).
	oppRack := ""
	for j := upto + 1; j < len(p.rows); j++ {
		if p.rows[j][0] != row[0] {
			oppRack = p.rows[j][3]
			break
		}
	}
	if oppRack == "" {
		return nil, 0, fmt.Errorf("opponent's rack unknown")
	}
	racks := make([]*tilemapping.Rack, 2)
	racks[me] = tilemapping.RackFromString(row[3], g.Alphabet())
	racks[1-me] = tilemapping.RackFromString(oppRack, g.Alphabet())
	if err := g.Game.SetRacksForBoth(racks); err != nil {
		return nil, 0, fmt.Errorf("seating racks: %v", err)
	}
	return g, me, nil
}

// findPosition picks the first turn of a logged game that sees minBag..maxBag
// tiles in the bag with the mover's spread within margin.
func findPosition(rules *game.GameRules, p position, minBag, maxBag, margin int) (position, bool) {
	seed, err := seedOf(p.gid)
	if err != nil {
		return p, false
	}
	players := []*pb.PlayerInfo{
		{Nickname: p.rows[0][0], RealName: p.rows[0][0]},
		{Nickname: p.rows[1][0], RealName: p.rows[1][0]},
	}
	g, err := turnplayer.BaseTurnPlayerFromRules(&turnplayer.GameOptions{
		BoardLayoutName: board.CrosswordGameLayout, Variant: game.VarClassic}, players, rules)
	if err != nil {
		return p, false
	}
	g.SetPairedBagMode(true)
	g.SeedBag(seed)
	g.StartGame()
	for j := 0; j < len(p.rows); j++ {
		bag := g.Bag().TilesRemaining()
		if bag < minBag {
			return p, false
		}
		if j > 0 && bag <= maxBag {
			me := 0
			if g.History().Players[1].Nickname == p.rows[j][0] {
				me = 1
			}
			sp := g.SpreadFor(me)
			if sp < -margin || sp > margin {
				return p, false
			}
			p.i, p.bag, p.spread = j, bag, sp
			return p, true
		}
		if err := automatic.PlayLoggedRow(g, p.rows[j], true); err != nil {
			return p, false
		}
	}
	return p, false
}

var endgameSecs = 15 * time.Second

// play finishes the game with bots[seat] moving for each seat, and returns
// the final spread for seat me.
func play(g *turnplayer.BaseTurnPlayer, bots [2]*bot.BotTurnPlayer, me int) playout {
	var out playout
	t0 := time.Now()
	for g.Playing() == pb.PlayState_PLAYING {
		onTurn := g.PlayerOnTurn()
		var ctx context.Context
		var cancel context.CancelFunc
		if g.Bag().TilesRemaining() == 0 {
			ctx, cancel = context.WithTimeout(context.Background(), endgameSecs)
		} else {
			// As in paired autoplay: no clock before the endgame, so a sim
			// stops on its own cutoff and a position plays the same way twice.
			ctx, cancel = context.WithCancel(context.Background())
		}
		m, err := bots[onTurn].BestPlay(ctx)
		cancel()
		if err != nil || m == nil {
			out.Err = fmt.Sprintf("best play: %v", err)
			return out
		}
		out.Moves = append(out.Moves, m.ShortDescription())
		if err := g.PlayMove(m, true, 0); err != nil {
			out.Err = fmt.Sprintf("play %s: %v", m.ShortDescription(), err)
			return out
		}
		bots[0].AddLastMove(m)
		bots[1].AddLastMove(m)
	}
	out.Spread = g.SpreadFor(me)
	out.Secs = time.Since(t0).Seconds()
	return out
}

func winOf(spread int) float64 {
	switch {
	case spread > 0:
		return 1
	case spread < 0:
		return 0
	}
	return 0.5
}

func run(rules *game.GameRules, confs [2]bot.BotConfig, code pb.BotRequest_BotCode, p position) record {
	rec := record{Key: fmt.Sprintf("%s|%s|%s|%d", p.src, p.gid, p.first, p.i), Bag: p.bag,
		Spread: p.spread, Rack: p.rows[p.i][3]}
	// variant k: bot A in the mover's seat (k = 0) or bot B (k = 1).
	for k := 0; k < 2; k++ {
		g, me, err := rebuild(rules, p, p.i)
		if err != nil {
			rec.Err = err.Error()
			return rec
		}
		// One transposition table per game, shared by its two bots, which
		// never solve at the same time.
		tt := &negamax.TranspositionTable{}
		var bots [2]*bot.BotTurnPlayer
		for seat := 0; seat < 2; seat++ {
			which := 0 // bot A
			if (seat == me) == (k == 1) {
				which = 1
			}
			conf := confs[which]
			conf.EndgameTTable = tt
			b, err := bot.NewBotTurnPlayerFromGame(g.Game, &conf, code)
			if err != nil {
				rec.Err = err.Error()
				return rec
			}
			bots[seat] = b
		}
		po := play(g, bots, me)
		if k == 0 {
			rec.AMover = po
		} else {
			rec.BMover = po
		}
		if po.Err != "" {
			rec.Err = po.Err
			return rec
		}
	}
	rec.AScore = (winOf(rec.AMover.Spread) + 1 - winOf(rec.BMover.Spread)) / 2
	return rec
}

func main() {
	turns := flag.String("turns", "", "per-turn logs of seeded game-pair runs, comma-separated")
	botCode := flag.String("bot", "SIMMING_BOT", "bot code of both bots")
	plies := flag.Int("plies", 2, "minSimPlies")
	minBag := flag.Int("minbag", 2, "first turn that sees at least this many tiles in the bag ...")
	maxBag := flag.Int("maxbag", 6, "... and at most this many")
	margin := flag.Int("margin", 60, "only positions with the mover's spread within this many points")
	n := flag.Int("n", 0, "sample this many positions (0 = all)")
	sample := flag.Int64("sample", 1, "sampling seed")
	ttFraction := flag.Float64("ttfrac", 0.003, "memory fraction of each game's endgame table")
	egSecs := flag.Int("egsecs", 15, "endgame solve budget, seconds")
	out := flag.String("out", "latepairs.jsonl", "output")
	parallel := flag.Int("parallel", 8, "pairs at once (each single-threaded)")
	flag.Parse()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	endgameSecs = time.Duration(*egSecs) * time.Second

	code, ok := pb.BotRequest_BotCode_value[*botCode]
	if !ok {
		log.Fatal().Str("bot", *botCode).Msg("unknown bot code")
	}
	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, "NWL23")
	cfg.Set(config.ConfigDefaultLetterDistribution, "english")
	rules, err := game.NewBasicGameRules(cfg, "NWL23", board.CrosswordGameLayout,
		"english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		log.Fatal().Err(err).Msg("rules")
	}

	var cands []position
	for _, path := range strings.Split(*turns, ",") {
		byGame, err := readLog(path)
		if err != nil {
			log.Fatal().Err(err).Str("log", path).Msg("turns")
		}
		src := strings.TrimSuffix(path[strings.LastIndex(path, "/")+1:], ".txt")
		for gid, rows := range byGame {
			for _, h := range automatic.SplitTurnHalves(rows) {
				cands = append(cands, position{src: src, gid: gid, first: h[0][0], rows: h})
			}
		}
	}
	// Map order is random; sort so a sample is the same every run.
	sort.Slice(cands, func(a, b int) bool {
		x, y := cands[a], cands[b]
		if x.src != y.src {
			return x.src < y.src
		}
		if x.gid != y.gid {
			return x.gid < y.gid
		}
		return x.first < y.first
	})
	rand.New(rand.NewSource(*sample)).Shuffle(len(cands), func(a, b int) { cands[a], cands[b] = cands[b], cands[a] })
	fmt.Fprintf(os.Stderr, "%d logged games\n", len(cands))

	var pos []position
	for _, c := range cands {
		if *n > 0 && len(pos) >= *n {
			break
		}
		if p, ok := findPosition(rules, c, *minBag, *maxBag, *margin); ok {
			pos = append(pos, p)
		}
	}
	fmt.Fprintf(os.Stderr, "%d positions\n", len(pos))

	confs := [2]bot.BotConfig{}
	for i := range confs {
		confs[i] = bot.BotConfig{Config: *cfg, MinSimPlies: *plies, SimThreads: 1,
			EndgameTTableFraction: *ttFraction, EndgameThreads: 1, PreendgameThreads: 1}
	}
	confs[0].SimEndgameLeaves = true

	f, err := os.Create(*out)
	if err != nil {
		log.Fatal().Err(err).Msg("out")
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	var mu sync.Mutex
	var done atomic.Int64
	var scoreSum float64
	var scored int
	jobs := make(chan position)
	var wg sync.WaitGroup
	for w := 0; w < *parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range jobs {
				rec := run(rules, confs, pb.BotRequest_BotCode(code), p)
				mu.Lock()
				enc.Encode(rec)
				if rec.Err == "" {
					scoreSum += rec.AScore
					scored++
				}
				d := done.Add(1)
				if d%50 == 0 {
					fmt.Fprintf(os.Stderr, "%d/%d pairs, A %.2f%% over %d\n", d, len(pos),
						100*scoreSum/float64(max(1, scored)), scored)
				}
				mu.Unlock()
			}
		}()
	}
	for _, p := range pos {
		jobs <- p
	}
	close(jobs)
	wg.Wait()
	fmt.Fprintf(os.Stderr, "done: %d pairs, A %.2f%% over %d scored; endgame leaves: %d solved (%.2f ms each), %d fell back\n",
		len(pos), 100*scoreSum/float64(max(1, scored)), scored,
		montecarlo.EndgameLeafSolves.Load(),
		float64(montecarlo.EndgameLeafNanos.Load())/1e6/float64(max(1, montecarlo.EndgameLeafSolves.Load())),
		montecarlo.EndgameLeafFallbacks.Load())
}
