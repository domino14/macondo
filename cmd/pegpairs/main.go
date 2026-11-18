// pegpairs plays pre-endgame positions out twice, with two bots swapping
// seats, to price how much a stronger pre-endgame is worth to the value net.
// Positions come from logged autoplay runs: in each logged game, the first
// turn that sees between -minbag and -maxbag tiles in the bag with the score
// within -margin. Each position is rebuilt from the log with the game's own
// seed, both racks as the logged game had them, and paired draws from there
// on, so the two playouts of a pair deal each seat the same tiles.
//
// Both bots play the net (FAST_ML_BOT, -model on Triton) while the bag holds
// more than -hybrid tiles, and both solve the endgame (empty bag) with the
// full solver. In between, bot A (the variant) plays BestBot (SIMMING_BOT:
// the pre-endgame solver at one tile in the bag, sims with endgame leaves at
// two, deep sims above) and bot B (the control) keeps playing the net.
//
//	pegpairs -turns match.txt -model macondo-nn-tf-nwl23l12 -n 3000 -out pairs.jsonl
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
	// HybridTurns counts the pre-endgame turns bot A played as BestBot.
	HybridTurns int    `json:"hybridTurns"`
	Err         string `json:"err,omitempty"`
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
	if err != nil || len(p.rows) < 2 {
		// A snapshot log can end mid-game with a one-row half.
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

// pegSecs (-pegsecs) budgets the pre-endgame solve at one tile in the bag.
var pegSecs time.Duration

// hybridMax (-hybrid): bot A plays BestBot with 1..hybridMax tiles in the bag.
var hybridMax int

// lateBag (-late-bag): with -late-model, both seats play that net with
// 1..lateBag tiles in the bag, so A and B differ only earlier in the game.
var lateBag int

// lateAOnly (-late-a): only bot A switches to -late-model; B keeps its net.
var lateAOnly bool

// seatBots holds the two engines one seat can move with.
type seatBots struct {
	net, best *bot.BotTurnPlayer
	late      *bot.BotTurnPlayer // with -late-model: both seats' net at 1..lateBag tiles
	isA       bool
}

// play finishes the game and returns the final spread for seat me. It counts
// the turns bot A played as BestBot.
func play(g *turnplayer.BaseTurnPlayer, seats [2]seatBots, me int, key string) (playout, int) {
	var out playout
	hybridTurns := 0
	t0 := time.Now()
	for g.Playing() == pb.PlayState_PLAYING {
		onTurn := g.PlayerOnTurn()
		bag := g.Bag().TilesRemaining()
		var ctx context.Context
		var cancel context.CancelFunc
		switch {
		case bag == 0:
			ctx, cancel = context.WithTimeout(context.Background(), endgameSecs)
		case bag == 1 && pegSecs > 0:
			ctx, cancel = context.WithTimeout(context.Background(), pegSecs)
		default:
			// No clock: a sim stops on its own cutoff, so a position plays
			// the same way twice.
			ctx, cancel = context.WithCancel(context.Background())
		}
		ctx = log.Logger.With().Str("key", key).Int("turn", g.Turn()).Logger().WithContext(ctx)
		b := seats[onTurn].net
		if seats[onTurn].late != nil && bag >= 1 && bag <= lateBag {
			b = seats[onTurn].late
		}
		if bag == 0 || (seats[onTurn].isA && bag <= hybridMax) {
			b = seats[onTurn].best
			if bag > 0 {
				hybridTurns++
			}
		}
		m, err := b.BestPlay(ctx)
		cancel()
		if err != nil || m == nil {
			out.Err = fmt.Sprintf("best play (bag %d): %v", bag, err)
			return out, hybridTurns
		}
		out.Moves = append(out.Moves, fmt.Sprintf("%d:%s", bag, m.ShortDescription()))
		if err := g.PlayMove(m, true, 0); err != nil {
			out.Err = fmt.Sprintf("play %s: %v", m.ShortDescription(), err)
			return out, hybridTurns
		}
		for _, sb := range seats {
			sb.net.AddLastMove(m)
			sb.best.AddLastMove(m)
			if sb.late != nil {
				sb.late.AddLastMove(m)
			}
		}
	}
	out.Spread = g.SpreadFor(me)
	out.Secs = time.Since(t0).Seconds()
	return out, hybridTurns
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

func run(rules *game.GameRules, netConf, netConfA, bestConf bot.BotConfig, lateConf *bot.BotConfig, p position) record {
	rec := record{Key: fmt.Sprintf("%s|%s|%s|%d", p.src, p.gid, p.first, p.i), Bag: p.bag,
		Spread: p.spread, Rack: p.rows[p.i][3]}
	// k = 0: bot A in the mover's seat; k = 1: bot B there.
	for k := 0; k < 2; k++ {
		g, me, err := rebuild(rules, p, p.i)
		if err != nil {
			rec.Err = err.Error()
			return rec
		}
		// One transposition table per game, shared by its bots, which never
		// solve at the same time.
		tt := &negamax.TranspositionTable{}
		var seats [2]seatBots
		for seat := 0; seat < 2; seat++ {
			isA := (seat == me) == (k == 0)
			nc, bc := netConf, bestConf
			if isA {
				nc = netConfA
			}
			nc.EndgameTTable, bc.EndgameTTable = tt, tt
			nb, err := bot.NewBotTurnPlayerFromGame(g.Game, &nc, pb.BotRequest_FAST_ML_BOT)
			if err != nil {
				rec.Err = err.Error()
				return rec
			}
			bb, err := bot.NewBotTurnPlayerFromGame(g.Game, &bc, pb.BotRequest_SIMMING_BOT)
			if err != nil {
				rec.Err = err.Error()
				return rec
			}
			seats[seat] = seatBots{net: nb, best: bb, isA: isA}
			if lateConf != nil && (isA || !lateAOnly) {
				lc := *lateConf
				lc.EndgameTTable = tt
				lb, err := bot.NewBotTurnPlayerFromGame(g.Game, &lc, pb.BotRequest_FAST_ML_BOT)
				if err != nil {
					rec.Err = err.Error()
					return rec
				}
				seats[seat].late = lb
			}
		}
		po, ht := play(g, seats, me, fmt.Sprintf("%s|k%d", rec.Key, k))
		rec.HybridTurns += ht
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
	model := flag.String("model", "macondo-nn-tf-nwl23l12", "Triton model both bots' net plays with")
	modelA := flag.String("model-a", "", "Triton model bot A's net plays with instead of -model (net vs net; use with -hybrid 0)")
	lateModel := flag.String("late-model", "", "Triton model both seats' net plays with at 1..-late-bag tiles in the bag")
	flag.BoolVar(&lateAOnly, "late-a", false, "with -late-model: only bot A switches to it (B keeps -model throughout)")
	flag.IntVar(&lateBag, "late-bag", 6, "with -late-model: the most tiles in the bag at which both seats play it")
	tritonURL := flag.String("triton", "localhost:8101", "Triton gRPC address")
	lexicon := flag.String("lexicon", "NWL23", "the logged games' lexicon")
	minBag := flag.Int("minbag", 1, "first turn that sees at least this many tiles in the bag ...")
	maxBag := flag.Int("maxbag", 6, "... and at most this many")
	margin := flag.Int("margin", 60, "only positions with the mover's spread within this many points")
	n := flag.Int("n", 0, "sample this many positions (0 = all)")
	sample := flag.Int64("sample", 1, "sampling seed")
	only := flag.String("key", "", "play only the position whose key starts with this (to reproduce one)")
	skip := flag.Int("skip", 0, "skip this many sampled positions first (to split a sample across machines)")
	ttFraction := flag.Float64("ttfrac", 0.003, "memory fraction of each game's endgame table")
	egSecs := flag.Int("egsecs", 15, "endgame solve budget, seconds (both bots)")
	pegs := flag.Int("pegsecs", 30, "bot A's pre-endgame solve budget at one tile in the bag, seconds")
	flag.IntVar(&hybridMax, "hybrid", 6, "bot A plays BestBot with 1..this many tiles in the bag")
	out := flag.String("out", "pegpairs.jsonl", "output")
	parallel := flag.Int("parallel", 8, "pairs at once (each single-threaded)")
	flag.Parse()
	pegSecs = time.Duration(*pegs) * time.Second
	endgameSecs = time.Duration(*egSecs) * time.Second
	zerolog.SetGlobalLevel(zerolog.WarnLevel)

	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, *lexicon)
	cfg.Set(config.ConfigDefaultLetterDistribution, "english")
	cfg.Set(config.ConfigTritonUseTriton, true)
	cfg.Set(config.ConfigTritonURL, *tritonURL)
	rules, err := game.NewBasicGameRules(cfg, *lexicon, board.CrosswordGameLayout,
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
		if *n > 0 && len(pos) >= *skip+*n {
			break
		}
		if *only != "" && !strings.HasPrefix(fmt.Sprintf("%s|%s|%s|", c.src, c.gid, c.first), *only) {
			continue
		}
		if p, ok := findPosition(rules, c, *minBag, *maxBag, *margin); ok {
			pos = append(pos, p)
		}
	}
	pos = pos[min(*skip, len(pos)):]
	fmt.Fprintf(os.Stderr, "%d positions\n", len(pos))

	netConf := bot.BotConfig{Config: *cfg, TritonModel: *model}
	netConfA := netConf
	var lateConf *bot.BotConfig
	if *lateModel != "" {
		lc := netConf
		lc.TritonModel = *lateModel
		lateConf = &lc
	}
	if *modelA != "" {
		netConfA.TritonModel = *modelA
	}
	bestConf := bot.BotConfig{Config: *cfg, MinSimPlies: 2, SimThreads: 1,
		EndgameTTableFraction: *ttFraction, EndgameThreads: 1, PreendgameThreads: 1}

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
				rec := run(rules, netConf, netConfA, bestConf, lateConf, p)
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
	fmt.Fprintf(os.Stderr, "done: %d pairs, A %.2f%% over %d scored\n",
		len(pos), 100*scoreSum/float64(max(1, scored)), scored)
}
