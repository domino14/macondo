// pegtiming prices pre-endgame choices by playing each candidate move out
// many times against a realistic opponent. Positions are the first turn of a
// logged game with exactly -bag tiles in the bag (score within -margin). The
// candidates are the net's pick, the net's best move that keeps tiles in the
// bag, its best move that empties the bag, and BestBot's pick. Each one is
// played out -trials times: the opponent's rack is redrawn at random from
// the unseen tiles (the mover can't see it), draws are random, and BestBot
// plays both seats from there (the pre-endgame solver at one tile in the bag
// without seeing the bag, the full solver once the bag is empty). Trial i
// uses the same seed for every candidate. The pre-endgame solver's own value
// for each candidate (its convention: the opponent answers each bag draw as
// if it knew it) is recorded beside, with -pegrank.
//
//	pegtiming -turns match.txt -bag 2 -n 300 -trials 48 -out t.jsonl
package main

import (
	"context"
	"crypto/sha256"
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

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/preendgame"
	"github.com/domino14/macondo/turnplayer"
)

type position struct {
	src, gid, first string
	rows            [][]string
	i               int // row index of the first move played out
	bag, spread     int
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

var endgameSecs, pegSecs time.Duration

type cand struct {
	Move    string    `json:"move"`
	Tiles   int       `json:"tiles"`
	Left    int       `json:"left"` // tiles left in the bag after it
	Equity  float64   `json:"equity"`
	EqRank  int       `json:"eqRank"` // 1-based rank by static equity
	Net     float32   `json:"net"`
	NetRank int       `json:"netRank"`
	Roles   []string  `json:"roles"`
	Wins    float64   `json:"wins"` // mean result over trials (ties half)
	Spread  float64   `json:"spread"`
	Trials  int       `json:"trials"`
	Errs    int       `json:"errs"`
	PEG     float64   `json:"peg"` // pre-endgame solver win fraction (-pegrank), -1 if not run
	Results []float64 `json:"results"`
	m       *move.Move
}

type record struct {
	Key    string  `json:"key"`
	Bag    int     `json:"bag"`
	Spread int     `json:"spread"`
	Rack   string  `json:"rack"`
	NMoves int     `json:"nMoves"`
	Cands  []*cand `json:"cands"`
	Secs   float64 `json:"secs"`
	Err    string  `json:"err,omitempty"`
}

func trialSeed(key string, i int) [32]byte {
	return sha256.Sum256([]byte(fmt.Sprintf("%s#%d", key, i)))
}

// trial plays candidate c out once: the opponent's rack redrawn from the
// unseen tiles, BestBot in both seats after c. It returns the mover's result.
func trial(rules *game.GameRules, bestConf bot.BotConfig, p position, c *cand, seed [32]byte) (float64, int, error) {
	g, me, err := rebuild(rules, p, p.i)
	if err != nil {
		return 0, 0, err
	}
	// Random draws from here on: the rebuilt game's bag is in paired
	// fixed order, which would deal every trial the same tiles.
	g.SetPairedBagMode(false)
	g.Bag().SetFixedOrder(false)
	g.SeedBag(seed)
	if _, err := g.SetRandomRack(1-me, nil); err != nil {
		return 0, 0, fmt.Errorf("opp rack: %v", err)
	}
	if err := g.PlayMove(c.m, true, 0); err != nil {
		return 0, 0, fmt.Errorf("play %s: %v", c.Move, err)
	}
	tt := &negamax.TranspositionTable{}
	var bots [2]*bot.BotTurnPlayer
	for seat := 0; seat < 2; seat++ {
		bc := bestConf
		bc.EndgameTTable = tt
		b, err := bot.NewBotTurnPlayerFromGame(g.Game, &bc, pb.BotRequest_SIMMING_BOT)
		if err != nil {
			return 0, 0, err
		}
		b.AddLastMove(c.m)
		bots[seat] = b
	}
	for g.Playing() == pb.PlayState_PLAYING {
		bag := g.Bag().TilesRemaining()
		var ctx context.Context
		var cancel context.CancelFunc
		switch {
		case bag == 0:
			ctx, cancel = context.WithTimeout(context.Background(), endgameSecs)
		case bag == 1:
			ctx, cancel = context.WithTimeout(context.Background(), pegSecs)
		default:
			ctx, cancel = context.WithCancel(context.Background())
		}
		m, err := bots[g.PlayerOnTurn()].BestPlay(ctx)
		cancel()
		if err != nil || m == nil {
			return 0, 0, fmt.Errorf("best play (bag %d): %v", bag, err)
		}
		if err := g.PlayMove(m, true, 0); err != nil {
			return 0, 0, err
		}
		bots[0].AddLastMove(m)
		bots[1].AddLastMove(m)
	}
	sp := g.SpreadFor(me)
	switch {
	case sp > 0:
		return 1, sp, nil
	case sp < 0:
		return 0, sp, nil
	}
	return 0.5, sp, nil
}

func analyse(rules *game.GameRules, netConf, bestConf bot.BotConfig, p position, trials int, pegRank bool, gd *kwg.KWG) record {
	t0 := time.Now()
	rec := record{Key: fmt.Sprintf("%s|%s|%s|%d", p.src, p.gid, p.first, p.i), Bag: p.bag,
		Spread: p.spread, Rack: p.rows[p.i][3]}
	g, _, err := rebuild(rules, p, p.i)
	if err != nil {
		rec.Err = err.Error()
		return rec
	}
	nb, err := bot.NewBotTurnPlayerFromGame(g.Game, &netConf, pb.BotRequest_FAST_ML_BOT)
	if err != nil {
		rec.Err = err.Error()
		return rec
	}
	moves := nb.GenerateMoves(100000)
	rec.NMoves = len(moves)
	vals, err := nb.MLScores(moves)
	if err != nil {
		rec.Err = "net: " + err.Error()
		return rec
	}
	bag := g.Bag().TilesRemaining()
	order := make([]int, len(moves))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return vals[order[a]] > vals[order[b]] })
	netRank := make([]int, len(moves))
	for r, i := range order {
		netRank[i] = r + 1
	}
	byKey := map[string]*cand{}
	add := func(i int, role string) {
		m := moves[i]
		k := m.ShortDescription()
		c, ok := byKey[k]
		if !ok {
			c = &cand{Move: k, Tiles: m.TilesPlayed(), Left: max(0, bag-m.TilesPlayed()), Equity: m.Equity(),
				EqRank: i + 1, Net: vals[i], NetRank: netRank[i], PEG: -1, m: m}
			byKey[k] = c
			rec.Cands = append(rec.Cands, c)
		}
		c.Roles = append(c.Roles, role)
	}
	add(order[0], "net")
	keptDone, emptyDone := false, false
	for _, i := range order {
		m := moves[i]
		if m.Action() == move.MoveTypePass || m.Action() == move.MoveTypeExchange {
			continue
		}
		if m.TilesPlayed() < bag && !keptDone {
			add(i, "netKeep")
			keptDone = true
		}
		if m.TilesPlayed() >= bag && !emptyDone {
			add(i, "netEmpty")
			emptyDone = true
		}
	}
	// The ML bot only scores the top candidates by equity: its pick there.
	topN := min(50, len(moves))
	bi := 0
	for i := 1; i < topN; i++ {
		if vals[i] > vals[bi] {
			bi = i
		}
	}
	add(bi, "netTop50")
	// BestBot's pick at this position.
	bc := bestConf
	bc.EndgameTTable = &negamax.TranspositionTable{}
	bb, err := bot.NewBotTurnPlayerFromGame(g.Game, &bc, pb.BotRequest_SIMMING_BOT)
	if err == nil {
		ctx, cancel := context.WithCancel(context.Background())
		if bag == 1 {
			ctx, cancel = context.WithTimeout(context.Background(), pegSecs)
		}
		if bm, err := bb.BestPlay(ctx); err == nil && bm != nil {
			found := false
			for i, m := range moves {
				if m.Equals(bm, false, false) {
					add(i, "bestbot")
					found = true
					break
				}
			}
			if !found {
				rec.Err = "bestbot move not in the move list: " + bm.ShortDescription()
			}
		}
		cancel()
	}
	if pegRank && bag <= 2 {
		var solveOnly []*move.Move
		for _, c := range rec.Cands {
			solveOnly = append(solveOnly, c.m)
		}
		s := &preendgame.Solver{}
		if err := s.Init(g.Game, gd); err == nil {
			s.SetThreads(1)
			s.SetTTable(&negamax.TranspositionTable{})
			s.SetTTableFraction(0.003)
			s.SetEndgamePlies(5)
			s.SetIterativeDeepening(true)
			s.SetEarlyCutoffOptim(false)
			s.SetSolveOnly(solveOnly)
			ctx, cancel := context.WithTimeout(context.Background(), 4*pegSecs)
			plays, err := s.Solve(ctx)
			cancel()
			if err == nil {
				for _, pl := range plays {
					if c, ok := byKey[pl.Play.ShortDescription()]; ok && !pl.Ignore {
						if n := pl.TotalOutcomes(); n > 0 {
							c.PEG = float64(pl.Points) / float64(n)
						}
					}
				}
			}
		}
	}
	for _, c := range rec.Cands {
		var sum, spr float64
		for t := 0; t < trials; t++ {
			w, sp, err := trial(rules, bestConf, p, c, trialSeed(rec.Key, t))
			if err != nil {
				c.Errs++
				c.Results = append(c.Results, -1)
				continue
			}
			c.Results = append(c.Results, w)
			sum += w
			spr += float64(sp)
			c.Trials++
		}
		if c.Trials > 0 {
			c.Wins = sum / float64(c.Trials)
			c.Spread = spr / float64(c.Trials)
		}
	}
	rec.Secs = time.Since(t0).Seconds()
	return rec
}

func main() {
	turns := flag.String("turns", "", "per-turn logs of seeded game-pair runs, comma-separated")
	model := flag.String("model", "macondo-nn-tf-nwl23l12", "Triton model for the net's ranking")
	tritonURL := flag.String("triton", "localhost:8101", "Triton gRPC address")
	lexicon := flag.String("lexicon", "NWL23", "the logged games' lexicon")
	bagN := flag.Int("bag", 2, "positions with exactly this many tiles in the bag")
	margin := flag.Int("margin", 60, "only positions with the mover's spread within this many points")
	n := flag.Int("n", 0, "this many positions (0 = all)")
	sample := flag.Int64("sample", 1, "sampling seed")
	skip := flag.Int("skip", 0, "skip this many sampled positions first")
	trials := flag.Int("trials", 48, "playouts per candidate")
	pegRank := flag.Bool("pegrank", true, "also run the pre-endgame solver on the candidates (bag <= 2)")
	egs := flag.Int("egsecs", 10, "endgame solve budget, seconds")
	pegs := flag.Int("pegsecs", 20, "pre-endgame solve budget at one tile in the bag, seconds")
	out := flag.String("out", "pegtiming.jsonl", "output")
	parallel := flag.Int("parallel", 8, "candidate playouts at once (positions in flight)")
	flag.Parse()
	endgameSecs = time.Duration(*egs) * time.Second
	pegSecs = time.Duration(*pegs) * time.Second
	zerolog.SetGlobalLevel(zerolog.WarnLevel)

	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, *lexicon)
	cfg.Set(config.ConfigDefaultLetterDistribution, "english")
	cfg.Set(config.ConfigTritonUseTriton, true)
	cfg.Set(config.ConfigTritonURL, *tritonURL)
	cfg.Set(config.ConfigTritonModelName, *model)
	rules, err := game.NewBasicGameRules(cfg, *lexicon, board.CrosswordGameLayout,
		"english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		log.Fatal().Err(err).Msg("rules")
	}
	gd, err := kwg.GetKWG(cfg.WGLConfig(), *lexicon)
	if err != nil {
		log.Fatal().Err(err).Msg("kwg")
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
		if p, ok := findPosition(rules, c, *bagN, *bagN, *margin); ok {
			pos = append(pos, p)
		}
	}
	pos = pos[min(*skip, len(pos)):]
	fmt.Fprintf(os.Stderr, "%d positions\n", len(pos))

	netConf := bot.BotConfig{Config: *cfg, TritonModel: *model}
	bestConf := bot.BotConfig{Config: *cfg, MinSimPlies: 2, SimThreads: 1,
		EndgameTTableFraction: 0.003, EndgameThreads: 1, PreendgameThreads: 1}

	f, err := os.Create(*out)
	if err != nil {
		log.Fatal().Err(err).Msg("out")
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	var mu sync.Mutex
	var done atomic.Int64
	jobs := make(chan position)
	var wg sync.WaitGroup
	for w := 0; w < *parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range jobs {
				rec := analyse(rules, netConf, bestConf, p, *trials, *pegRank, gd)
				mu.Lock()
				enc.Encode(rec)
				if d := done.Add(1); d%10 == 0 {
					fmt.Fprintf(os.Stderr, "%s %d/%d positions\n", time.Now().Format("15:04:05"), d, len(pos))
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
	fmt.Fprintf(os.Stderr, "done: %d positions\n", len(pos))
}
