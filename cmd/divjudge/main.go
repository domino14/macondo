// divjudge finds where two paired autoplay runs first part ways and judges
// the two moves. Runs that share a seed play identical games until one bot
// chooses differently; that first differing move is played from the same
// position in both runs, so a sim that knows the opponent's real leave can
// say which choice was better, and the games' results say which side the
// divergence ended up favouring.
//
//	divjudge -a lq.txt -b base.txt -ga games-lq.txt -gb games-base.txt -out div.jsonl
//
// With -mode screen it judges screendecide output instead: positions from
// the baseline log (-b) where a variant's move (-decide) differs from the
// control's (-control, or else the logged move), with no games played.
//
//	divjudge -mode screen -b base.txt -decide var.jsonl [-control ctl.jsonl] -out s.jsonl
package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"
	"sync"
	"sync/atomic"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/preendgame"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/rangefinder"
	"github.com/domino14/macondo/turnplayer"
)

type half struct {
	gid, first string // game ID and the player ID moving first
	rows       [][]string
}

// readTurns returns each seating of each game in a per-turn log, keyed by
// game ID and first mover.
func readTurns(path string) (map[string]*half, error) {
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
	out := map[string]*half{}
	for gid, rows := range byGame {
		for _, h := range automatic.SplitTurnHalves(rows) {
			out[gid+"|"+h[0][0]] = &half{gid: gid, first: h[0][0], rows: h}
		}
	}
	return out, nil
}

// readResults returns p1's result (1, 0.5, 0) and final spread for each
// seating, keyed like readTurns. In the games file, "first" names a bot; the
// bot in the first score column is p1.
func readResults(path string) (map[string][2]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	hdr, err := r.Read()
	if err != nil {
		return nil, err
	}
	p1Name := strings.TrimSuffix(hdr[1], "_score")
	out := map[string][2]float64{}
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		a, err1 := strconv.Atoi(rec[1])
		b, err2 := strconv.Atoi(rec[2])
		if err1 != nil || err2 != nil {
			continue
		}
		first := "p2"
		if rec[7] == p1Name {
			first = "p1"
		}
		res := 0.0
		if a > b {
			res = 1
		} else if a == b {
			res = 0.5
		}
		out[rec[0]+"|"+first] = [2]float64{res, float64(a - b)}
	}
	return out, nil
}

type judged struct {
	Move  string  `json:"move"`
	Score int     `json:"score"`
	Tiles int     `json:"tiles"`
	Leave string  `json:"leave"`
	Win   float64 `json:"win"`            // judge win %, opponent's leave known
	SE    float64 `json:"se"`             // its standard error
	WinR  float64 `json:"winR"`           // judge win %, opponent's leave unknown (random racks)
	SER   float64 `json:"seR"`            // its standard error
	WinI  float64 `json:"winI"` // judge win %, opponent's rack from the mover's inference (-inferjudge)
	SEI   float64 `json:"seI"`  // its standard error
	// -pegjudge: the pre-endgame solver's win share (draws half, the
	// solver's pessimistic convention) with the opponent's rack unknown
	// (PegU) and their last leave known (PegK); nil if not solved.
	PegU *float64 `json:"pegU,omitempty"`
	PegK *float64 `json:"pegK,omitempty"`
	Eq    float64 `json:"equity"`         // logged equity
}

type record struct {
	GameID    string  `json:"gameID"`
	First     string  `json:"first"`
	Turn      int     `json:"turn"`
	Mover     string  `json:"mover"`
	Rack      string  `json:"rack"`
	Bag       int     `json:"bag"`
	Spread    int     `json:"spread"` // mover's spread before the move
	OppPlay   string  `json:"oppPlay"`
	OppScore  int     `json:"oppScore"`
	OppPlayed int     `json:"oppPlayed"` // tiles in the opponent's last play, -1 exchange/pass
	OppLeave  string  `json:"oppLeave"`
	A         judged  `json:"a"`
	B         judged  `json:"b"`
	ResA      float64 `json:"resA"` // p1's game result in run A
	ResB      float64 `json:"resB"`
	SpreadA   float64 `json:"spreadA"`
	SpreadB   float64 `json:"spreadB"`
	Iters     int     `json:"iters"`
	Inferred  int     `json:"inferred,omitempty"` // leaves in the -inferjudge read (0: none, judged on random racks)
	Plies     int     `json:"plies"`
	Err       string  `json:"err,omitempty"`
}

func moveFromRow(g *turnplayer.BaseTurnPlayer, pidx int, play string) (*move.Move, error) {
	switch {
	case strings.HasPrefix(play, "(exch"):
		cmd := strings.Split(play, " ")
		return g.NewExchangeMove(pidx, strings.TrimSuffix(cmd[1], ")"))
	case play == "(Pass)":
		return g.NewPassMove(pidx)
	default:
		p := strings.Split(strings.TrimSpace(play), " ")
		if len(p) < 2 {
			return nil, fmt.Errorf("cannot parse play %q", play)
		}
		return g.NewPlacementMove(pidx, p[0], p[1], false)
	}
}

func judge(cfg *config.Config, rules *game.GameRules, ha, hb *half, i int, iters int) record {
	ra, rb := ha.rows[i], hb.rows[i]
	rec := record{GameID: ha.gid, First: ha.first, Turn: i + 1, Mover: ra[0], Rack: ra[3]}
	rec.A = judged{Move: strings.TrimSpace(ra[4]), Leave: ra[8]}
	rec.B = judged{Move: strings.TrimSpace(rb[4]), Leave: rb[8]}
	rec.A.Score, _ = strconv.Atoi(ra[5])
	rec.B.Score, _ = strconv.Atoi(rb[5])
	rec.A.Tiles, _ = strconv.Atoi(ra[7])
	rec.B.Tiles, _ = strconv.Atoi(rb[7])
	rec.A.Eq, _ = strconv.ParseFloat(ra[9], 64)
	rec.B.Eq, _ = strconv.ParseFloat(rb[9], 64)
	rec.OppPlayed = -1

	players := []*pb.PlayerInfo{
		{Nickname: ha.rows[0][0], RealName: ha.rows[0][0]},
		{Nickname: ha.rows[1][0], RealName: ha.rows[1][0]},
	}
	g, err := turnplayer.BaseTurnPlayerFromRules(&turnplayer.GameOptions{
		BoardLayoutName: board.CrosswordGameLayout, Variant: game.VarClassic}, players, rules)
	if err != nil {
		rec.Err = err.Error()
		return rec
	}
	g.StartGame()
	for j := 0; j < i; j++ {
		if err := automatic.PlayLoggedRow(g, ha.rows[j], true); err != nil {
			rec.Err = fmt.Sprintf("replay row %d: %v", j+1, err)
			return rec
		}
	}
	me := 0
	if g.History().Players[1].Nickname == ra[0] {
		me = 1
	}
	// The opponent's last row gives their leave; their next row (identical
	// in both runs, as nothing they hold depends on this move) their rack.
	var oppLeave, oppRack string
	for j := i - 1; j >= 0; j-- {
		if ha.rows[j][0] != ra[0] {
			oppLeave = ha.rows[j][8]
			rec.OppPlay = strings.TrimSpace(ha.rows[j][4])
			rec.OppScore, _ = strconv.Atoi(ha.rows[j][5])
			if !strings.HasPrefix(ha.rows[j][4], "(") {
				rec.OppPlayed, _ = strconv.Atoi(ha.rows[j][7])
			}
			break
		}
	}
	for j := i + 1; j < len(hb.rows); j++ {
		if hb.rows[j][0] != ra[0] {
			oppRack = hb.rows[j][3]
			break
		}
	}
	rec.OppLeave = oppLeave
	racks := make([]*tilemapping.Rack, 2)
	racks[me] = tilemapping.RackFromString(ra[3], g.Alphabet())
	if oppRack != "" {
		racks[1-me] = tilemapping.RackFromString(oppRack, g.Alphabet())
		if err := g.Game.SetRacksForBoth(racks); err != nil {
			rec.Err = "seating racks: " + err.Error()
			return rec
		}
	} else if err := g.SetRackFor(me, racks[me]); err != nil {
		rec.Err = "seating rack: " + err.Error()
		return rec
	}
	rec.Bag = g.Bag().TilesRemaining()
	rec.Spread = g.SpreadFor(me)

	ma, err := moveFromRow(g, me, ra[4])
	if err != nil {
		rec.Err = "move a: " + err.Error()
		return rec
	}
	mb, err := moveFromRow(g, me, rb[4])
	if err != nil {
		rec.Err = "move b: " + err.Error()
		return rec
	}
	calc, err := equity.NewCombinedStaticCalculator(g.LexiconName(), cfg, "", equity.PEGAdjustmentFilename)
	if err != nil {
		rec.Err = err.Error()
		return rec
	}
	if pegMaxBag > 0 {
		if rec.Bag > pegMaxBag {
			rec.Err = "bag too big for -pegjudge"
			return rec
		}
		known := tilemapping.RackFromString(oppLeave, g.Alphabet()).TilesOn()
		for _, k := range [][]tilemapping.MachineLetter{nil, known} {
			pa, pb, err := pegSolve(g.Game, ma, mb, k)
			if err != nil {
				rec.Err = "peg: " + err.Error()
				return rec
			}
			if k == nil {
				rec.A.PegU, rec.B.PegU = pa, pb
			} else {
				rec.A.PegK, rec.B.PegK = pa, pb
			}
		}
		return rec
	}
	plies := 5
	if unseen := g.Bag().TilesRemaining() + int(g.RackFor(1-me).NumTiles()); unseen <= 14 {
		plies = unseen
	}
	rec.Plies = plies
	run := func(known []tilemapping.MachineLetter, inf *rangefinder.Inference) (map[string][2]float64, int, error) {
		s := &montecarlo.Simmer{}
		s.Init(g.Game.CopyWithHistory(), []equity.EquityCalculator{calc}, calc, cfg)
		s.SetThreads(1)
		if err := s.PrepareSim(plies, []*move.Move{ma, mb}); err != nil {
			return nil, 0, err
		}
		s.SetStoppingCondition(montecarlo.StopNone)
		if known != nil {
			s.SetKnownOppRack(known)
		}
		if inf != nil {
			// As the inference bot uses its read (BestPlay in ai/bot/elite.go).
			if inf.Complete {
				s.SetInferences(inf.InferredRacks, inf.RackLength, montecarlo.InferenceCompletePosterior)
			} else if len(inf.InferredRacks) > bot.InferencesSimLimit {
				s.SetInferences(inf.InferredRacks, inf.RackLength, montecarlo.InferenceWeightedRandomRacks)
			}
		}
		s.SimSingleThread(iters, plies)
		w := map[string][2]float64{}
		for _, sp := range s.PlaysByWinProb().PlaysNoLock() {
			w[sp.Move().ShortDescription()] = [2]float64{sp.WinProb(), sp.WinProbStdErr()}
		}
		return w, s.Iterations(), nil
	}
	known := tilemapping.RackFromString(oppLeave, g.Alphabet()).TilesOn()
	if known == nil {
		known = []tilemapping.MachineLetter{}
	}
	w, n, err := run(known, nil)
	if err != nil {
		rec.Err = "judge: " + err.Error()
		return rec
	}
	wr, _, err := run(nil, nil)
	if err != nil {
		rec.Err = "judge random: " + err.Error()
		return rec
	}
	if inferJudgeBudget > 0 {
		inf := inferRead(cfg, g.Game, calc, racks[me].TilesOn())
		if inf != nil && (inf.Complete || len(inf.InferredRacks) > bot.InferencesSimLimit) {
			rec.Inferred = len(inf.InferredRacks)
		}
		wi, _, err := run(nil, inf)
		if err != nil {
			rec.Err = "judge inferred: " + err.Error()
			return rec
		}
		sa, sb := ma.ShortDescription(), mb.ShortDescription()
		rec.A.WinI, rec.A.SEI = wi[sa][0], wi[sa][1]
		rec.B.WinI, rec.B.SEI = wi[sb][0], wi[sb][1]
	}
	rec.Iters = n
	sa, sb := ma.ShortDescription(), mb.ShortDescription()
	rec.A.Win, rec.A.SE = w[sa][0], w[sa][1]
	rec.B.Win, rec.B.SE = w[sb][0], w[sb][1]
	rec.A.WinR, rec.A.SER = wr[sa][0], wr[sa][1]
	rec.B.WinR, rec.B.SER = wr[sb][0], wr[sb][1]
	return rec
}

// pegMaxBag, when positive, judges positions with at most this many tiles in
// the bag with the pre-endgame solver instead of sims: an exact search to a
// 3-ply endgame, against every bag the opponent's rack leaves possible.
var pegMaxBag int

// pegTTFraction is the share of memory each solver's transposition table takes.
var pegTTFraction float64

// pegSolve returns the solver's win shares of a and b (nil for a play it
// skips, such as one that leaves more than one tile in the bag).
func pegSolve(g *game.Game, a, b *move.Move, known []tilemapping.MachineLetter) (*float64, *float64, error) {
	s := &preendgame.Solver{}
	if err := s.Init(g.CopyWithHistory()); err != nil {
		return nil, nil, err
	}
	s.SetThreads(1)
	s.SetEndgamePlies(3)
	// One table per solver, many solvers at once: keep each small.
	s.SetTTableFraction(pegTTFraction)
	s.SetEarlyCutoffOptim(false)
	s.SetSkipLossOptim(false)
	s.SetIterativeDeepening(false)
	s.SetSolveOnly([]*move.Move{a, b})
	if known != nil {
		s.SetKnownOppRack(known)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	plays, err := s.Solve(ctx)
	if err != nil {
		return nil, nil, err
	}
	share := func(m *move.Move) *float64 {
		for _, p := range plays {
			if p.Play.Equals(m, false, false) && !p.Ignore && p.Points+p.FoundLosses > 0 {
				v := float64(p.Points / (p.Points + p.FoundLosses))
				return &v
			}
		}
		return nil
	}
	return share(a), share(b), nil
}

// inferJudgeBudget, when positive, adds a third verdict: the judge sim draws
// the opponent's rack from the mover's own read of them, made as the baseline
// inference bot makes it (default tau curve, this leaf budget).
var inferJudgeBudget int

// inferRead reads the opponent from the mover's side of g, or returns nil
// when there is nothing to read (the opponent's first move, an exchange,
// a bingo).
func inferRead(cfg *config.Config, g *game.Game, calc equity.EquityCalculator, myRack []tilemapping.MachineLetter) *rangefinder.Inference {
	rf := &rangefinder.RangeFinder{}
	rf.Init(g, []equity.EquityCalculator{calc}, cfg)
	rf.SetThreads(1)
	rf.SetTauCap(0)
	rf.SetMiniSim(0, 0)
	rf.SetBudget(inferJudgeBudget)
	if err := rf.PrepareFinder(myRack); err != nil {
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := rf.Infer(ctx); err != nil {
		log.Debug().Err(err).Msg("infer-judge")
	}
	return rf.Inferences()
}

func main() {
	a := flag.String("a", "", "per-turn log of run A (the variant)")
	b := flag.String("b", "", "per-turn log of run B (the baseline, same seed)")
	ga := flag.String("ga", "", "games file of run A")
	gb := flag.String("gb", "", "games file of run B")
	out := flag.String("out", "divjudge.jsonl", "output")
	iters := flag.Int("iters", 5000, "judge iterations per sim")
	flag.Float64Var(&pegTTFraction, "pegtt", 0.003, "memory fraction of each -pegjudge solver's transposition table")
	flag.IntVar(&pegMaxBag, "pegjudge", 0, "judge positions with at most this many tiles in the bag with the pre-endgame solver instead (0 = off)")
	flag.IntVar(&inferJudgeBudget, "inferjudge", 0, "also judge with the opponent's rack drawn from the mover's inference, at this leaf budget (0 = off; the bots use 200)")
	parallel := flag.Int("parallel", 8, "positions at once (each single-threaded)")
	limit := flag.Int("limit", 0, "judge at most this many divergences (0 = all)")
	mode := flag.String("mode", "div", "div (first divergence of two runs) or screen (screendecide output)")
	decideF := flag.String("decide", "", "screen: the variant's screendecide output")
	controlF := flag.String("control", "", "screen: the control's screendecide output (default: the logged moves)")
	flag.Parse()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	if *mode == "screen" {
		screen(*b, *decideF, *controlF, *out, *iters, *parallel, *limit)
		return
	}

	ta, err := readTurns(*a)
	if err != nil {
		log.Fatal().Err(err).Msg("a")
	}
	tb, err := readTurns(*b)
	if err != nil {
		log.Fatal().Err(err).Msg("b")
	}
	resA, err := readResults(*ga)
	if err != nil {
		log.Fatal().Err(err).Msg("ga")
	}
	resB, err := readResults(*gb)
	if err != nil {
		log.Fatal().Err(err).Msg("gb")
	}
	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, "NWL23")
	cfg.Set(config.ConfigDefaultLetterDistribution, "english")
	rules, err := game.NewBasicGameRules(cfg, "NWL23", board.CrosswordGameLayout,
		"english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		log.Fatal().Err(err).Msg("rules")
	}

	type job struct {
		ha, hb *half
		i      int
		key    string
	}
	var jobs []job
	common, same := 0, 0
	for key, ha := range ta {
		hb, ok := tb[key]
		if !ok {
			continue
		}
		common++
		n := min(len(ha.rows), len(hb.rows))
		div := -1
		for i := 0; i < n; i++ {
			if ha.rows[i][4] != hb.rows[i][4] || ha.rows[i][3] != hb.rows[i][3] {
				div = i
				break
			}
		}
		if div < 0 {
			if len(ha.rows) == len(hb.rows) {
				same++
			}
			continue
		}
		if ha.rows[div][3] != hb.rows[div][3] {
			// Different racks with the same history: the runs are not paired.
			fmt.Fprintf(os.Stderr, "%s turn %d: racks differ (%s vs %s), skipped\n",
				key, div+1, ha.rows[div][3], hb.rows[div][3])
			continue
		}
		jobs = append(jobs, job{ha, hb, div, key})
	}
	fmt.Fprintf(os.Stderr, "%d seatings in both runs, %d identical, %d diverge\n", common, same, len(jobs))
	if *limit > 0 && len(jobs) > *limit {
		jobs = jobs[:*limit]
	}

	f, err := os.Create(*out)
	if err != nil {
		log.Fatal().Err(err).Msg("out")
	}
	defer f.Close()
	var mu sync.Mutex
	enc := json.NewEncoder(f)
	ch := make(chan job)
	var wg sync.WaitGroup
	var done atomic.Int64
	for w := 0; w < *parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range ch {
				rec := judge(cfg, rules, j.ha, j.hb, j.i, *iters)
				if r, ok := resA[j.key]; ok {
					rec.ResA, rec.SpreadA = r[0], r[1]
				}
				if r, ok := resB[j.key]; ok {
					rec.ResB, rec.SpreadB = r[0], r[1]
				}
				mu.Lock()
				enc.Encode(rec)
				mu.Unlock()
				if d := done.Add(1); d%50 == 0 {
					fmt.Fprintf(os.Stderr, "%d/%d\n", d, len(jobs))
				}
			}
		}()
	}
	for _, j := range jobs {
		ch <- j
	}
	close(ch)
	wg.Wait()
}
