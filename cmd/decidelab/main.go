// decidelab asks whether inference changes what the bot plays, and whether
// the changes win games.
//
// Better reads (lift bits) only turn into wins when they move the sim's choice,
// and only by however much the new choice is really better. For each position
// from a finished run's turn log -- the inferring bot to move, right after the
// opponent's play -- it lets three bots choose a move:
//
//	none   SIMMING_BOT_NO_EG, as the match's opponent
//	none2  the same again: how often the choice changes from sim noise alone
//	infer  SIMMING_INFER_BOT_NO_EG on the engine's current defaults
//	oracle SIMMING_BOT_NO_EG simming against the opponent's true leave: the
//	       ceiling, what a perfect read would choose
//
// -arms picks which of these run, and -variants adds inferring bots with other
// settings, e.g. -variants "sharp:scale=0.5;b1000:budget=1000;t05:tau=0.05".
//
// and then judges the distinct choices with a long sim that knows the
// opponent's true leave. The judge's win % difference between infer's choice
// and none's is what inference gained in that position; the same for none2
// against none is the noise floor.
//
// Usage:
//
//	decidelab -turnlog ~/Downloads/infer-v-sim-5000.txt -positions 400 -parallel 10
//
// Each choice also records its static rank: its place in the static move list,
// 1 being the top play, to show whether the sim's 100 candidates are enough.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/rangefinder"
)

type choice struct {
	Move      string  `json:"move"`
	Equity    float64 `json:"equity"`
	ElapsedMS int64   `json:"elapsedMs"`
	JudgeWin  float64 `json:"judgeWin"`
	JudgeSE   float64 `json:"judgeSE"`
	// StaticRank is the move's 1-based place in the static ranking; 0 = beyond
	// rankDepth.
	StaticRank int `json:"staticRank"`
	// InfoBits is an inferring chooser's read strength (RangeFinder.InfoBits);
	// absent when it did not infer or the posterior was incomplete.
	InfoBits *float64 `json:"infoBits,omitempty"`
	// Read grades an inferring chooser's posterior against the true leave.
	Read *readScore `json:"read,omitempty"`
}

// readScore is a read graded against the true leave: the exact-leave figures
// (rank, lift) and the partial-credit ones (rangefinder.Closeness).
type readScore struct {
	Rank      int      `json:"rank"` // 0 = ruled out
	Leaves    int      `json:"leaves"`
	LiftBits  *float64 `json:"liftBits,omitempty"`
	Post      float64  `json:"post"`
	Prior     float64  `json:"prior"`
	Complete  bool     `json:"complete"`
	Overlap   float64  `json:"overlap"`
	OverlapPr float64  `json:"overlapPrior"`
	VDist     float64  `json:"valueDist"`
	VDistPr   float64  `json:"valueDistPrior"`
	SoftOver  float64  `json:"softLiftOverlap"`
	SoftValue float64  `json:"softLiftValue"`
	TrueValue float64  `json:"trueValue"`
	MeanValue float64  `json:"meanValue"`
	MeanValPr float64  `json:"meanValuePrior"`
}

// agreement asks whether the leaves a read believes in would make the bot
// play well against the leave the opponent really kept. Leaves are sampled
// from the posterior and from the prior; each is simmed as the opponent's
// known rack over the same candidates, and the move it picks is priced by a
// long sim against the true leave. A leave's regret is the win % its pick
// gives up there against the best candidate: zero for leaves that lead to the
// right play, whatever their tiles. The truth itself is decided the same way,
// so its regret is the sim-noise floor. The pricing sim, the truth's own
// decision and the prior samples are shared by every arm checked.
type agreement struct {
	TruthMove   string               `json:"truthMove"`   // the true-leave pricing sim's best candidate
	TruthRegret float64              `json:"truthRegret"` // regret of a short sim that knows the truth
	Prior       float64              `json:"prior"`       // share of prior samples picking truthMove
	PriorRegret float64              `json:"priorRegret"` // their mean regret, win pts
	Cands       int                  `json:"cands"`
	PriorLeaves []agreeLeaf          `json:"priorLeaves"`
	Arms        map[string]*armAgree `json:"arms"`
}

// armAgree is one inferring arm's posterior samples.
type armAgree struct {
	Post       float64     `json:"post"`       // share picking truthMove
	PostRegret float64     `json:"postRegret"` // mean regret, win pts
	Leaves     []agreeLeaf `json:"leaves"`
}

type agreeLeaf struct {
	Leave  string  `json:"leave"`
	Move   string  `json:"move"`
	Regret float64 `json:"regret"`
}

// agreeOpts sets up the decision-agreement sims.
type agreeOpts struct {
	arms    map[string]bool // nil = every inferring arm
	samples int
	plays   int
	iters   int
	plies   int
	judge   int
	// lq, when set, adds leave-quality reads to the agreement: every leave
	// weighted by prior x exp(scale x beta x leave value), beta fitted per
	// tiles played and score of the opponent's play (lqN), and our own
	// posterior times the same factor (lqxN), one pair per scale.
	lq       lqTable
	lqScales []float64
}

// lqTable is the leave-quality weight, per tiles played: [score_lo,
// score_hi, beta] bins in score order.
type lqTable map[string][][3]float64

func (t lqTable) beta(played, score int) (float64, bool) {
	bins := t[strconv.Itoa(played)]
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

// rankDepth is how deep in the static ranking a choice is looked for.
const rankDepth = 1000

type record struct {
	GameID    string             `json:"gameID"`
	Half      int                `json:"half"`
	Turn      int                `json:"turn"`
	OppPlay   string             `json:"oppPlay"`
	TrueLeave string             `json:"trueLeave"`
	MyRack    string             `json:"myRack"`
	Bag       int                `json:"bag"`
	Kept      int                `json:"kept"`
	Exchange  bool               `json:"exchange"`
	Spread    int                `json:"spread"`
	OppTiles  int                `json:"oppTiles"`
	OppScore  int                `json:"oppScore"`
	OppPlayed int                `json:"oppPlayed"` // tiles the opponent placed or threw
	Choices   map[string]*choice `json:"choices"`
	JudgeIter int                `json:"judgeIters"`
	Agree     *agreement         `json:"agree,omitempty"`
	Err       string             `json:"err,omitempty"`
}

func main() {
	turnlog := flag.String("turnlog", "", "per-turn log from a finished run (required)")
	lexicon := flag.String("lexicon", "NWL23", "lexicon")
	out := flag.String("out", "decidelab.jsonl", "output")
	pool := flag.Int("pool", 6000, "positions to load from the log before sampling")
	npos := flag.Int("positions", 400, "positions to sample from the pool")
	seed := flag.Int64("seed", 1, "sampling seed")
	parallel := flag.Int("parallel", 10, "positions at once (each single-threaded)")
	judgeIters := flag.Int("judge-iters", 3000, "judge sim iterations")
	maxBag := flag.Int("maxbag", 0, "only positions with at most this many in the bag (0 = no limit)")
	maxPlayed := flag.Int("maxplayed", 0, "only placements of at most this many tiles by the opponent (0 = no limit; drops exchanges)")
	minBag := flag.Int("minbag", 2, "only positions the log has at least this many tiles in the bag for")
	plies := flag.Int("plies", 5, "sim plies of the deciding bots (the judge always sims 5)")
	budget := flag.Int("budget", 200, "inference leaf budget")
	armList := flag.String("arms", "none,none2,infer,oracle", "built-in choosers to run (none always runs)")
	variantList := flag.String("variants", "", "extra inferring choosers, name:key=value,...;... "+
		"(keys tau, scale, cap, budget, minbits, iters, plays, mplies: mini-sim iterations, plays and plies)")
	agreeN := flag.Int("agree", 0, "decision agreement: leaves sampled from each of the posterior and the prior (0 = off)")
	agreeArm := flag.String("agree-arm", "all", "inferring choosers whose reads are checked for decision agreement, comma-separated, or all")
	agreePlays := flag.Int("agree-plays", 20, "decision agreement: top static plays simmed (plus every chooser's pick)")
	agreeIters := flag.Int("agree-iters", 300, "decision agreement: iterations per sampled-leave sim")
	agreeJudge := flag.Int("agree-judge", 2000, "decision agreement: iterations of the true-leave pricing sim")
	lqFile := flag.String("lq", "", "decision agreement: leave-quality weight table (JSON) to add lq reads")
	lqScaleList := flag.String("lq-scales", "1", "decision agreement: multipliers of the leave-quality weight, comma-separated")
	flag.Parse()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	if *turnlog == "" {
		flag.Usage()
		os.Exit(2)
	}
	arms, err := parseArms(*armList, *variantList, *plies, *budget)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	var agreeArms map[string]bool
	if *agreeArm != "all" {
		agreeArms = map[string]bool{}
		for _, n := range strings.Split(*agreeArm, ",") {
			agreeArms[strings.TrimSpace(n)] = true
		}
	}
	var lq lqTable
	var lqScales []float64
	if *lqFile != "" {
		b, err := os.ReadFile(*lqFile)
		if err == nil {
			err = json.Unmarshal(b, &lq)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, "lq:", err)
			os.Exit(2)
		}
		for _, x := range strings.Split(*lqScaleList, ",") {
			v, err := strconv.ParseFloat(strings.TrimSpace(x), 64)
			if err != nil {
				fmt.Fprintln(os.Stderr, "lq-scales:", err)
				os.Exit(2)
			}
			lqScales = append(lqScales, v)
		}
	}
	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, *lexicon)
	cfg.Set(config.ConfigDefaultLetterDistribution, "english")

	corpus, skipped, err := automatic.LoadCorpusVerbose(cfg, *turnlog, *lexicon, "english", "",
		automatic.CorpusFilter{AllLifts: true, MinBag: *minBag, MaxBag: *maxBag, Limit: *pool})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	// The log's bag count is from before the opponent drew; what the bot faces
	// is the replayed bag. With fewer than 2 in it the bot does not sim at all.
	kept := corpus[:0]
	for _, p := range corpus {
		if *maxPlayed > 0 && (len(p.OppPlay) > 0 && p.OppPlay[0] == '(' ||
			len([]rune(p.OppRack))-p.LeaveLen > *maxPlayed) {
			continue
		}
		if p.Game.Bag().TilesRemaining() >= 2 {
			kept = append(kept, p)
		}
	}
	corpus = kept
	rng := rand.New(rand.NewSource(*seed))
	rng.Shuffle(len(corpus), func(i, j int) { corpus[i], corpus[j] = corpus[j], corpus[i] })
	if len(corpus) > *npos {
		corpus = corpus[:*npos]
	}
	fmt.Fprintf(os.Stderr, "%d positions (%d games skipped)\n", len(corpus), skipped)

	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	var mu sync.Mutex
	var done atomic.Int64
	start := time.Now()
	work := make(chan *automatic.CorpusPosition)
	var wg sync.WaitGroup
	for w := 0; w < *parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for pos := range work {
				rec := decide(pos, cfg, *judgeIters, arms,
					agreeOpts{arms: agreeArms, samples: *agreeN, plays: *agreePlays, iters: *agreeIters, plies: *plies, judge: *agreeJudge,
						lq: lq, lqScales: lqScales})
				mu.Lock()
				enc.Encode(rec)
				mu.Unlock()
				n := done.Add(1)
				fmt.Fprintf(os.Stderr, "%d/%d  %.1f min\n", n, len(corpus), time.Since(start).Minutes())
			}
		}()
	}
	for _, pos := range corpus {
		work <- pos
	}
	close(work)
	wg.Wait()
}

// arm is one chooser: a bot and the settings it decides with.
type arm struct {
	name    string
	code    pb.BotRequest_BotCode
	plies   int
	budget  int
	tau     float64
	scale   float64
	minBits float64
	tauCap  float64
	oracle  bool
	iters   int // mini-sim iterations
	mplays  int // mini-sim static plays
	mplies  int // mini-sim plies
}

func parseArms(list, variants string, plies, budget int) ([]arm, error) {
	builtin := map[string]arm{
		"none":   {code: pb.BotRequest_SIMMING_BOT_NO_EG},
		"none2":  {code: pb.BotRequest_SIMMING_BOT_NO_EG},
		"infer":  {code: pb.BotRequest_SIMMING_INFER_BOT_NO_EG},
		"oracle": {code: pb.BotRequest_SIMMING_BOT_NO_EG, oracle: true},
	}
	arms := []arm{{name: "none", code: pb.BotRequest_SIMMING_BOT_NO_EG, plies: plies, budget: budget}}
	for _, n := range strings.Split(list, ",") {
		n = strings.TrimSpace(n)
		if n == "" || n == "none" {
			continue
		}
		a, ok := builtin[n]
		if !ok {
			return nil, fmt.Errorf("unknown arm %q", n)
		}
		a.name, a.plies, a.budget = n, plies, budget
		arms = append(arms, a)
	}
	for _, v := range strings.Split(variants, ";") {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		name, settings, _ := strings.Cut(v, ":")
		a := arm{name: name, code: pb.BotRequest_SIMMING_INFER_BOT_NO_EG, plies: plies, budget: budget}
		for _, kv := range strings.Split(settings, ",") {
			k, val, ok := strings.Cut(kv, "=")
			if !ok {
				return nil, fmt.Errorf("variant %q: want key=value, got %q", name, kv)
			}
			x, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("variant %q: %v", name, err)
			}
			switch k {
			case "tau":
				a.tau = x
			case "scale":
				a.scale = x
			case "budget":
				a.budget = int(x)
			case "minbits":
				a.minBits = x
			case "cap":
				a.tauCap = x
			case "iters":
				a.iters = int(x)
			case "plays":
				a.mplays = int(x)
			case "mplies":
				a.mplies = int(x)
			default:
				return nil, fmt.Errorf("variant %q: unknown key %q", name, k)
			}
		}
		arms = append(arms, a)
	}
	return arms, nil
}

func botConfig(cfg *config.Config, a arm) *bot.BotConfig {
	return &bot.BotConfig{Config: *cfg, MinSimPlies: a.plies, SimThreads: 1, InferenceBudget: a.budget,
		InferenceTau: a.tau, InferenceTauScale: a.scale, InferenceMinInfoBits: a.minBits, InferenceTauCap: a.tauCap,
		OracleInference: a.oracle, InferenceSimIters: a.iters, InferenceMiniSimPlays: a.mplays,
		InferenceMiniSimPlies: a.mplies}
}

func decide(pos *automatic.CorpusPosition, cfg *config.Config, judgeIters int, arms []arm, ao agreeOpts) record {
	g := pos.Game
	rec := record{GameID: pos.GameID, Half: pos.Half, Turn: pos.Turn, OppPlay: pos.OppPlay,
		TrueLeave: pos.TrueLeave, MyRack: pos.MyRack, Bag: g.Bag().TilesRemaining(), Kept: pos.LeaveLen,
		Exchange: len(pos.OppPlay) > 0 && pos.OppPlay[0] == '(',
		Spread:   g.SpreadFor(g.PlayerOnTurn()), Choices: map[string]*choice{},
		OppTiles: int(g.RackFor(g.NextPlayer()).NumTiles()),
		OppScore: pos.OppScore, OppPlayed: len([]rune(pos.OppRack)) - pos.LeaveLen}

	moves := map[string]*move.Move{}
	rank := map[string]int{}
	var static []*move.Move
	var agreeNames []string
	agreeRF := map[string]*rangefinder.RangeFinder{}
	trueLeave := tilemapping.RackFromString(pos.TrueLeave, g.Alphabet()).TilesOn()
	for _, a := range arms {
		gc := g.CopyWithHistory()
		p, err := bot.NewBotTurnPlayerFromGame(gc, botConfig(cfg, a), a.code)
		if err != nil {
			rec.Err = err.Error()
			return rec
		}
		if len(rank) == 0 {
			static = p.GenerateMoves(rankDepth)
			for i, m := range static {
				if _, ok := rank[m.ShortDescription()]; !ok {
					rank[m.ShortDescription()] = i + 1
				}
			}
		}
		t0 := time.Now()
		m, err := p.BestPlay(context.Background())
		if err != nil {
			rec.Err = a.name + ": " + err.Error()
			return rec
		}
		s := m.ShortDescription()
		rec.Choices[a.name] = &choice{Move: s, Equity: m.Equity(), ElapsedMS: time.Since(t0).Milliseconds(),
			StaticRank: rank[s]}
		if ib := p.LastInferenceInfoBits(); a.code == pb.BotRequest_SIMMING_INFER_BOT_NO_EG && !math.IsNaN(ib) {
			rec.Choices[a.name].InfoBits = &ib
		}
		if rf := p.Inferencer(); rf != nil && a.code == pb.BotRequest_SIMMING_INFER_BOT_NO_EG {
			rec.Choices[a.name].Read = gradeRead(rf, trueLeave)
			if ao.arms == nil || ao.arms[a.name] {
				agreeNames = append(agreeNames, a.name)
				agreeRF[a.name] = rf
			}
		}
		moves[s] = m
	}

	// The judge: every distinct choice simmed against the opponent's real leave.
	if len(moves) > 1 {
		calc, err := equity.NewCombinedStaticCalculator(g.LexiconName(), cfg, "", equity.PEGAdjustmentFilename)
		if err != nil {
			rec.Err = err.Error()
			return rec
		}
		gc := g.CopyWithHistory()
		plays := make([]*move.Move, 0, len(moves))
		for _, m := range moves {
			plays = append(plays, m)
		}
		s := &montecarlo.Simmer{}
		s.Init(gc, []equity.EquityCalculator{calc}, calc, cfg)
		s.SetThreads(1)
		plies := 5
		if unseen := g.Bag().TilesRemaining() + int(g.RackFor(g.NextPlayer()).NumTiles()); unseen <= 14 {
			plies = unseen
		}
		if err := s.PrepareSim(plies, plays); err != nil {
			rec.Err = err.Error()
			return rec
		}
		s.SetStoppingCondition(montecarlo.StopNone)
		s.SetKnownOppRack(trueLeave)
		s.SimSingleThread(judgeIters, plies)
		rec.JudgeIter = s.Iterations()
		win := map[string][2]float64{}
		for _, sp := range s.PlaysByWinProb().PlaysNoLock() {
			win[sp.Move().ShortDescription()] = [2]float64{sp.WinProb(), sp.WinProbStdErr()}
		}
		for _, c := range rec.Choices {
			c.JudgeWin, c.JudgeSE = win[c.Move][0], win[c.Move][1]
		}
	}
	if ao.samples > 0 && len(agreeNames) > 0 {
		cands := static[:min(ao.plays, len(static))]
		for s, m := range moves {
			if _, ok := rank[s]; !ok || rank[s] > ao.plays {
				cands = append(cands, m)
			}
		}
		ag, err := agree(g, cfg, agreeNames, agreeRF, trueLeave, cands, ao, pos)
		if err != nil {
			rec.Err = "agree: " + err.Error()
			return rec
		}
		rec.Agree = ag // nil when no read was complete (an exchange)
	}
	return rec
}

func gradeRead(rf *rangefinder.RangeFinder, trueLeave []tilemapping.MachineLetter) *readScore {
	sc := rf.ScoreLeave(trueLeave)
	r := &readScore{Rank: sc.Rank, Leaves: sc.Leaves, Post: sc.Posterior, Prior: sc.Prior}
	if !math.IsNaN(sc.LiftBits) {
		lb := sc.LiftBits
		r.LiftBits = &lb
	}
	if c, ok := rf.ReadCloseness(trueLeave); ok {
		r.Complete = true
		r.Overlap, r.OverlapPr = c.Overlap, c.OverlapPrior
		r.VDist, r.VDistPr = c.ValueDist, c.ValueDistPrior
		r.SoftOver, r.SoftValue = c.SoftLiftOverlap, c.SoftLiftValue
		r.TrueValue, r.MeanValue, r.MeanValPr = c.TrueValue, c.MeanValue, c.MeanValuePrior
	}
	return r
}

// agree runs the decision-agreement sims for one position's reads.
func agree(g *game.Game, cfg *config.Config, names []string, rfs map[string]*rangefinder.RangeFinder,
	trueLeave []tilemapping.MachineLetter, cands []*move.Move, ao agreeOpts,
	pos *automatic.CorpusPosition) (*agreement, error) {

	type read struct {
		name   string
		leaves [][]tilemapping.MachineLetter
		post   []float64
	}
	var reads []read
	for _, n := range names {
		leaves, post, complete := rfs[n].Posterior()
		if complete && len(leaves) > 0 {
			reads = append(reads, read{n, leaves, post})
		}
	}
	if len(reads) == 0 {
		return nil, nil
	}
	calc, err := equity.NewCombinedStaticCalculator(g.LexiconName(), cfg, "", equity.PEGAdjustmentFilename)
	if err != nil {
		return nil, err
	}
	exchange := len(pos.OppPlay) > 0 && pos.OppPlay[0] == '('
	if beta, ok := ao.lq.beta(len([]rune(pos.OppRack))-pos.LeaveLen, pos.OppScore); ok && !exchange && pos.LeaveLen > 0 {
		r0 := reads[0]
		rf := rfs[r0.name]
		all := enumerateLeaves(rf.BagMap(), pos.LeaveLen)
		value := func(ls [][]tilemapping.MachineLetter) ([]float64, float64) {
			vs := make([]float64, len(ls))
			top := math.Inf(-1)
			for i, l := range ls {
				c := slices.Clone(l)
				slices.Sort(c)
				vs[i] = calc.LeaveValue(tilemapping.MachineWord(c))
				top = max(top, vs[i])
			}
			return vs, top
		}
		allV, allTop := value(all)
		ourV, ourTop := value(r0.leaves)
		for _, sc := range ao.lqScales {
			b := sc * beta
			w := make([]float64, len(all))
			for i, l := range all {
				w[i] = rf.LeavePrior(l) * math.Exp(b*(allV[i]-allTop))
			}
			reads = append(reads, read{fmt.Sprintf("lq%g", sc), all, w})
			wx := make([]float64, len(r0.leaves))
			for i := range r0.leaves {
				wx[i] = r0.post[i] * math.Exp(b*(ourV[i]-ourTop))
			}
			reads = append(reads, read{fmt.Sprintf("lqx%g", sc), r0.leaves, wx})
		}
	}
	plies := ao.plies
	if unseen := g.Bag().TilesRemaining() + int(g.RackFor(g.NextPlayer()).NumTiles()); unseen <= 14 {
		plies = min(plies, unseen)
	}
	simAgainst := func(leave []tilemapping.MachineLetter, iters int) (*montecarlo.Simmer, error) {
		s := &montecarlo.Simmer{}
		s.Init(g.CopyWithHistory(), []equity.EquityCalculator{calc}, calc, cfg)
		s.SetThreads(1)
		if err := s.PrepareSim(plies, cands); err != nil {
			return nil, err
		}
		s.SetStoppingCondition(montecarlo.StopNone)
		s.SetKnownOppRack(leave)
		s.SimSingleThread(iters, plies)
		return s, nil
	}
	key := func(l []tilemapping.MachineLetter) string {
		c := slices.Clone(l)
		slices.Sort(c)
		return tilemapping.MachineWord(c).UserVisible(g.Alphabet())
	}
	// Each distinct leave is decided once, whichever arm or the prior drew it.
	cache := map[string]string{}
	decideFor := func(leave []tilemapping.MachineLetter) (string, error) {
		k := key(leave)
		if m, ok := cache[k]; ok {
			return m, nil
		}
		s, err := simAgainst(leave, ao.iters)
		if err != nil {
			return "", err
		}
		m := s.WinningPlay().Move().ShortDescription()
		cache[k] = m
		return m, nil
	}

	ag := &agreement{Cands: len(cands), Arms: map[string]*armAgree{}}
	pricing, err := simAgainst(trueLeave, ao.judge)
	if err != nil {
		return nil, err
	}
	win := map[string]float64{}
	best := -1.0
	for _, sp := range pricing.PlaysByWinProb().PlaysNoLock() {
		d := sp.Move().ShortDescription()
		win[d] = sp.WinProb()
		if sp.WinProb() > best {
			best, ag.TruthMove = sp.WinProb(), d
		}
	}
	regret := func(m string) float64 { return 100 * (best - win[m]) }
	short, err := decideFor(trueLeave)
	if err != nil {
		return nil, err
	}
	ag.TruthRegret = regret(short)

	// sample draws n leaves by weight, with a stream seeded by the position and
	// the source, so every arm meets the same random numbers.
	sample := func(src string, leaves [][]tilemapping.MachineLetter, w []float64) (float64, float64, []agreeLeaf, error) {
		h := sha256.Sum256([]byte(fmt.Sprintf("%s/%d/%d/%s", pos.GameID, pos.Half, pos.Turn, src)))
		rng := rand.New(rand.NewSource(int64(binary.LittleEndian.Uint64(h[:8]))))
		cdf := make([]float64, len(w))
		tot := 0.0
		for i, x := range w {
			tot += x
			cdf[i] = tot
		}
		same, reg := 0, 0.0
		var out []agreeLeaf
		for range ao.samples {
			i, _ := slices.BinarySearch(cdf, rng.Float64()*tot)
			i = min(i, len(leaves)-1)
			m, err := decideFor(leaves[i])
			if err != nil {
				return 0, 0, nil, err
			}
			if m == ag.TruthMove {
				same++
			}
			reg += regret(m)
			out = append(out, agreeLeaf{Leave: key(leaves[i]), Move: m, Regret: regret(m)})
		}
		n := float64(ao.samples)
		return float64(same) / n, reg / n, out, nil
	}

	r0 := reads[0]
	prior := make([]float64, len(r0.leaves))
	for i, l := range r0.leaves {
		prior[i] = rfs[r0.name].LeavePrior(l)
	}
	if ag.Prior, ag.PriorRegret, ag.PriorLeaves, err = sample("prior", r0.leaves, prior); err != nil {
		return nil, err
	}
	for _, r := range reads {
		a := &armAgree{}
		if a.Post, a.PostRegret, a.Leaves, err = sample("post", r.leaves, r.post); err != nil {
			return nil, err
		}
		ag.Arms[r.name] = a
	}
	return ag, nil
}

// enumerateLeaves lists every leave of k tiles the unseen pool can hold.
func enumerateLeaves(bag []uint8, k int) [][]tilemapping.MachineLetter {
	var out [][]tilemapping.MachineLetter
	buf := make([]tilemapping.MachineLetter, 0, k)
	var rec func(minIdx, remaining int)
	rec = func(minIdx, remaining int) {
		if remaining == 0 {
			out = append(out, slices.Clone(buf))
			return
		}
		for i := minIdx; i < len(bag); i++ {
			if bag[i] == 0 {
				continue
			}
			prev := len(buf)
			for c := 1; c <= int(bag[i]) && c <= remaining; c++ {
				buf = append(buf, tilemapping.MachineLetter(i))
				rec(i+1, remaining-c)
			}
			buf = buf[:prev]
		}
	}
	rec(0, k)
	return out
}
