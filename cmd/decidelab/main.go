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
//
// and then judges the distinct choices with a long sim that knows the
// opponent's true leave. The judge's win % difference between infer's choice
// and none's is what inference gained in that position; the same for none2
// against none is the noise floor.
//
// Usage:
//
//	decidelab -turnlog ~/Downloads/infer-v-sim-5000.txt -positions 400 -parallel 10
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
)

type choice struct {
	Move      string  `json:"move"`
	Equity    float64 `json:"equity"`
	ElapsedMS int64   `json:"elapsedMs"`
	JudgeWin  float64 `json:"judgeWin"`
	JudgeSE   float64 `json:"judgeSE"`
}

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
	Choices   map[string]*choice `json:"choices"`
	JudgeIter int                `json:"judgeIters"`
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
	minBag := flag.Int("minbag", 2, "only positions the log has at least this many tiles in the bag for")
	flag.Parse()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	if *turnlog == "" {
		flag.Usage()
		os.Exit(2)
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
				rec := decide(pos, cfg, *judgeIters)
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

func botConfig(cfg *config.Config) *bot.BotConfig {
	return &bot.BotConfig{Config: *cfg, MinSimPlies: 5, SimThreads: 1, InferenceBudget: 200}
}

func decide(pos *automatic.CorpusPosition, cfg *config.Config, judgeIters int) record {
	g := pos.Game
	rec := record{GameID: pos.GameID, Half: pos.Half, Turn: pos.Turn, OppPlay: pos.OppPlay,
		TrueLeave: pos.TrueLeave, MyRack: pos.MyRack, Bag: g.Bag().TilesRemaining(), Kept: pos.LeaveLen,
		Exchange: len(pos.OppPlay) > 0 && pos.OppPlay[0] == '(',
		Spread:   g.SpreadFor(g.PlayerOnTurn()), Choices: map[string]*choice{},
		OppTiles: int(g.RackFor(g.NextPlayer()).NumTiles())}

	arms := []struct {
		name string
		code pb.BotRequest_BotCode
	}{
		{"none", pb.BotRequest_SIMMING_BOT_NO_EG},
		{"none2", pb.BotRequest_SIMMING_BOT_NO_EG},
		{"infer", pb.BotRequest_SIMMING_INFER_BOT_NO_EG},
	}
	moves := map[string]*move.Move{}
	for _, a := range arms {
		gc := g.CopyWithHistory()
		p, err := bot.NewBotTurnPlayerFromGame(gc, botConfig(cfg), a.code)
		if err != nil {
			rec.Err = err.Error()
			return rec
		}
		t0 := time.Now()
		m, err := p.BestPlay(context.Background())
		if err != nil {
			rec.Err = a.name + ": " + err.Error()
			return rec
		}
		s := m.ShortDescription()
		rec.Choices[a.name] = &choice{Move: s, Equity: m.Equity(), ElapsedMS: time.Since(t0).Milliseconds()}
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
		leave := tilemapping.RackFromString(pos.TrueLeave, g.Alphabet()).TilesOn()
		s.SetKnownOppRack(leave)
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
	return rec
}
