package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/movegen"
)

func simMain(args []string) {
	fs := flag.NewFlagSet("sim", flag.ExitOnError)
	var inPath, outPath string
	var threads, plies, cands, shard, shards, limit int
	fs.StringVar(&inPath, "in", "positions.jsonl", "positions from simlabel select (.jsonl or .jsonl.gz)")
	fs.StringVar(&outPath, "out", "labels.jsonl", "labels; appended to, positions already there are skipped")
	fs.IntVar(&threads, "threads", 4, "positions simulated at once (one thread each)")
	fs.IntVar(&plies, "plies", 5, "sim depth")
	fs.IntVar(&cands, "cands", 50, "candidates: the top plays by static equity")
	fs.IntVar(&shard, "shard", 0, "this machine's shard, 0..shards-1")
	fs.IntVar(&shards, "shards", 1, "number of machines splitting the positions")
	fs.IntVar(&limit, "limit", 0, "stop after this many new positions (0 = all)")
	fs.Parse(args)
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	cfg := config.DefaultConfig()

	done := map[string]bool{}
	if f, err := os.Open(outPath); err == nil {
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<24)
		for sc.Scan() {
			var l struct {
				Key string `json:"key"`
			}
			if json.Unmarshal(sc.Bytes(), &l) == nil && l.Key != "" {
				done[l.Key] = true
			}
		}
		f.Close()
	}
	in, err := openMaybeGzip(inPath)
	if err != nil {
		log.Fatal().Err(err).Msg("positions")
	}
	var todo []Position
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	n := 0
	for sc.Scan() {
		var p Position
		if err := json.Unmarshal(sc.Bytes(), &p); err != nil {
			log.Fatal().Err(err).Msg("position line")
		}
		if n%shards == shard && !done[p.Key] {
			todo = append(todo, p)
		}
		n++
	}
	in.Close()
	if limit > 0 && len(todo) > limit {
		todo = todo[:limit]
	}
	out, err := os.OpenFile(outPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatal().Err(err).Msg("labels")
	}
	defer out.Close()
	fmt.Fprintf(os.Stderr, "%d positions in %s; shard %d/%d; %d already labelled; %d to do; %d-ply, top %d, %d at once\n",
		n, inPath, shard, shards, len(done), len(todo), plies, cands, threads)

	var mu sync.Mutex
	var finished, iters atomic.Int64
	start := time.Now()
	stop := make(chan struct{})
	go func() {
		t := time.NewTicker(30 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				f := finished.Load()
				el := time.Since(start)
				rate := float64(f) / el.Hours()
				eta := "?"
				if f > 0 {
					eta = (time.Duration(float64(el) / float64(f) * float64(int64(len(todo))-f))).Round(time.Minute).String()
				}
				fmt.Fprintf(os.Stderr, "%s  %d/%d positions  %.0f/hour  ETA %s\n", time.Now().Format("2006-01-02 15:04:05"), f, len(todo), rate, eta)
			}
		}
	}()
	work := make(chan Position)
	var wg sync.WaitGroup
	for w := 0; w < threads; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range work {
				l, err := simOne(cfg, p, plies, cands)
				if err != nil {
					log.Error().Err(err).Str("key", p.Key).Msg("sim failed; skipped")
					continue
				}
				b, _ := json.Marshal(l)
				mu.Lock()
				out.Write(append(b, '\n'))
				mu.Unlock()
				finished.Add(1)
				iters.Add(int64(l.Iterations))
			}
		}()
	}
	for _, p := range todo {
		work <- p
	}
	close(work)
	wg.Wait()
	close(stop)
	el := time.Since(start)
	fmt.Fprintf(os.Stderr, "done: %d positions in %s (%.1f CPU-seconds each)\n", finished.Load(), el.Round(time.Second),
		el.Seconds()*float64(threads)/float64(max(1, finished.Load())))
}

// simOne simulates one position: the top cands plays by static equity,
// plies deep, stopping rule Stop99, one thread, the opponent's rack drawn
// from the unseen tiles every iteration (no inference).
func simOne(cfg *config.Config, p Position, plies, cands int) (*Label, error) {
	t0 := time.Now()
	g, err := cgp.ParseCGP(cfg, p.CGP)
	if err != nil {
		return nil, err
	}
	conf := &bot.BotConfig{Config: *cfg}
	btp, err := bot.NewBotTurnPlayerFromGame(g.Game, conf, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return nil, err
	}
	btp.SetBackupMode(game.SimulationMode)
	btp.SetStateStackLength(1)
	btp.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	btp.RecalculateBoard()
	btp.MoveGenerator().(*movegen.GordonGenerator).SetPlayRecorder(movegen.AllPlaysRecorder)
	moves := btp.GenerateMoves(cands)
	if len(moves) == 0 {
		return nil, fmt.Errorf("no moves")
	}
	c, err := equity.NewCombinedStaticCalculator(btp.LexiconName(), cfg, "", equity.PEGAdjustmentFilename)
	if err != nil {
		return nil, err
	}
	s := &montecarlo.Simmer{}
	s.Init(btp.Game, []equity.EquityCalculator{c}, c, cfg)
	s.TryLoadWMP(cfg.WGLConfig(), btp.LexiconName())
	s.SetThreads(1)
	if err := s.PrepareSim(plies, moves); err != nil {
		return nil, err
	}
	s.SetStoppingCondition(montecarlo.Stop99)
	if err := s.Simulate(context.Background()); err != nil {
		return nil, err
	}
	l := &Label{Key: p.Key, Plies: plies, Iterations: s.Iterations()}
	for _, sp := range s.PlaysByWinProb().PlaysNoLock() {
		m := sp.Move()
		l.Cands = append(l.Cands, Candidate{Move: m.ShortDescription(), Score: m.Score(), Equity: m.Equity(),
			Win: sp.WinProb(), WinSE: sp.WinProbStdErr(), Eq: sp.EquityMean(), EqSE: sp.EquityStdErr(),
			Iters: sp.WinProbIterations(), Ignored: sp.IsIgnored()})
	}
	l.Seconds = time.Since(t0).Seconds()
	return l, nil
}
