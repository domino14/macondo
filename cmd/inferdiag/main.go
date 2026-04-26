// inferdiag — Tier 1 + Tier 2 Bayesian inference diagnostic harness.
//
// Generates simming-v-hasty games in parallel. Player 0 is a simming bot
// (SIMMING_BOT_NO_EG); player 1 is a hasty bot that runs inference.
// After each simming-bot move, the hasty bot's RangeFinder infers the
// simmer's kept leave (Tier 1) and optionally evaluates move quality under
// three rack assumptions (Tier 2): no-info, inferred posterior, oracle
// (true kept leave). Hasty-bot moves are skipped because softmaxLikelihood
// is calibrated for simming play only.
//
// Threading model: within a single game, sim, inference, and tier2 eval are
// all sequential, so peak thread usage per game = max(sim-threads,
// infer-threads, tier2-threads). Total = parallel-games × that peak.
//
// Usage:
//
//	inferdiag [flags]
//
// Example (192-core machine, 24 parallel games × 8 threads = 192 cores):
//
//	inferdiag --games 500 --parallel-games 24 \
//	          --sim-plies 5 --sim-threads 8 \
//	          --infer-threads 8 --infer-time-secs 20 \
//	          --tier2 --tier2-threads 8 \
//	          --lexicon NWL23 --output results.jsonl
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/rangefinder"
)

// ---------------------------------------------------------------------------
// Record types
// ---------------------------------------------------------------------------

// Tier2Eval is the result of evaluating one rack assumption via sim.
type Tier2Eval struct {
	Move   string  `json:"move"`
	Score  int     `json:"score"`
	WinPct float64 `json:"win_pct"`
}

// Tier2Result holds evaluations under three rack assumptions.
type Tier2Result struct {
	NoInfo              Tier2Eval `json:"no_info"`
	Inferred            Tier2Eval `json:"inferred"`
	Oracle              Tier2Eval `json:"oracle"`
	NoInfoOracleAgree   bool      `json:"no_info_oracle_agree"`
	InferredOracleAgree bool      `json:"inferred_oracle_agree"`
	// OracleAvailable is false when the true leave wasn't found (no oracle eval).
	OracleAvailable bool `json:"oracle_available"`
}

// Record is one JSONL line emitted per inference position.
type Record struct {
	GameID                 int64            `json:"game_id"`
	Turn                   int              `json:"turn"`
	PlayerOnTurn           int              `json:"player_on_turn"`
	MyRack                 string           `json:"my_rack"`
	TrueOppLeave           string           `json:"true_opp_leave"`
	TrueOppLeaveFound      bool             `json:"true_opp_leave_found"`
	RackLength             int              `json:"rack_length"`
	BagUnseen              map[string]int   `json:"bag_unseen"`
	TilesInBag             int              `json:"tiles_in_bag"`
	Mode                   string           `json:"mode"`
	SimCount               uint64           `json:"sim_count"`
	ExhaustiveTotal        int              `json:"exhaustive_total"`
	ElapsedMs              int64            `json:"elapsed_ms"`
	Tau                    float64          `json:"tau"`
	Posterior              []PosteriorEntry `json:"posterior"`
	PosteriorTruncated     bool             `json:"posterior_truncated"`
	PosteriorSumAllWeights float64          `json:"posterior_sum_all_weights"`
	LastOppMoveType        string           `json:"last_opp_move_type"`
	LastOppMoveScore       int32            `json:"last_opp_move_score"`
	LastOppMoveTiles       string           `json:"last_opp_move_tiles,omitempty"`
	Tier2                  *Tier2Result     `json:"tier2,omitempty"`
}

// PosteriorEntry is one rack from the inferred posterior.
type PosteriorEntry struct {
	Leave  string  `json:"leave"`
	Weight float64 `json:"weight"`
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type harnessCfg struct {
	lexicon             string
	letterDist          string
	tau                 float64
	simIters            int
	maxEnumeratedLeaves int
	inferTimeSecs       int
	inferThreads        int
	posteriorTopK       int
	simPlies            int
	simThreads          int
	// tier2
	tier2Enabled    bool
	tier2Plies      int
	tier2Iters      int
	tier2Threads    int
	tier2Candidates int
}

// ---------------------------------------------------------------------------
// Tier 2 evaluator
// ---------------------------------------------------------------------------

// tier2Evaluator holds a Simmer and a move generator for evaluating positions
// under different rack assumptions. Created once per worker goroutine.
type tier2Evaluator struct {
	simmer *montecarlo.Simmer
	gen    *movegen.GordonGenerator
	player *turnplayer.AIStaticTurnPlayer
	hcfg   *harnessCfg
}

func newTier2Evaluator(g *game.Game, calcs []equity.EquityCalculator, cfg *config.Config, hcfg *harnessCfg) (*tier2Evaluator, error) {
	p, err := turnplayer.NewAIStaticTurnPlayerFromGame(g, cfg, calcs)
	if err != nil {
		return nil, err
	}
	gen := p.MoveGenerator().(*movegen.GordonGenerator)
	gen.SetGame(g)

	s := &montecarlo.Simmer{}
	s.Init(g, calcs, calcs[0].(*equity.CombinedStaticCalculator), cfg)
	s.SetThreads(hcfg.tier2Threads)
	s.SetStoppingCondition(montecarlo.StopNone)
	s.SetAutostopIterationsCutoff(hcfg.tier2Iters)

	return &tier2Evaluator{simmer: s, gen: gen, player: p, hcfg: hcfg}, nil
}

// evalUnderAssumption runs a short sim under a given rack assumption and
// returns the winning play's description, score, and win probability.
// The caller must have called PrepareSim with the candidate plays before calling this.
func (e *tier2Evaluator) evalUnderAssumption(ctx context.Context) (Tier2Eval, error) {
	timeout := time.Duration(e.hcfg.tier2Iters/10+5) * time.Second
	simCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if err := e.simmer.Simulate(simCtx); err != nil && simCtx.Err() == nil {
		return Tier2Eval{}, err
	}

	winner := e.simmer.WinningPlay()
	if winner == nil {
		return Tier2Eval{}, fmt.Errorf("no winning play")
	}
	m := winner.Move()
	return Tier2Eval{
		Move:   m.ShortDescription(),
		Score:  m.Score(),
		WinPct: winner.WinProb(),
	}, nil
}

// runTier2 evaluates the current position under no-info, inferred, and oracle
// rack assumptions using short sims.
func runTier2(
	g *game.Game,
	ev *tier2Evaluator,
	inferredRacks []montecarlo.InferredRack,
	rackLength int,
	trueLeave []tilemapping.MachineLetter,
	trueLeaveFound bool,
	ctx context.Context,
) (*Tier2Result, error) {
	playerIdx := g.PlayerOnTurn()

	// Generate top-N candidate moves statically.
	oppRack := g.RackFor(1 - playerIdx)
	unseen := int(oppRack.NumTiles()) + g.Bag().TilesRemaining()
	exchAllowed := unseen-game.RackTileLimit >= g.ExchangeLimit()
	ev.gen.SetGame(g)
	ev.gen.SetRecordNTopPlays(ev.hcfg.tier2Candidates)
	ev.gen.SetMaxCanExchange(game.MaxCanExchange(unseen-game.RackTileLimit, g.ExchangeLimit()))
	ev.gen.GenAll(g.RackFor(playerIdx), exchAllowed)
	plays := ev.gen.Plays()
	if len(plays) == 0 {
		return nil, fmt.Errorf("no candidate plays generated")
	}

	// Convert []*move.Move to []*move.Move (Plays() already returns that).
	movePtrs := make([]*move.Move, len(plays))
	copy(movePtrs, plays)

	result := &Tier2Result{OracleAvailable: trueLeaveFound}

	// Helper: prepare sim, set assumption, evaluate.
	runEval := func(setAssumption func()) (Tier2Eval, error) {
		if err := ev.simmer.PrepareSim(ev.hcfg.tier2Plies, movePtrs); err != nil {
			return Tier2Eval{}, err
		}
		setAssumption()
		return ev.evalUnderAssumption(ctx)
	}

	var err error

	// 1. No-info: no rack constraint (PrepareSim already clears knownOppRack).
	result.NoInfo, err = runEval(func() {})
	if err != nil {
		return nil, fmt.Errorf("tier2 no-info: %w", err)
	}

	// 2. Inferred: use posterior racks.
	if len(inferredRacks) > 0 {
		result.Inferred, err = runEval(func() {
			ev.simmer.SetInferences(inferredRacks, rackLength, montecarlo.InferenceWeightedRandomRacks)
		})
		if err != nil {
			return nil, fmt.Errorf("tier2 inferred: %w", err)
		}
	} else {
		result.Inferred = result.NoInfo // nothing to infer from
	}

	// 3. Oracle: use true kept leave (only if we found it).
	if trueLeaveFound {
		result.Oracle, err = runEval(func() {
			ev.simmer.SetKnownOppRack(trueLeave)
		})
		if err != nil {
			return nil, fmt.Errorf("tier2 oracle: %w", err)
		}
	}

	result.NoInfoOracleAgree = trueLeaveFound && result.NoInfo.Move == result.Oracle.Move
	result.InferredOracleAgree = trueLeaveFound && result.Inferred.Move == result.Oracle.Move

	return result, nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// computeSeed derives a deterministic [32]byte game seed from a base seed and
// game index so that parallel runs over the same --seed produce identical games.
func computeSeed(baseSeed int64, gameIdx int) [32]byte {
	var seed [32]byte
	binary.LittleEndian.PutUint64(seed[0:], uint64(baseSeed))
	binary.LittleEndian.PutUint64(seed[8:], uint64(gameIdx))
	binary.LittleEndian.PutUint64(seed[16:], uint64(baseSeed^int64(gameIdx)))
	binary.LittleEndian.PutUint64(seed[24:], uint64(baseSeed+int64(gameIdx)*1_000_003))
	return seed
}

// buildCalcs constructs the equity calculators needed by the RangeFinder.
// Uses the word-golib cache, so concurrent calls for the same lexicon are free.
func buildCalcs(lexicon string, cfg *config.Config) ([]equity.EquityCalculator, error) {
	c, err := equity.NewCombinedStaticCalculator(lexicon, cfg, "", equity.PEGAdjustmentFilename)
	if err != nil {
		return nil, err
	}
	return []equity.EquityCalculator{c}, nil
}

// ---------------------------------------------------------------------------
// Game loop
// ---------------------------------------------------------------------------

// playGame runs one simming-v-hasty game. After each simming-bot move, the
// hasty bot's RangeFinder infers the simmer's kept leave (Tier 1) and
// optionally evaluates move quality under three rack assumptions (Tier 2).
func playGame(
	runner *automatic.GameRunner,
	ev *tier2Evaluator, // nil if tier2 disabled
	gameIdx int64,
	seed [32]byte,
	calcs []equity.EquityCalculator,
	hcfg *harnessCfg,
	cfg *config.Config,
	recordChan chan<- Record,
	ctx context.Context,
) error {
	runner.StartGameWithSeed(int(gameIdx), seed)
	g := runner.Game()

	for g.Playing() == pb.PlayState_PLAYING {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		currentPlayer := g.PlayerOnTurn()
		err := runner.PlayBestTurn(currentPlayer, true)
		if err != nil {
			return fmt.Errorf("game %d turn %d: %w", gameIdx, g.Turn(), err)
		}

		if g.Bag().TilesRemaining() == 0 {
			continue
		}

		if runner.BotTypeFor(currentPlayer) != pb.BotRequest_SIMMING_BOT_NO_EG {
			continue
		}

		rec, err := runInference(g, ev, gameIdx, calcs, hcfg, cfg, ctx)
		if err != nil {
			log.Debug().Err(err).Int64("game", gameIdx).Int("turn", g.Turn()-1).
				Msg("inference-skipped")
			continue
		}
		recordChan <- rec
	}
	return nil
}

// runInference runs RangeFinder on the current game state and returns a Record.
// g must have its most recent move recorded in history (addToHistory=true).
func runInference(
	g *game.Game,
	ev *tier2Evaluator, // nil if tier2 disabled
	gameIdx int64,
	calcs []equity.EquityCalculator,
	hcfg *harnessCfg,
	cfg *config.Config,
	ctx context.Context,
) (Record, error) {

	inferencer := &rangefinder.RangeFinder{}
	inferencer.Init(g, calcs, cfg)
	if hcfg.inferThreads > 0 {
		inferencer.SetThreads(hcfg.inferThreads)
	}
	if hcfg.tau > 0 {
		inferencer.SetTau(hcfg.tau)
	}
	if hcfg.simIters > 0 {
		inferencer.SetSimIters(hcfg.simIters)
	}
	if hcfg.maxEnumeratedLeaves > 0 {
		inferencer.SetMaxEnumeratedLeaves(hcfg.maxEnumeratedLeaves)
	}

	myRack := g.RackFor(g.PlayerOnTurn()).TilesOn()
	if err := inferencer.PrepareFinder([]tilemapping.MachineLetter(myRack)); err != nil {
		return Record{}, fmt.Errorf("PrepareFinder: %w", err)
	}

	bagMap := inferencer.BagMap()
	alph := g.Alphabet()
	bagUnseen := make(map[string]int, len(bagMap))
	for i, count := range bagMap {
		if count > 0 {
			letter := tilemapping.MachineLetter(i).UserVisible(alph, false)
			bagUnseen[letter] = int(count)
		}
	}

	trueLeave, leaveErr := game.ExtractLastOppLeave(g)
	trueLeaveStr := ""
	trueLeaveFound := leaveErr == nil
	if trueLeaveFound {
		trueLeaveStr = tilemapping.MachineWord(trueLeave).UserVisible(alph)
	}

	lastOppMoveType := ""
	lastOppMoveScore := int32(0)
	lastOppMoveTiles := ""
	evts := g.History().Events
	if len(evts) > 0 {
		evt := evts[len(evts)-1]
		lastOppMoveType = evt.Type.String()
		lastOppMoveScore = evt.Score
		lastOppMoveTiles = evt.PlayedTiles
	}

	timeout := time.Duration(hcfg.inferTimeSecs) * time.Second
	if timeout == 0 {
		timeout = 20 * time.Second
	}
	inferCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if err := inferencer.Infer(inferCtx); err != nil && inferCtx.Err() == nil {
		return Record{}, fmt.Errorf("Infer: %w", err)
	}

	inferences := inferencer.Inferences()
	allRacks := inferences.InferredRacks

	sorted := make([]montecarlo.InferredRack, len(allRacks))
	copy(sorted, allRacks)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Weight > sorted[j].Weight
	})
	totalWeight := 0.0
	for _, ir := range sorted {
		totalWeight += ir.Weight
	}

	topK := hcfg.posteriorTopK
	if topK <= 0 {
		topK = 1000
	}
	truncated := len(sorted) > topK
	if truncated {
		sorted = sorted[:topK]
	}

	posterior := make([]PosteriorEntry, len(sorted))
	for i, ir := range sorted {
		posterior[i] = PosteriorEntry{
			Leave:  tilemapping.MachineWord(ir.Leave).UserVisible(alph),
			Weight: ir.Weight,
		}
	}

	mode := "mc"
	if inferencer.ExhaustiveTotal() > 0 {
		mode = "enumerate"
	}

	rec := Record{
		GameID:                 gameIdx,
		Turn:                   g.Turn() - 1,
		PlayerOnTurn:           g.PlayerOnTurn(),
		MyRack:                 tilemapping.MachineWord(myRack).UserVisible(alph),
		TrueOppLeave:           trueLeaveStr,
		TrueOppLeaveFound:      trueLeaveFound,
		RackLength:             inferences.RackLength,
		BagUnseen:              bagUnseen,
		TilesInBag:             g.Bag().TilesRemaining(),
		Mode:                   mode,
		SimCount:               inferencer.SimCount(),
		ExhaustiveTotal:        inferencer.ExhaustiveTotal(),
		ElapsedMs:              inferencer.InferElapsed().Milliseconds(),
		Tau:                    inferencer.Tau(),
		Posterior:              posterior,
		PosteriorTruncated:     truncated,
		PosteriorSumAllWeights: totalWeight,
		LastOppMoveType:        lastOppMoveType,
		LastOppMoveScore:       lastOppMoveScore,
		LastOppMoveTiles:       lastOppMoveTiles,
	}

	// Tier 2: evaluate move quality under three rack assumptions.
	if ev != nil {
		t2, err := runTier2(g, ev, allRacks, inferences.RackLength, trueLeave, trueLeaveFound, ctx)
		if err != nil {
			log.Debug().Err(err).Msg("tier2-skipped")
		} else {
			rec.Tier2 = t2
		}
	}

	return rec, nil
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

func workerFunc(
	gameIdxChan <-chan int,
	recordChan chan<- Record,
	cfg *config.Config,
	hcfg *harnessCfg,
	baseSeed int64,
	gamesCompleted *atomic.Int64,
	ctx context.Context,
	wg *sync.WaitGroup,
) {
	defer wg.Done()

	calcs, err := buildCalcs(hcfg.lexicon, cfg)
	if err != nil {
		log.Fatal().Err(err).Msg("worker: failed to build equity calculators")
	}

	runner := automatic.NewGameRunner(nil, cfg)
	if err := runner.Init([]automatic.AutomaticRunnerPlayer{
		{
			BotCode:     pb.BotRequest_SIMMING_BOT_NO_EG,
			MinSimPlies: hcfg.simPlies,
			SimThreads:  hcfg.simThreads,
		},
		{
			BotCode: pb.BotRequest_HASTY_BOT,
		},
	}); err != nil {
		log.Fatal().Err(err).Msg("worker: failed to init game runner")
	}

	var ev *tier2Evaluator
	if hcfg.tier2Enabled {
		ev, err = newTier2Evaluator(runner.Game(), calcs, cfg, hcfg)
		if err != nil {
			log.Fatal().Err(err).Msg("worker: failed to init tier2 evaluator")
		}
	}

	for gameIdx := range gameIdxChan {
		if ctx.Err() != nil {
			return
		}
		seed := computeSeed(baseSeed, gameIdx)
		if err := playGame(runner, ev, int64(gameIdx), seed, calcs, hcfg, cfg, recordChan, ctx); err != nil && ctx.Err() == nil {
			log.Err(err).Int("game", gameIdx).Msg("game error")
		}
		gamesCompleted.Add(1)
	}
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	numGames := flag.Int("games", 100, "total number of games to generate")
	parallelGames := flag.Int("parallel-games", 4, "number of games to run concurrently")
	simPlies := flag.Int("sim-plies", 5, "min sim plies for the simming player")
	simThreads := flag.Int("sim-threads", 4, "threads per simming-bot BestPlay call")
	simTimeSecs := flag.Int("sim-time-secs", 20, "timeout per simming-bot move in seconds")
	inferThreads := flag.Int("infer-threads", 0, "threads per RangeFinder call (0 = auto = GOMAXPROCS)")
	inferTimeSecs := flag.Int("infer-time-secs", 20, "inference timeout per position in seconds")
	seed := flag.Int64("seed", 42, "base RNG seed for deterministic games")
	lexicon := flag.String("lexicon", "NWL23", "lexicon name")
	letterDist := flag.String("letter-distribution", "English", "letter distribution name")
	tau := flag.Float64("tau", 0.0, "softmax temperature for inference (0 = use default 0.05)")
	simIters := flag.Int("sim-iters", 0, "mini-sim iterations per rack candidate (0 = default 200)")
	maxEnumLeaves := flag.Int("max-enumerated-leaves", 0, "max leaves for exhaustive enumeration (0 = default 750)")
	posteriorTopK := flag.Int("posterior-top-k", 1000, "max posterior racks to store per record (by weight)")
	output := flag.String("output", "", "output JSONL path (default: inferdiag-<timestamp>.jsonl)")
	verbose := flag.Bool("verbose", false, "enable verbose logging")
	// Tier 2 flags
	tier2 := flag.Bool("tier2", false, "enable Tier 2 move-quality evaluation")
	tier2Plies := flag.Int("tier2-plies", 2, "sim plies for Tier 2 evaluation")
	tier2Iters := flag.Int("tier2-iters", 200, "sim iterations per Tier 2 evaluation")
	tier2Threads := flag.Int("tier2-threads", 4, "threads per Tier 2 sim")
	tier2Candidates := flag.Int("tier2-candidates", 10, "top-N static candidate moves for Tier 2 sim")
	flag.Parse()

	if *verbose {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}

	if *simTimeSecs > 0 {
		automatic.MaxTimePerTurn = time.Duration(*simTimeSecs) * time.Second
	}

	outPath := *output
	if outPath == "" {
		outPath = fmt.Sprintf("inferdiag-%s.jsonl", time.Now().Format("20060102-150405"))
	}

	outFile, err := os.Create(outPath)
	if err != nil {
		log.Fatal().Err(err).Str("path", outPath).Msg("failed to create output file")
	}
	defer outFile.Close()

	writer := bufio.NewWriterSize(outFile, 1<<20)

	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, *lexicon)
	cfg.Set(config.ConfigDefaultLetterDistribution, *letterDist)

	hcfg := &harnessCfg{
		lexicon:             *lexicon,
		letterDist:          *letterDist,
		tau:                 *tau,
		simIters:            *simIters,
		maxEnumeratedLeaves: *maxEnumLeaves,
		inferTimeSecs:       *inferTimeSecs,
		inferThreads:        *inferThreads,
		posteriorTopK:       *posteriorTopK,
		simPlies:            *simPlies,
		simThreads:          *simThreads,
		tier2Enabled:        *tier2,
		tier2Plies:          *tier2Plies,
		tier2Iters:          *tier2Iters,
		tier2Threads:        *tier2Threads,
		tier2Candidates:     *tier2Candidates,
	}

	ctx, cancel := context.WithCancel(context.Background())
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Warn().Str("signal", sig.String()).Msg("shutting down — flushing output")
		cancel()
	}()

	gameIdxChan := make(chan int, *parallelGames*2)
	recordChan := make(chan Record, 1024)

	var gamesCompleted atomic.Int64
	var recordsEmitted atomic.Int64

	var writerWg sync.WaitGroup
	writerWg.Add(1)
	go func() {
		defer writerWg.Done()
		enc := json.NewEncoder(writer)
		for rec := range recordChan {
			if err := enc.Encode(rec); err != nil {
				log.Error().Err(err).Msg("failed to write record")
			}
			recordsEmitted.Add(1)
		}
		writer.Flush()
	}()

	stopProgress := make(chan struct{})
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				done := gamesCompleted.Load()
				emitted := recordsEmitted.Load()
				fmt.Fprintf(os.Stderr, "progress: %d/%d games done, %d records emitted\n",
					done, *numGames, emitted)
			case <-stopProgress:
				return
			}
		}
	}()

	var workerWg sync.WaitGroup
	for i := 0; i < *parallelGames; i++ {
		workerWg.Add(1)
		go workerFunc(gameIdxChan, recordChan, cfg, hcfg, *seed,
			&gamesCompleted, ctx, &workerWg)
	}

	for i := 0; i < *numGames; i++ {
		if ctx.Err() != nil {
			break
		}
		gameIdxChan <- i
	}
	close(gameIdxChan)

	workerWg.Wait()
	close(recordChan)
	writerWg.Wait()

	close(stopProgress)
	cancel()

	done := gamesCompleted.Load()
	emitted := recordsEmitted.Load()
	fmt.Fprintf(os.Stderr, "done: %d games, %d records written to %s\n", done, emitted, outPath)
}
