// inferlab replays a finished autoplay run's inference positions and reports,
// for each one, what the engine believed and what was actually true.
//
// The point is paired comparison. A change to the rangefinder tested by running
// fresh games needs thousands of them to see past the noise, and the run these
// positions came from took 43 hours. Replaying the same positions under two
// settings removes the noise instead of averaging it away: identical boards,
// identical racks, identical answers, so every position is its own control.
//
// Each position produces one JSONL record holding the final score against the
// true leave, the per-round summary of the measure-impute-recalibrate loop, and
// -- with -trace -- every leaf the engine chose to measure, with what the model
// predicted for it beforehand. That last part is what separates "the model
// mis-scored the leaves it saw" from "the model never looked at the right
// leaf", which are the two ways a read goes wrong and want opposite fixes.
//
// Usage:
//
//	inferlab -turnlog experiments/infer-v-sim-200.txt -out lab.jsonl \
//	    -leavelen 6 -positions 200 -tau 0.05 -budget 200 -trace
package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/rangefinder"
)

// Record is one replayed position.
type Record struct {
	GameID string `json:"gameID"`
	Half   int    `json:"half"`
	Turn   int    `json:"turn"`

	// Variant labels the configuration that produced this record, so several
	// runs can be concatenated into one file and compared by it.
	Variant string `json:"variant"`

	// The problem.
	TrueLeave      string `json:"trueLeave"`
	OppRack        string `json:"oppRack"`
	OppPlay        string `json:"oppPlay"`
	OppScore       int    `json:"oppScore"`
	TilesRemaining int    `json:"tilesRemaining"`
	LeaveLen       int    `json:"leaveLen"`

	// The answer this replay gave.
	Posterior float64 `json:"posterior"`
	Prior     float64 `json:"prior"`
	LiftBits  float64 `json:"liftBits"`
	Rank      int     `json:"rank"`
	Leaves    int     `json:"leaves"`
	Measured  bool    `json:"measured"`
	RuledOut  bool    `json:"ruledOut"`

	// What the loop spent and how it thought it was doing.
	SimCount     int     `json:"simCount"`
	MeasuredMass float64 `json:"measuredMass"`
	LogCalib     float64 `json:"logCalib"`
	LogCalibIn   float64 `json:"logCalibInSample"`
	ElapsedMS    int64   `json:"elapsedMs"`
	Seed         uint64  `json:"seed"`
	Repeat       int     `json:"repeat"`

	// What the original run scored this same position, for a sanity check that
	// the replay is reproducing it.
	LoggedLiftBits float64 `json:"loggedLiftBits"`
	LoggedMeasured bool    `json:"loggedMeasured"`
	HasLogged      bool    `json:"hasLogged"`

	Rounds []rangefinder.RoundRecord `json:"rounds,omitempty"`
	Draws  []rangefinder.DrawRecord  `json:"draws,omitempty"`

	Err string `json:"err,omitempty"`
}

func main() {
	var (
		turnlog  = flag.String("turnlog", "", "per-turn log from a finished run (required)")
		lexicon  = flag.String("lexicon", "NWL23", "lexicon")
		letterd  = flag.String("letterdist", "english", "letter distribution")
		out      = flag.String("out", "inferlab.jsonl", "where to write records")
		variant  = flag.String("variant", "baseline", "label for this configuration")
		parallel = flag.Int("parallel", runtime.NumCPU(), "positions to replay at once")

		// Which positions.
		positions = flag.Int("positions", 0, "cap on positions (0 = all matching)")
		leavelen  = flag.Int("leavelen", 0, "only leaves of this many tiles (6 = one-tile plays)")
		minbag    = flag.Int("minbag", 0, "only positions with at least this many tiles in the bag")
		maxbag    = flag.Int("maxbag", 0, "only positions with at most this many (0 = no limit)")
		worseThan = flag.Float64("worsethan", 0, "only inferences the run scored below this many bits")
		allLifts  = flag.Bool("alllifts", true, "take every inference, not only the bad ones")

		// How to infer.
		tau         = flag.Float64("tau", 0, "softmax temperature (0 = engine default)")
		budget      = flag.Int("budget", 200, "leaves to measure")
		rounds      = flag.Int("rounds", rangefinder.DefaultMaxRefineRounds, "refine rounds")
		simIters    = flag.Int("simiters", 0, "mini-sim iterations per leaf (0 = default)")
		maxLeaves   = flag.Int("maxleaves", 0, "enumeration ceiling (0 = default)")
		inferThread = flag.Int("infer-threads", 1, "threads inside one inference")
		trace       = flag.Bool("trace", false, "record every measured leaf and what was predicted for it")
		seed        = flag.Uint64("seed", 1, "common random numbers: every variant replaying a position "+
			"with the same seed draws the same leaves, so a comparison measures the change and not the RNG. "+
			"0 leaves each replay unseeded, which is only useful for measuring that noise.")
		repeat = flag.Int("repeat", 1, "replay each position this many times, with seed, seed+1, ... "+
			"Averaging over repeats is how to see past the per-position noise.")
	)
	flag.Parse()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)

	if *turnlog == "" {
		fmt.Fprintln(os.Stderr, "-turnlog is required")
		flag.Usage()
		os.Exit(2)
	}

	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, *lexicon)
	cfg.Set(config.ConfigDefaultLetterDistribution, *letterd)

	filter := automatic.CorpusFilter{
		LeaveLen: *leavelen, MinBag: *minbag, MaxBag: *maxbag,
		WorseThan: *worseThan, AllLifts: *allLifts, Limit: *positions,
	}
	fmt.Fprintf(os.Stderr, "loading corpus from %s ...\n", *turnlog)
	corpus, skipped, err := automatic.LoadCorpusVerbose(cfg, *turnlog, *lexicon, *letterd, "", filter)
	if err != nil {
		log.Fatal().Err(err).Msg("loading corpus")
	}
	fmt.Fprintf(os.Stderr, "%d positions (%d games skipped as unreplayable)\n", len(corpus), skipped)
	if len(corpus) == 0 {
		return
	}

	f, err := os.Create(*out)
	if err != nil {
		log.Fatal().Err(err).Msg("creating output")
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()

	var writeMu sync.Mutex
	enc := json.NewEncoder(w)

	work := make(chan *automatic.CorpusPosition)
	var done int64
	start := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < *parallel; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			calc, err := equity.NewCombinedStaticCalculator(*lexicon, cfg, "", equity.PEGAdjustmentFilename)
			if err != nil {
				log.Error().Err(err).Msg("building equity calculator")
				return
			}
			calcs := []equity.EquityCalculator{calc}

			for pos := range work {
				for rep := 0; rep < max(1, *repeat); rep++ {
					rec := replay(pos, calcs, cfg, *variant, replayOpts{
						tau: *tau, budget: *budget, rounds: *rounds, simIters: *simIters,
						maxLeaves: *maxLeaves, threads: *inferThread, trace: *trace,
						seed: *seed, rep: rep,
					})
					writeMu.Lock()
					if err := enc.Encode(rec); err != nil {
						log.Error().Err(err).Msg("writing record")
					}
					writeMu.Unlock()
				}
				n := atomic.AddInt64(&done, 1)
				if n%25 == 0 || int(n) == len(corpus) {
					el := time.Since(start)
					fmt.Fprintf(os.Stderr, "  %d/%d  %.0fs elapsed, ~%.0fs left\n",
						n, len(corpus), el.Seconds(),
						el.Seconds()/float64(n)*float64(len(corpus)-int(n)))
				}
			}
		}()
	}
	for _, pos := range corpus {
		work <- pos
	}
	close(work)
	wg.Wait()
	fmt.Fprintf(os.Stderr, "done in %.0fs -> %s\n", time.Since(start).Seconds(), *out)
}

type replayOpts struct {
	tau                                          float64
	budget, rounds, simIters, maxLeaves, threads int
	trace                                        bool
	seed                                         uint64
	rep                                          int
}

// positionSeed mixes a position's identity with the run's seed, so the same
// position gets the same draws across variants and different positions stay
// independent of one another.
func positionSeed(pos *automatic.CorpusPosition, seed uint64) [32]byte {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%d|%d|%d", pos.GameID, pos.Half, pos.Turn, seed)
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

// replay runs one position through the rangefinder and grades it.
func replay(pos *automatic.CorpusPosition, calcs []equity.EquityCalculator,
	cfg *config.Config, variant string, o replayOpts) Record {

	rec := Record{
		GameID: pos.GameID, Half: pos.Half, Turn: pos.Turn, Variant: variant,
		TrueLeave: pos.TrueLeave, OppRack: pos.OppRack, OppPlay: pos.OppPlay,
		OppScore: pos.OppScore, TilesRemaining: pos.TilesRemaining,
		LeaveLen:       pos.LeaveLen,
		LoggedLiftBits: pos.LoggedLiftBits, LoggedMeasured: pos.LoggedMeasured,
		HasLogged: pos.HasLoggedResult,
	}

	// Common random numbers. A replayed game carries no seed of its own, so
	// inference would sample differently every time and a comparison between
	// two settings would mostly be measuring that. Seeding from the position's
	// own identity gives every variant the same draws on the same position
	// while keeping different positions independent.
	g := pos.Game
	if o.seed != 0 {
		g = pos.Game.CopyWithHistory()
		g.SeedBag(positionSeed(pos, o.seed+uint64(o.rep)))
	}

	rf := &rangefinder.RangeFinder{}
	rf.Init(g, calcs, cfg)
	rf.SetThreads(max(1, o.threads))
	rf.SetTau(o.tau)
	rf.SetBudget(o.budget)
	rf.SetMaxRounds(o.rounds)
	rf.SetMaxEnumeratedLeaves(o.maxLeaves)
	if o.simIters > 0 {
		rf.SetSimIters(o.simIters)
	}
	rf.SetTracing(o.trace)

	rec.Seed = o.seed + uint64(o.rep)
	rec.Repeat = o.rep
	myRack := g.RackFor(g.PlayerOnTurn()).TilesOn()
	if err := rf.PrepareFinder([]tilemapping.MachineLetter(myRack)); err != nil {
		rec.Err = err.Error()
		return rec
	}

	// A leaf budget is what stops it; the deadline is only a backstop against
	// a position that will not terminate.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	started := time.Now()
	if err := rf.Infer(ctx); err != nil {
		rec.Err = err.Error()
		return rec
	}
	rec.ElapsedMS = time.Since(started).Milliseconds()

	truth := tilemapping.RackFromString(pos.TrueLeave, g.Alphabet()).TilesOn()
	score := rf.ScoreLeave(truth)
	rec.Posterior = score.Posterior
	rec.Prior = score.Prior
	rec.LiftBits = score.LiftBits
	rec.Rank = score.Rank
	rec.Leaves = score.Leaves
	rec.Measured = score.Measured
	rec.RuledOut = !score.InPosterior || score.Posterior == 0
	rec.SimCount = int(rf.SimCount())

	if tr := rf.Trace(); tr != nil {
		rec.Rounds = tr.Rounds
		rec.Draws = tr.Draws
		if n := len(tr.Rounds); n > 0 {
			rec.MeasuredMass = tr.Rounds[n-1].MeasuredMass
			rec.LogCalib = tr.Rounds[n-1].LogCalib
			rec.LogCalibIn = tr.Rounds[n-1].LogCalibInSample
		}
	}
	return rec
}
