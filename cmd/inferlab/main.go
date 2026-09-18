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
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/rangefinder"
)

// nullFloat marshals a non-finite float as JSON null instead of failing. The
// lift is NaN exactly when the posterior gave the true leave no weight at all,
// which is the worst outcome there is and the case most worth keeping;
// encoding/json refuses NaN outright, so without this those were the records
// being dropped.
type nullFloat float64

func (f nullFloat) MarshalJSON() ([]byte, error) {
	v := float64(f)
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return []byte("null"), nil
	}
	return json.Marshal(v)
}

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
	Posterior float64   `json:"posterior"`
	Prior     float64   `json:"prior"`
	LiftBits  nullFloat `json:"liftBits"`
	Rank      int       `json:"rank"`
	Leaves    int       `json:"leaves"`
	Measured  bool      `json:"measured"`
	RuledOut  bool      `json:"ruledOut"`

	// What the loop spent and how it thought it was doing.
	SimCount     int     `json:"simCount"`
	MeasuredMass float64 `json:"measuredMass"`
	LogCalib     float64 `json:"logCalib"`
	LogCalibIn   float64 `json:"logCalibInSample"`
	ElapsedMS    int64   `json:"elapsedMs"`
	ValueBeta    float64 `json:"valueBeta"`
	Seed         uint64  `json:"seed"`
	SeedMode     string  `json:"seedMode"`
	ForcedTruth  bool    `json:"forcedTruth"`
	// The decoy is a leave drawn at random from the same unseen pool and forced
	// in beside the truth, so the two are measured under identical conditions.
	DecoyLeave    string    `json:"decoyLeave,omitempty"`
	DecoyLiftBits nullFloat `json:"decoyLiftBits"`
	DecoyRank     int       `json:"decoyRank"`
	DecoyMeasured bool      `json:"decoyMeasured"`
	Proposal      string    `json:"proposal"`
	Repeat        int       `json:"repeat"`

	// What the original run scored this same position, for a sanity check that
	// the replay is reproducing it.
	LoggedLiftBits nullFloat `json:"loggedLiftBits"`
	LoggedMeasured bool      `json:"loggedMeasured"`
	HasLogged      bool      `json:"hasLogged"`

	Rounds []rangefinder.RoundRecord  `json:"rounds,omitempty"`
	Probe  *rangefinder.OrderingProbe `json:"probe,omitempty"`
	Draws  []rangefinder.DrawRecord   `json:"draws,omitempty"`

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
		tauSchedule = flag.Bool("tau-schedule", false, "use the bag-size tau schedule (0.1 at 8-20 in the bag, 0.3 at 7 or fewer) instead of the fixed default")
		budget      = flag.Int("budget", 200, "leaves to measure")
		rounds      = flag.Int("rounds", rangefinder.DefaultMaxRefineRounds, "refine rounds")
		simIters    = flag.Int("simiters", 0, "mini-sim iterations per leaf (0 = default)")
		maxLeaves   = flag.Int("maxleaves", 0, "enumeration ceiling (0 = default)")
		inferThread = flag.Int("infer-threads", 1, "threads inside one inference")
		trace       = flag.Bool("trace", false, "record every measured leaf and what was predicted for it")
		proposal    = flag.String("proposal", "posterior", "how refine rounds pick leaves: "+
			"posterior (the engine's own), prior (ignore the model and draw from the tile counts), "+
			"floor (posterior with -floor of each round's mass reserved for the prior)")
		floor = flag.Float64("floor", 0.25, "share of each round's draws reserved for the prior, with -proposal floor")
		order = flag.Int("order", 0, "cap on the sub-leave expansion order (0 = the engine's rule, "+
			"ceil(k/2) capped at 3). 4 lets the model carry four-way interactions.")
		valueTerm = flag.String("value", "auto", "static leave value in the imputation: auto (the engine's "+
			"rule, value-first at five tiles and up), off, residual "+
			"(added on top of the sub-leave terms, slope fit to what they leave unexplained), only "+
			"(in place of them), or first (value as the baseline, sub-leave terms fit to what it leaves unexplained)")
		impLambda = flag.Float64("lambda", 0, "imputation shrinkage pseudo-count (0 = the engine's 10). "+
			"Larger pulls thin sub-leave terms harder toward no effect.")
		calibShrink = flag.Float64("calib-shrink", -1, "how far the imputation's calibration constant "+
			"moves from its in-sample fit toward the cross-fitted one: 1 is the engine's behavior, 0 keeps "+
			"the in-sample constant. -1 leaves it alone.")
		probe = flag.Bool("probe", false, "after inference, measure leaves the model ranked above and "+
			"below the true leave without feeding them back, to test whether imputed weights are in the "+
			"right order. Uses the answer to choose where to look, so diagnostic only.")
		probeTop   = flag.Int("probe-top", 10, "with -probe: highest-weighted unmeasured leaves above the truth to measure")
		probeAbove = flag.Int("probe-above", 15, "with -probe: further leaves drawn uniformly from above the truth")
		probeBelow = flag.Int("probe-below", 15, "with -probe: leaves drawn uniformly from below the truth")
		forceTruth = flag.Bool("force-truth", false, "measure the leave the opponent really held, whether "+
			"or not the proposal would have found it. Uses the answer, so it is a diagnostic only: it says "+
			"whether a bad read is the model mis-scoring the true leave or never looking at it. A decoy "+
			"leave, drawn at random from the same unseen pool, is forced in alongside it and scored the "+
			"same way -- without that control there is no telling a model that recognizes the truth from "+
			"one that simply flatters whatever it measures.")
		seed = flag.String("seed", "original", "which random numbers to replay with. "+
			"\"original\" takes the game's own seed out of its ID, so the baseline reproduces the run "+
			"exactly and the logged results serve as a free control. A number instead derives a seed from "+
			"the position and that number, which is what -repeat varies. \"none\" leaves each replay "+
			"unseeded, useful only for measuring how noisy a single inference is. "+
			"Either way every variant sees the same draws on the same position.")
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

	var mode rangefinder.ProposalMode
	switch *proposal {
	case "posterior":
		mode = rangefinder.ProposalPosterior
	case "prior":
		mode = rangefinder.ProposalPrior
	case "floor":
		mode = rangefinder.ProposalFloor
	default:
		fmt.Fprintf(os.Stderr, "unknown -proposal %q: want posterior, prior or floor\n", *proposal)
		os.Exit(2)
	}

	seedMode, seedNum := *seed, uint64(0)
	switch *seed {
	case "original", "none":
	default:
		n, err := strconv.ParseUint(*seed, 10, 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "-seed wants \"original\", \"none\" or a number, not %q\n", *seed)
			os.Exit(2)
		}
		seedMode, seedNum = "number", n
	}

	var valueMode rangefinder.ValueMode
	valuePinned := true
	switch *valueTerm {
	case "auto":
		valuePinned = false
	case "off":
		valueMode = rangefinder.ValueOff
	case "residual":
		valueMode = rangefinder.ValueResidual
	case "only":
		valueMode = rangefinder.ValueOnly
	case "first":
		valueMode = rangefinder.ValueFirst
	default:
		fmt.Fprintf(os.Stderr, "-value wants auto, off, residual, only or first, not %q\n", *valueTerm)
		os.Exit(2)
	}

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
						tau: *tau, tauSchedule: *tauSchedule, budget: *budget, rounds: *rounds, simIters: *simIters,
						maxLeaves: *maxLeaves, threads: *inferThread, trace: *trace,
						seed: seedNum, seedMode: seedMode, rep: rep,
						proposal: mode, floor: *floor, forceTruth: *forceTruth,
						lambda: *impLambda, calibShrink: *calibShrink, order: *order, value: valueMode, valuePinned: valuePinned,
						probe: *probe, probeTop: *probeTop, probeAbove: *probeAbove, probeBelow: *probeBelow,
					})
					writeMu.Lock()
					if err := enc.Encode(rec); err != nil {
						// A record that will not serialize is a bug, and losing
						// it quietly biases the corpus toward the cases that do.
						log.Fatal().Err(err).Str("leave", rec.TrueLeave).
							Msg("could not write a record")
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
	tauSchedule                                  bool
	budget, rounds, simIters, maxLeaves, threads int
	trace                                        bool
	seed                                         uint64
	seedMode                                     string
	rep                                          int
	proposal                                     rangefinder.ProposalMode
	floor                                        float64
	forceTruth                                   bool
	lambda, calibShrink                          float64
	order                                        int
	value                                        rangefinder.ValueMode
	valuePinned                                  bool
	probe                                        bool
	probeTop, probeAbove, probeBelow             int
}

// seedFromGameID recovers the seed a paired run gave a game. The ID is
// "seed:" followed by its base64, so a replay can take the very stream the run
// used instead of an independent one.
func seedFromGameID(gid string) ([32]byte, bool) {
	var out [32]byte
	raw, ok := strings.CutPrefix(gid, "seed:")
	if !ok {
		return out, false
	}
	b, err := base64.RawURLEncoding.DecodeString(raw)
	if err != nil || len(b) != 32 {
		return out, false
	}
	copy(out[:], b)
	return out, true
}

// drawDecoy picks k tiles at random from everything the inferring player cannot
// see -- the bag plus the opponent's rack, which is exactly the pool the true
// leave came from. Seeded, so a decoy is the same across variants.
func drawDecoy(g *game.Game, k int, seed int64) []tilemapping.MachineLetter {
	pool := append([]tilemapping.MachineLetter{}, g.Bag().Peek()...)
	pool = append(pool, g.RackFor(1-g.PlayerOnTurn()).TilesOn()...)
	if len(pool) < k {
		return nil
	}
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	out := append([]tilemapping.MachineLetter{}, pool[:k]...)
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
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
		LoggedLiftBits: nullFloat(pos.LoggedLiftBits), LoggedMeasured: pos.LoggedMeasured,
		HasLogged: pos.HasLoggedResult,
	}

	// Common random numbers. A replayed game carries no seed of its own, so
	// inference would sample differently every time and a comparison between
	// two settings would mostly be measuring that.
	g := pos.Game
	switch o.seedMode {
	case "original":
		// The game's ID is its seed, so the replay can use the very stream the
		// run used and reproduce its answer rather than merely resemble it.
		if sd, ok := seedFromGameID(pos.GameID); ok {
			g = pos.Game.CopyWithHistory()
			g.SeedBag(sd)
			rec.SeedMode = "original"
		} else {
			rec.SeedMode = "original(unavailable)"
		}
	case "number":
		g = pos.Game.CopyWithHistory()
		g.SeedBag(positionSeed(pos, o.seed+uint64(o.rep)))
		rec.SeedMode = "number"
	default:
		rec.SeedMode = "none"
	}

	rf := &rangefinder.RangeFinder{}
	rf.Init(g, calcs, cfg)
	rf.SetThreads(max(1, o.threads))
	rf.SetTau(o.tau)
	rf.SetTauSchedule(o.tauSchedule)
	rf.SetBudget(o.budget)
	rf.SetMaxRounds(o.rounds)
	rf.SetMaxEnumeratedLeaves(o.maxLeaves)
	if o.simIters > 0 {
		rf.SetSimIters(o.simIters)
	}
	rf.SetTracing(o.trace)
	rf.SetProposalMode(o.proposal)
	rf.SetExplorationFloor(o.floor)
	if o.lambda > 0 {
		rf.SetImputationLambda(o.lambda)
	}
	if o.calibShrink >= 0 {
		rf.SetCalibrationShrink(o.calibShrink)
	}
	if o.order > 0 {
		rf.SetMaxMarginalOrder(o.order)
	}
	if o.valuePinned {
		rf.SetValueTerm(o.value)
	}

	rec.Seed = o.seed + uint64(o.rep)
	rec.Proposal = []string{"posterior", "prior", "floor"}[o.proposal]
	rec.Repeat = o.rep
	truthTiles := tilemapping.RackFromString(pos.TrueLeave, g.Alphabet()).TilesOn()
	var decoyTiles []tilemapping.MachineLetter
	if o.forceTruth {
		decoyTiles = drawDecoy(g, len(truthTiles), int64(o.seed)+int64(o.rep)+int64(pos.Turn))
		forced := [][]tilemapping.MachineLetter{truthTiles}
		if decoyTiles != nil {
			forced = append(forced, decoyTiles)
			rec.DecoyLeave = tilemapping.MachineWord(decoyTiles).UserVisible(g.Alphabet())
		}
		rf.SetForcedLeaves(forced)
		rec.ForcedTruth = true
	}

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

	score := rf.ScoreLeave(truthTiles)
	if o.probe {
		pr, err := rf.ProbeOrdering(ctx, truthTiles, o.probeTop, o.probeAbove, o.probeBelow,
			int64(o.seed)+int64(o.rep)+int64(pos.Turn)*7919)
		if err != nil {
			rec.Err = "probe: " + err.Error()
		} else {
			rec.Probe = pr
		}
	}
	if decoyTiles != nil {
		d := rf.ScoreLeave(decoyTiles)
		rec.DecoyLiftBits = nullFloat(d.LiftBits)
		rec.DecoyRank = d.Rank
		rec.DecoyMeasured = d.Measured
	}
	rec.Posterior = score.Posterior
	rec.Prior = score.Prior
	rec.LiftBits = nullFloat(score.LiftBits)
	rec.Rank = score.Rank
	rec.Leaves = score.Leaves
	rec.Measured = score.Measured
	rec.RuledOut = !score.InPosterior || score.Posterior == 0
	rec.SimCount = int(rf.SimCount())

	if lc, lci, mm, vb, ok := rf.ImputeStats(); ok {
		rec.LogCalib, rec.LogCalibIn, rec.MeasuredMass, rec.ValueBeta = lc, lci, mm, vb
	}
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
