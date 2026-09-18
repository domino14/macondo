package rangefinder

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"runtime"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/proto"
	"gopkg.in/yaml.v3"

	"github.com/domino14/macondo/ai/simplesimmer"
	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
)

var ErrMoveTypeNotSupported = errors.New("opponent move type not suitable for inference")
var ErrNoEvents = errors.New("no events")
var ErrBagEmpty = errors.New("bag is empty")
var ErrNoInformation = errors.New("not enough information to infer")

const (
	// SoftmaxTemperature controls how "rational" we assume the opponent to be
	// when computing P(play | leave). Lower values assume near-optimal play;
	// higher values allow more weight for sub-optimal plays. This is the
	// early-game value; tauForBag raises it as the bag empties.
	// Softmax is applied over log-odds of win probabilities, so tau is on the
	// log-odds scale. Typical positions (20%-80% win prob) span roughly [-1.4, 1.4];
	// strongly won/lost positions (5%-95%) reach about [-3, 3].
	SoftmaxTemperature = 0.05

	// logitEps clamps win probabilities away from 0 and 1 before logit
	// conversion to avoid ±Inf.
	logitEps = 1e-6
)

type LogIteration struct {
	Iteration      int     `json:"iteration" yaml:"iteration"`
	Thread         int     `json:"thread" yaml:"thread"`
	Rack           string  `json:"rack" yaml:"rack"`
	TopMoveWinProb float64 `json:"topMoveWinProb" yaml:"topMoveWinProb"`
	TopMove        string  `json:"topMove" yaml:"topMove"`
	// InferredMoveWinProb is the win prob of the move we are inferring, given
	// that they drew "Rack"
	InferredMoveWinProb float64 `json:"inferredMoveWinProb" yaml:"inferredMoveWinProb"`
	// Likelihood = softmax P(play|leave) with temperature tau
	Likelihood float64 `json:"likelihood" yaml:"likelihood"`
	SimLogFile string  `json:"simLogFile,omitempty" yaml:"simLogFile,omitempty"`
}

type Inference struct {
	RackLength    int
	InferredRacks []montecarlo.InferredRack
	// Complete is true when InferredRacks covers every feasible leave — a
	// full posterior the simmer can sample from directly, with no random
	// fallback. Tile-placement inference always produces a complete
	// posterior (measured leaves use their evaluated likelihood, the rest
	// are imputed from marginal lifts); exchange inference does not.
	Complete bool
	seen     map[string]int // leaveKey -> index in InferredRacks (exchange path)
}

func NewInference() *Inference {
	return &Inference{
		InferredRacks: []montecarlo.InferredRack{},
		seen:          map[string]int{},
	}
}

// leaveKey returns a canonical string key for a leave, for deduplication.
func leaveKey(leave []tilemapping.MachineLetter) string {
	sorted := make([]tilemapping.MachineLetter, len(leave))
	copy(sorted, leave)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	b := make([]byte, len(sorted))
	for i, ml := range sorted {
		b[i] = byte(ml)
	}
	return string(b)
}

// combinatorialPrior computes P(leave) as the multivariate hypergeometric
// probability of drawing exactly the tiles in leave from the bag described
// by bagMap. Returns 0 for impossible leaves.
//
// NOTE: this function is intentionally NOT called by inferSingle (the Monte Carlo
// sampling path).  There, SetRandomRack already draws from the hypergeometric prior,
// so the importance-sampling weight is likelihood only — multiplying by
// combinatorialPrior again would double-count it.
// combinatorialPrior IS used by inferEnumerated, where each leaf is visited
// exactly once (not sampled), so the prior must be supplied explicitly.
func combinatorialPrior(leave []tilemapping.MachineLetter, bagMap []uint8) float64 {
	if len(leave) == 0 {
		return 0
	}
	leaveCounts := map[tilemapping.MachineLetter]int{}
	for _, t := range leave {
		leaveCounts[t]++
	}

	N := 0
	for i, c := range bagMap {
		N += int(c)
		if leaveCounts[tilemapping.MachineLetter(i)] > int(c) {
			return 0 // impossible leave
		}
	}
	if N == 0 {
		return 0
	}
	k := len(leave)

	// logP = Σ logC(bagMap[t], leaveCount[t]) - logC(N, k)
	logP := -logBinomial(N, k)
	for t, lc := range leaveCounts {
		logP += logBinomial(int(bagMap[t]), lc)
	}
	return math.Exp(logP)
}

func logBinomial(n, k int) float64 {
	if k < 0 || k > n {
		return math.Inf(-1)
	}
	lgn, _ := math.Lgamma(float64(n + 1))
	lgk, _ := math.Lgamma(float64(k + 1))
	lgnk, _ := math.Lgamma(float64(n - k + 1))
	return lgn - lgk - lgnk
}

// Logit converts a win probability to log-odds: ln(p / (1-p)).
// p is clamped to [logitEps, 1-logitEps] to avoid ±Inf.
func Logit(p float64) float64 {
	if p < logitEps {
		p = logitEps
	} else if p > 1-logitEps {
		p = 1 - logitEps
	}
	return math.Log(p / (1 - p))
}

// SoftmaxOverLogOdds returns the probability of the play at targetIdx under a
// softmax with temperature tau over the given log-odds vector (win
// probabilities converted with Logit). Using log-odds undoes the implicit
// sigmoid in win probabilities, giving softmax unbounded inputs it is
// designed for. Log-odds are tau-independent, so callers (e.g. tau-fitting
// harnesses) can store them once and evaluate many taus cheaply. This is the
// single source of truth for the likelihood math.
func SoftmaxOverLogOdds(logOdds []float64, targetIdx int, tau float64) float64 {
	if targetIdx < 0 || targetIdx >= len(logOdds) {
		return 0
	}
	// Numerical stability: subtract max before exp.
	maxLogOdds := math.Inf(-1)
	for _, lo := range logOdds {
		if lo > maxLogOdds {
			maxLogOdds = lo
		}
	}
	sum := 0.0
	for _, lo := range logOdds {
		sum += math.Exp((lo - maxLogOdds) / tau)
	}
	return math.Exp((logOdds[targetIdx]-maxLogOdds)/tau) / sum
}

// softmaxLikelihood computes P(targetMove | leave) as a softmax over the
// log-odds of win probabilities of all simmed plays. Returns (likelihood,
// targetWinProb); likelihood is 0 if the target move is not found among the
// plays.
func softmaxLikelihood(plays []*montecarlo.SimmedPlay, targetMove *move.Move, b *board.GameBoard, tau float64) (float64, float64) {
	if len(plays) == 0 {
		return 0, 0
	}

	logOdds := make([]float64, len(plays))
	targetIdx := -1
	for i, sp := range plays {
		logOdds[i] = Logit(sp.WinProb())
		if movesAreTheSame(sp.Move(), targetMove, b) {
			targetIdx = i
		}
	}
	if targetIdx == -1 {
		return 0, 0
	}
	return SoftmaxOverLogOdds(logOdds, targetIdx, tau), plays[targetIdx].WinProb()
}

type RangeFinder struct {
	origGame          *game.Game
	gameCopies        []*game.Game
	equityCalculators []equity.EquityCalculator
	aiplayers         []aiturnplayer.AITurnPlayer
	iterationCount    int
	simCount          atomic.Uint64
	inferElapsed      time.Duration
	// exhaustiveTotal is set when inferEnumerated is used. It records the total
	// number of distinct leaves that existed (before any context timeout). When
	// non-zero, the inference ran in enumeration mode rather than MC sampling.
	exhaustiveTotal int
	threads         int
	// tau is the softmax temperature used when computing P(play | leave).
	// Lower values assume the opponent plays more optimally. Defaults to
	// SoftmaxTemperature if not set explicitly.
	tau float64
	// tauSchedule turns on the bag-size temperature schedule (tauForBag) for
	// positions where tau was not pinned. Off by default: the engine runs at a
	// fixed SoftmaxTemperature until the schedule is tuned as a whole.
	tauSchedule bool
	// phaseTau is the schedule's temperature for the current position, set
	// by PrepareFinder when tauSchedule is on and tau was not pinned.
	phaseTau float64
	// simIters is the max mini-sim iterations per rack candidate.
	// 0 means use the SimpleSimmer default (200).
	simIters int
	// maxEnumeratedLeaves is the threshold for switching from Monte Carlo sampling
	// to exhaustive enumeration. When the number of distinct leaves drawable from
	// the bag (countMultisets) is ≤ this value, inferEnumerated is used instead of
	// inferSingle. 0 means use DefaultMaxEnumeratedLeaves.
	maxEnumeratedLeaves int
	// budget is how many leaves may be evaluated, one mini-sim each. When set it
	// replaces the caller's deadline as the thing that stops inference: a
	// deadline makes the answer depend on how busy the machine was, which is the
	// noise a game-pair run exists to remove. 0 leaves inference bounded by time
	// as before.
	budget int
	// seed derives every random choice inference makes, so that the same
	// position always produces the same posterior. Zero means fall back to the
	// global source, which is what live play wants.
	seed [32]byte
	// stage0Sims is what round 0 spent of the budget, so refinement knows what
	// is left.
	stage0Sims int

	working      bool
	readyToInfer bool

	inferenceBagMap []uint8
	cfg             *config.Config
	lastOppMove     *move.Move
	// tiles used by the last opponent's move, from their rack:
	lastOppMoveRackTiles []tilemapping.MachineLetter
	inference            *Inference

	// Imputation state (tile-placement inference only): containment-marginal
	// accumulator, the same samples partitioned into cross-fitting folds for
	// calibration, per-distinct-leave measured likelihood means, and the
	// diagnostics of the last imputation run.
	acc       *subleaveAccumulator
	foldAccs  []*subleaveAccumulator
	measured  map[string]*measuredLeave
	imputeRes *imputationResult

	// Refinement state: how many rounds of posterior-guided measurement to
	// run after round 0 (0 disables it), how many leaves they measured, and
	// the per-round convergence statistics.
	maxRounds    int
	refinedCount int
	roundLog     []roundStats
	// tracing records every draw and round for offline diagnosis; see trace.go.
	tracing bool
	trace   *InferenceTrace
	// proposalMode and explorationFloor decide how refine rounds pick the
	// leaves they measure; see SetProposalMode.
	proposalMode     ProposalMode
	explorationFloor float64
	// forcedLeaves are measured whether or not the proposal would have found
	// them; see SetForcedLeaves.
	forcedLeaves [][]tilemapping.MachineLetter
	// tuning varies the imputation's own constants; see SetImputationLambda
	// and SetCalibrationShrink.
	tuning        imputationTuning
	stage0Elapsed time.Duration
	// currentRound stamps newly measured leaves with the round that measured
	// them. Written only by refineRounds, between batches.
	currentRound int

	logStream io.Writer
}

func (r *RangeFinder) Init(game *game.Game, eqCalcs []equity.EquityCalculator,
	cfg *config.Config) {

	r.origGame = game
	r.equityCalculators = eqCalcs
	r.threads = max(1, runtime.NumCPU())
	r.cfg = cfg
	r.maxRounds = DefaultMaxRefineRounds
}

func (r *RangeFinder) SetThreads(t int) {
	r.threads = t
}

// SetTau sets the softmax temperature for P(play | leave). Lower values
// assume the opponent plays more optimally; higher values give more weight
// to sub-optimal plays. Must be called before PrepareFinder.
func (r *RangeFinder) SetTau(tau float64) {
	r.tau = tau
}

// Tau is the softmax temperature in use: the value SetTau pinned, or else
// the phase schedule for the position PrepareFinder was given.
func (r *RangeFinder) Tau() float64 {
	if r.tau != 0 {
		return r.tau
	}
	if r.phaseTau != 0 {
		return r.phaseTau
	}
	return SoftmaxTemperature
}

// SetTauSchedule turns the bag-size temperature schedule on or off. A pinned
// tau wins over the schedule either way.
func (r *RangeFinder) SetTauSchedule(on bool) {
	r.tauSchedule = on
}

// scheduleTau sets the schedule's temperature for a position with this many
// tiles left in the bag: tauForBag when the schedule is on, nothing otherwise.
func (r *RangeFinder) scheduleTau(bag int) {
	r.phaseTau = 0
	if r.tauSchedule {
		r.phaseTau = tauForBag(bag)
	}
}

// tauForBag is the softmax temperature for a position with this many tiles
// left in the bag, when the schedule is on and none was pinned.
//
// The temperature says how far the opponent's actual play is trusted to be
// the mini-sim's best one. Early in the game it should be: replayed over an
// independent 5,000-pair run, 0.1 loses a quarter of a bit against 0.05 on
// one-to-four-tile leaves with 21 or more in the bag, and 0.026 is no better.
// Later it should not. With 8 to 20 in the bag, 0.1 gains two thirds of a bit;
// with 7 or fewer, 0.3 gains three and a half, and takes the leaves ruled out
// entirely from 7 to 1 in 143 positions -- at 0.05 a play the mini-sim did not
// rank first gets almost no likelihood, and in the pre-endgame the play a
// 5-ply bot makes is the one a 2-ply mini-sim ranks first least often. 0.92,
// what an MLE fit on real games found for that phase, measures the same as
// 0.3 there, so the schedule takes the value nearer the rest of it.
func tauForBag(bag int) float64 {
	switch {
	case bag <= 7:
		return 0.3
	case bag <= 20:
		return 0.1
	}
	return SoftmaxTemperature
}

// SetMaxRounds sets how many measure–impute–recalibrate rounds run after the
// prior-sampled round 0. 0 disables refinement, leaving single-stage
// inference. Init seeds it with DefaultMaxRefineRounds.
func (r *RangeFinder) SetMaxRounds(n int) {
	r.maxRounds = max(0, n)
}

func (r *RangeFinder) MaxRounds() int { return max(0, r.maxRounds) }

func (r *RangeFinder) SetSimIters(n int) {
	r.simIters = n
}

func (r *RangeFinder) SimIters() int {
	if r.simIters == 0 {
		return 200 // default matches SimpleSimmer default
	}
	return r.simIters
}

// SetMaxEnumeratedLeaves sets the maximum number of distinct leaves for which
// exhaustive enumeration (inferEnumerated) is used instead of Monte Carlo sampling.
// If 0, DefaultMaxEnumeratedLeaves is used.
func (r *RangeFinder) SetMaxEnumeratedLeaves(n int) {
	r.maxEnumeratedLeaves = n
}

// inferenceSeedTag keeps this stream apart from the simmer's, which derives its
// own from the same game seed and turn number.
const inferenceSeedTag uint64 = 0x494e464552 // "INFER"

// deriveSeed builds inference's seed from the game's own, mixing in the turn so
// that each position gets a fresh stream rather than every inference in a game
// replaying the same one. An unseeded game returns the zero value, which leaves
// inference on the global source -- what live play wants.
func (r *RangeFinder) deriveSeed() [32]byte {
	live := r.origGame.Seed()
	if live == ([32]byte{}) {
		return [32]byte{}
	}
	var buf [48]byte
	copy(buf[:32], live[:])
	binary.BigEndian.PutUint64(buf[32:40], uint64(r.origGame.Turn()))
	binary.BigEndian.PutUint64(buf[40:48], inferenceSeedTag)
	return sha256.Sum256(buf[:])
}

// refineSeed derives the refinement sampler's seed from the game's, so the
// leaves it picks to measure are the same on every run. Falls back to a
// clock-based seed for an unseeded game, where nothing replays anyway.
func (r *RangeFinder) refineSeed() int64 {
	if r.seed == ([32]byte{}) {
		return time.Now().UnixNano()
	}
	return int64(binary.BigEndian.Uint64(r.seed[:8]))
}

// stage0Budget is how many leaves round 0 may measure before refinement takes
// over. Refinement only earns its share when there are rounds to run.
func (r *RangeFinder) stage0Budget() int {
	if r.budget <= 0 {
		return 0
	}
	if r.acc == nil || r.MaxRounds() <= 0 {
		return r.budget
	}
	n := int(float64(r.budget) * refineStage0Frac)
	if n < 1 {
		n = 1
	}
	return n
}

// SetBudget bounds inference by leaf evaluations rather than by wall clock.
// Each unit is one mini-sim, so the cost of a turn becomes predictable and,
// more to the point, repeatable. Pass 0 to go back to a time bound.
//
// When a budget is set it also decides the enumerate-or-sample split unless
// SetMaxEnumeratedLeaves says otherwise: a leaf space that fits in the budget
// is measured exactly, and a larger one is sampled to the same count.
func (r *RangeFinder) SetBudget(n int) {
	r.budget = n
}

// Budget returns the leaf-evaluation budget, or 0 when inference is bounded by
// time instead.
func (r *RangeFinder) Budget() int { return r.budget }

// enumerationLimit is the leaf count at or below which the space is measured
// exactly rather than sampled.
func (r *RangeFinder) enumerationLimit() int {
	if r.maxEnumeratedLeaves > 0 {
		return r.maxEnumeratedLeaves
	}
	if r.budget > 0 {
		return r.budget
	}
	return DefaultMaxEnumeratedLeaves
}

func (r *RangeFinder) SetLogStream(l io.Writer) {
	r.logStream = l
}

// BagMap returns a copy of the inferenceBagMap after PrepareFinder has been
// called. It represents the pool of tiles from which the opponent's leave was
// drawn (bag + both racks, minus opp's played tiles this turn).
func (r *RangeFinder) BagMap() []uint8 {
	result := make([]uint8, len(r.inferenceBagMap))
	copy(result, r.inferenceBagMap)
	return result
}

// ExhaustiveTotal returns the total number of distinct leaves that existed
// when inferEnumerated was used. Zero means the MC sampling path was taken.
func (r *RangeFinder) ExhaustiveTotal() int { return r.exhaustiveTotal }

// SimCount returns the total number of mini-simulations run during Infer.
func (r *RangeFinder) SimCount() uint64 { return r.simCount.Load() }

// InferElapsed returns the wall-clock duration of the last Infer call.
func (r *RangeFinder) InferElapsed() time.Duration { return r.inferElapsed }

func (r *RangeFinder) PrepareFinder(myRack []tilemapping.MachineLetter) error {
	r.inference = NewInference()
	evts := r.origGame.History().Events[:r.origGame.Turn()]
	if len(evts) == 0 {
		return ErrNoEvents
	}
	if r.origGame.Bag().TilesRemaining() == 0 {
		return ErrBagEmpty
	}
	r.scheduleTau(r.origGame.Bag().TilesRemaining())
	r.tuning.valueOf = r.leaveValueFunc()

	oppEvtIdx := len(evts) - 1
	oppIdx := evts[oppEvtIdx].PlayerIndex
	var oppEvt *macondo.GameEvent
	foundOppEvent := false
	for oppEvtIdx >= 0 {
		oppEvt = evts[oppEvtIdx]
		if oppEvt.PlayerIndex != oppIdx {
			break
		}
		if oppEvt.Type == macondo.GameEvent_CHALLENGE_BONUS {
			oppEvtIdx--
			continue
		}
		if oppEvt.Type == macondo.GameEvent_EXCHANGE || oppEvt.Type == macondo.GameEvent_TILE_PLACEMENT_MOVE {
			foundOppEvent = true
			break
		}
		oppEvtIdx--
	}
	if !foundOppEvent {
		return ErrMoveTypeNotSupported
	}

	// We must reset the game back to what it looked like before the opp's move.
	var gameCopy *game.Game
	var err error

	history := proto.Clone(r.origGame.History()).(*macondo.GameHistory)
	history.Events = history.Events[:oppEvtIdx]

	if r.origGame.History().StartingCgp != "" {

		parsedCGP, err := cgp.ParseCGP(r.cfg, r.origGame.History().StartingCgp)
		if err != nil {
			return err
		}
		gameCopy = parsedCGP.Game
		gameCopy.History().Events = history.Events

		for t := 0; t < len(history.Events); t++ {
			err = gameCopy.PlayTurn(t)
			if err != nil {
				return err
			}
		}
		gameCopy.SetPlayerOnTurn(int(oppIdx))
		gameCopy.RecalculateBoard()
	} else {
		gameCopy, err = game.NewFromHistory(history, r.origGame.Rules(), len(history.Events))
		if err != nil {
			return err
		}
	}

	// create rack from the last move.
	r.lastOppMove, err = game.MoveFromEvent(oppEvt, r.origGame.Alphabet(), gameCopy.Board())
	if err != nil {
		return err
	}

	if r.lastOppMove.TilesPlayed() == game.RackTileLimit {
		return ErrNoInformation
	}
	r.lastOppMoveRackTiles = []tilemapping.MachineLetter{}
	for _, t := range r.lastOppMove.Tiles() {
		if t == 0 {
			// 0 is the played-through marker when part of a played move.
			continue
		}
		ml := t.IntrinsicTileIdx()
		r.lastOppMoveRackTiles = append(r.lastOppMoveRackTiles, ml)
	}
	r.inference.RackLength = game.RackTileLimit - len(r.lastOppMoveRackTiles)
	log.Info().Int("inference-rack-length", r.inference.RackLength).Msg("preparing inference")

	// Sort to make it easy to compare to other plays:
	sort.Slice(r.lastOppMoveRackTiles, func(i, j int) bool {
		return r.lastOppMoveRackTiles[i] < r.lastOppMoveRackTiles[j]
	})
	gameCopy.ThrowRacksIn()

	if len(myRack) > 0 {
		// Assign my rack first, so that the inferencer doesn't try to
		// assign letters from my rack.
		rack := tilemapping.NewRack(r.origGame.Alphabet())
		rack.Set(myRack)
		err = gameCopy.SetRackForOnly(1-gameCopy.PlayerOnTurn(), rack)
		if err != nil {
			return err
		}
	}

	// Seed the reconstructed position before anything draws from it. This game
	// is built from history rather than copied, so it arrives with a fresh
	// unseeded bag; left alone, every rack inference draws -- and every mini-sim
	// rollout underneath it -- would come off the global source and no two runs
	// would agree.
	r.seed = r.deriveSeed()
	if r.seed != ([32]byte{}) {
		gameCopy.SeedBag(r.seed)
		// The bag this position was rebuilt with came from MakeBag, which
		// shuffles off the global source: its contents follow from the history
		// but its order does not, and a redraw picks tiles by that order. Sort
		// it into a canonical order and shuffle again from the seed, so the
		// order is a function of the seed like everything else.
		tiles := gameCopy.Bag().Tiles()
		slices.Sort(tiles)
		gameCopy.Bag().Shuffle()
	}

	r.inferenceBagMap = gameCopy.Bag().PeekMap()
	if oppEvt.Type == macondo.GameEvent_TILE_PLACEMENT_MOVE {

		_, err = gameCopy.SetRandomRack(gameCopy.PlayerOnTurn(), r.lastOppMoveRackTiles)
		if err != nil {
			return err
		}
		// Save the state of the bag after we assign the random rack. Remove only
		// lastOppMove rack but nothing else.
		for _, ml := range r.lastOppMoveRackTiles {
			r.inferenceBagMap[ml]--
		}
	} else if oppEvt.Type == macondo.GameEvent_EXCHANGE {
		// If this is an exchange move, lastOppMove etc is just a guess.
		// Set any random rack, and don't remove anything from the bagMap.
		// The bagMap already contains the tiles we're about to assign to
		// the user here:
		gameCopy.SetRandomRack(gameCopy.PlayerOnTurn(), nil)
	}
	r.gameCopies = []*game.Game{}
	r.aiplayers = []aiturnplayer.AITurnPlayer{}

	for i := 0; i < r.threads; i++ {
		gc := gameCopy.Copy()
		if r.seed != ([32]byte{}) {
			// Copy shares the source bag's RNG pointer, so give each thread its
			// own object. They all start from the same seed on purpose: a leaf's
			// mini-sim then draws the same numbers whichever thread happens to
			// pick it up, and the leaves are compared under common random
			// numbers rather than independent noise.
			gc.SeedBag(r.seed)
		}
		r.gameCopies = append(r.gameCopies, gc)
		gc.SetRules(gameCopy.Rules())
		simmer, err := simplesimmer.NewSimpleSimmerFromGame(r.gameCopies[i])
		if err != nil {
			return err
		}
		if r.simIters > 0 {
			simmer.SetMaxIters(r.simIters)
		}
		r.aiplayers = append(r.aiplayers, simmer)
	}

	r.readyToInfer = true
	r.iterationCount = 0
	r.simCount.Store(0)
	r.exhaustiveTotal = 0
	r.acc = nil
	r.foldAccs = nil
	r.measured = nil
	r.imputeRes = nil
	r.refinedCount = 0
	r.roundLog = nil
	r.startTrace()
	r.stage0Elapsed = 0
	r.stage0Sims = 0
	return nil
}

// initImputationState (re)allocates the containment-marginal accumulator, the
// per-fold accumulators used to cross-fit the calibration constant, and the
// measured-leave map.
func (r *RangeFinder) initImputationState() {
	order := marginalOrderCapped(r.inference.RackLength, r.tuning.maxOrder)
	r.acc = newSubleaveAccumulator(len(r.inferenceBagMap), order)
	r.foldAccs = make([]*subleaveAccumulator, calibrationFolds)
	for i := range r.foldAccs {
		r.foldAccs[i] = newSubleaveAccumulator(len(r.inferenceBagMap), order)
	}
	r.measured = map[string]*measuredLeave{}
}

// recordPlacementSample records one evaluated leave and its measured
// likelihood into the imputation accumulator and the per-distinct-leave
// means. u is the draw's importance weight P(L)/q(L); prior-sampled draws
// pass 1. The caller must synchronize; leave is sorted in place.
func (r *RangeFinder) recordPlacementSample(leave []tilemapping.MachineLetter, w, u float64) {
	sort.Slice(leave, func(i, j int) bool { return leave[i] < leave[j] })
	r.acc.record(leave, w, u)
	b := make([]byte, len(leave))
	for i, ml := range leave {
		b[i] = byte(ml)
	}
	key := string(b)
	if n := len(r.foldAccs); n > 0 {
		// Keyed by leave, not by sample, so every draw of a leave lands in
		// the same fold and the complement model has never seen it.
		r.foldAccs[foldForKey(key, n)].record(leave, w, u)
	}
	ml := r.measured[key]
	if ml == nil {
		ml = &measuredLeave{round: r.currentRound}
		r.measured[key] = ml
	}
	ml.sumW += w
	ml.count++
	ml.sumU += u
}

// finalizePlacementPosterior turns the recorded samples into a complete
// posterior over every feasible leave: measured leaves keep their evaluated
// mean likelihood, unmeasured ones get a likelihood imputed from marginal
// lifts. No-op if nothing was recorded (Complete stays false and the simmer
// falls back to random racks).
func (r *RangeFinder) finalizePlacementPosterior() {
	if r.acc == nil || r.acc.n == 0 {
		return
	}
	res := imputeFullPosteriorTuned(r.inferenceBagMap, r.inference.RackLength, r.acc,
		r.foldAccs, r.measured, r.threads, r.tuning)
	r.imputeRes = res
	if len(res.racks) == 0 {
		return
	}
	r.inference.InferredRacks = res.racks
	r.inference.Complete = true
	log.Info().Int("measured-leaves", res.measuredLeaves).
		Int("imputed-leaves", res.imputedLeaves).
		Float64("measured-mass", res.measuredMass).
		Int("marginal-order", res.marginalOrder).
		Float64("log-calib", res.logCalib).
		Float64("log-calib-in-sample", res.logCalibInSample).
		Msg("imputed-complete-posterior")
}

func (r *RangeFinder) Infer(ctx context.Context) error {
	if !r.readyToInfer {
		return errors.New("not ready")
	}
	r.working = true
	inferStart := time.Now()
	defer func() {
		r.inferElapsed = time.Since(inferStart)
		r.working = false
		log.Info().Msg("inference engine quitting")
	}()

	// Exhaustive enumeration: when the leave space is small enough, visit every
	// distinct leave exactly once and apply full Bayesian weighting
	// (prior × likelihood) rather than importance sampling.
	// Exchange moves are excluded because their "leave" semantics differ.
	isExchange := r.lastOppMove != nil && r.lastOppMove.Action() == move.MoveTypeExchange
	if !isExchange && r.inference.RackLength >= 1 {
		maxLeaves := r.enumerationLimit()
		m := countMultisets(r.inferenceBagMap, r.inference.RackLength)
		if m <= maxLeaves {
			log.Info().Int("leaf-count", m).Int("rack-length", r.inference.RackLength).
				Msg("using-exhaustive-enumeration")
			return r.inferEnumerated(ctx)
		}
		log.Info().Int("leaf-count", m).Int("max-leaves", maxLeaves).
			Msg("leaf-space-too-large-using-sampling")
	}

	if !isExchange && r.inference.RackLength >= 1 {
		// Monte Carlo sampling path for tile placements: every sampled leave
		// (whatever its likelihood) feeds the marginal-lift accumulator so a
		// complete posterior can be imputed after sampling ends.
		r.initImputationState()
	}

	logChan := make(chan []byte)
	syncExitChan := make(chan bool, r.threads)
	logDone := make(chan bool)

	ctrl := errgroup.Group{}
	writer := errgroup.Group{}

	// Round 0 is blind, prior-sampled exploration. When refinement rounds
	// will follow, it only gets refineStage0Frac of the budget; the rest pays
	// for posterior-guided measurement.
	parentCtx := ctx
	rounds := 0
	if r.acc != nil {
		rounds = r.MaxRounds()
	}
	if rounds > 0 && r.budget <= 0 {
		if deadline, ok := parentCtx.Deadline(); ok {
			stage0End := time.Now().Add(
				time.Duration(float64(time.Until(deadline)) * refineStage0Frac))
			var cancelStage0 context.CancelFunc
			ctx, cancelStage0 = context.WithDeadline(parentCtx, stage0End)
			defer cancelStage0()
		}
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	ctrl.Go(func() error {
		defer func() {
			log.Debug().Msgf("Inference engine controller thread exiting")
		}()
		for range ctx.Done() {
		}
		log.Debug().Msgf("Context is done: %v", ctx.Err())
		for t := 0; t < r.threads; t++ {
			syncExitChan <- true
		}
		log.Debug().Msgf("Sent sync-exit messages to children threads...")

		return ctx.Err()
	})

	if r.logStream != nil {
		writer.Go(func() error {
			defer func() {
				log.Debug().Msgf("Writer routine exiting")
			}()
			for {
				select {
				case bytes := <-logChan:
					r.logStream.Write(bytes)
				case <-logDone:
					log.Debug().Msgf("Got quit signal...")
					return nil
				}
			}
		})
	}

	g := errgroup.Group{}
	var iterMutex sync.Mutex
	for t := 0; t < r.threads; t++ {
		t := t
		g.Go(func() error {
			defer func() {
				log.Debug().Msgf("Thread %v exiting inferrer", t)
			}()
			log.Debug().Msgf("Thread %v starting inferrer", t)
			stage0 := r.stage0Budget()
			for {
				iterMutex.Lock()
				r.iterationCount++
				iterNum := r.iterationCount
				iterMutex.Unlock()
				if stage0 > 0 && iterNum > stage0 {
					// Round 0 has spent its share. Stopping on a count rather
					// than a clock is what lets the same position produce the
					// same posterior twice.
					cancel()
					return nil
				}
				newRacks, err := r.inferSingle(t, iterNum, logChan)
				if err != nil {
					log.Err(err).Msg("infer-single-error")
					cancel()
				}
				if len(newRacks) > 0 {
					iterMutex.Lock()
					if r.acc != nil {
						// Tile-placement path: Weight carries the measured
						// likelihood; accumulate for imputation.
						for _, ir := range newRacks {
							// Round 0 draws from the prior, so u = 1.
							r.recordPlacementSample(ir.Leave, ir.Weight, 1)
							r.traceDraw(DrawRecord{Round: 0, U: 1, Mult: 1,
								Measured: ir.Weight}, ir.Leave)
						}
					} else {
						// Exchange path: keep distinct racks, first weight wins.
						for _, ir := range newRacks {
							key := leaveKey(ir.Leave)
							if _, exists := r.inference.seen[key]; !exists {
								r.inference.InferredRacks = append(r.inference.InferredRacks, ir)
								r.inference.seen[key] = len(r.inference.InferredRacks) - 1
							}
						}
					}
					iterMutex.Unlock()
				}
				select {
				case v := <-syncExitChan:
					log.Debug().Msgf("Thread %v got sync msg %v", t, v)
					return nil
				default:
					// Do nothing
				}
			}
		})
	}

	err := g.Wait()
	log.Debug().Msgf("errgroup returned err %v", err)

	if r.logStream != nil {
		close(logDone)
		writer.Wait()
	}

	ctrlErr := ctrl.Wait()
	log.Debug().Msgf("ctrl errgroup returned err %v", ctrlErr)

	// Round 0 has ended (typically by deadline); build the complete posterior
	// from what was measured. CPU-bound and fast, so it runs even though ctx
	// is already done.
	r.stage0Elapsed = time.Since(inferStart)
	r.stage0Sims = int(r.simCount.Load())
	r.measureForcedLeaves(parentCtx)
	r.finalizePlacementPosterior()

	// Then alternate measurement and imputation on the remaining budget,
	// drawing leaves from the posterior the model just produced.
	r.refineRounds(parentCtx, rounds)

	if ctrlErr == context.Canceled || ctrlErr == context.DeadlineExceeded {
		// Not actually an error
		log.Debug().AnErr("ctrlErr", ctrlErr).Msg("inferencer-it's ok, not an error")
		return nil
	}
	return ctrlErr

}

func (r *RangeFinder) inferSingle(thread, iterNum int, logChan chan []byte) ([]montecarlo.InferredRack, error) {
	g := r.gameCopies[thread]
	// Since we took back the last move, the player on turn should be our opponent
	// (the person whose rack we are inferring)
	opp := g.PlayerOnTurn()
	var extraDrawn []tilemapping.MachineLetter
	var err error
	isExchange := r.lastOppMove.Action() == move.MoveTypeExchange
	// otherwise, it's a tile placement play.

	if isExchange {
		return r.inferSingleExchange(thread, iterNum, logChan)
	}

	extraDrawn, err = g.SetRandomRack(opp, r.lastOppMoveRackTiles)
	if err != nil {
		return nil, err
	}
	// Copy the last opp move but set the leave to what would be the leave with
	// this new rack.
	lastOppMove := &move.Move{}
	lastOppMove.CopyFrom(r.lastOppMove)
	lastOppMove.SetLeave(extraDrawn)

	logIter := LogIteration{Iteration: iterNum, Thread: thread, Rack: g.RackLettersFor(opp)}
	if r.logStream != nil {
		r.aiplayers[thread].(*simplesimmer.SimpleSimmer).SetLogging(true)
	}

	logfilename, err := r.aiplayers[thread].(*simplesimmer.SimpleSimmer).GenAndSim(
		context.Background(), 10, lastOppMove)
	if err != nil {
		return nil, err
	}
	r.simCount.Add(1)

	bestPlays := r.aiplayers[thread].(*simplesimmer.SimpleSimmer).BestPlays().PlaysNoLock()
	if r.logStream != nil {
		logIter.TopMove = bestPlays[0].Move().ShortDescription()
		logIter.TopMoveWinProb = bestPlays[0].WinProb()
		logIter.SimLogFile = logfilename
	}

	// The returned Weight is the measured likelihood P(play | leave); the
	// prior enters later, when the full posterior is assembled (measured
	// leaves get prior × mean measured likelihood). Zero-likelihood samples
	// are returned too: they still count toward the containment-marginal
	// denominators used for imputation.
	likelihoodP, targetWinProb := softmaxLikelihood(bestPlays, lastOppMove, g.Board(), r.Tau())

	tiles := make([]tilemapping.MachineLetter, len(extraDrawn))
	copy(tiles, extraDrawn)

	if r.logStream != nil {
		logIter.InferredMoveWinProb = targetWinProb
		logIter.Likelihood = likelihoodP
		out, err := yaml.Marshal([]LogIteration{logIter})
		if err != nil {
			log.Err(err).Msg("marshalling log")
			return nil, err
		}
		logChan <- out
	}

	return []montecarlo.InferredRack{{Leave: tiles, Weight: likelihoodP}}, nil
}

func (r *RangeFinder) inferSingleExchange(thread, iterNum int, logChan chan []byte) ([]montecarlo.InferredRack, error) {
	g := r.gameCopies[thread]
	// Since we took back the last move, the player on turn should be our opponent
	// (the person whose rack we are inferring)
	opp := g.PlayerOnTurn()
	g.SetRandomRack(opp, nil)
	logIter := LogIteration{Iteration: iterNum, Thread: thread, Rack: g.RackLettersFor(opp)}

	// Only run the simmer if an exchange with the same number of tiles is found
	// in the static plays.
	numMoves := 15

	allMoves := r.aiplayers[thread].(*simplesimmer.SimpleSimmer).GenerateMoves(numMoves)
	exchangeCount := 0
	for i := range allMoves {
		if allMoves[i].Action() == move.MoveTypeExchange && allMoves[i].TilesPlayed() == r.lastOppMove.TilesPlayed() {
			exchangeCount++
		}
	}
	if exchangeCount < 1 {
		// Don't infer.
		return nil, nil
	}

	// Since we don't know what the opp actually exchanged, don't pass in their
	// specific exchange. The single exchange inferrer just looks for n-tile plays.
	if r.logStream != nil {
		r.aiplayers[thread].(*simplesimmer.SimpleSimmer).SetLogging(true)
	}

	logfilename, err := r.aiplayers[thread].(*simplesimmer.SimpleSimmer).GenAndSim(
		context.Background(), numMoves, nil)
	if err != nil {
		return nil, err
	}
	r.simCount.Add(1)

	bestPlays := r.aiplayers[thread].(*simplesimmer.SimpleSimmer).BestPlays().PlaysNoLock()
	if r.logStream != nil {
		logIter.TopMove = bestPlays[0].Move().ShortDescription()
		logIter.TopMoveWinProb = bestPlays[0].WinProb()
		logIter.SimLogFile = logfilename
	}

	var result []montecarlo.InferredRack
	for _, m := range bestPlays {
		if m.Move().TilesPlayed() != r.lastOppMove.TilesPlayed() ||
			m.Move().Action() != move.MoveTypeExchange {
			continue
		}

		// For exchange inference we use the full rack (all tiles) as the "leave"
		// since we don't know which specific tiles were exchanged — only the kept tiles.
		leave := m.Move().Leave()
		// SetRandomRack already samples from the prior; weight = likelihood only.
		likelihoodP, targetWinProb := softmaxLikelihood(bestPlays, m.Move(), g.Board(), r.Tau())
		bayesianWeight := likelihoodP

		if bayesianWeight <= 0 {
			continue
		}

		tiles := make([]tilemapping.MachineLetter, len(leave))
		copy(tiles, leave)

		if r.logStream != nil {
			logIter.InferredMoveWinProb = targetWinProb
			logIter.Likelihood = likelihoodP
			out, err := yaml.Marshal([]LogIteration{logIter})
			if err != nil {
				log.Err(err).Msg("marshalling log")
				return nil, err
			}
			logChan <- out
		}
		result = append(result, montecarlo.InferredRack{Leave: tiles, Weight: bayesianWeight})
	}
	return result, nil
}

func (r *RangeFinder) Inferences() *Inference {
	return r.inference
}

func (r *RangeFinder) Reset() {
	r.inference = NewInference()
	r.readyToInfer = false
}

func (r *RangeFinder) IsBusy() bool {
	return r.working
}

// MovesAreTheSame reports whether two moves are considered identical for
// likelihood purposes: exact equality (with transposition allowed on an empty
// board) or equivalent single-tile placements. Exported for harnesses that
// must match an observed move against simmed candidates with the same
// semantics the rangefinder uses.
func MovesAreTheSame(m1 *move.Move, m2 *move.Move, g *board.GameBoard) bool {
	return movesAreTheSame(m1, m2, g)
}

func movesAreTheSame(m1 *move.Move, m2 *move.Move, g *board.GameBoard) bool {
	checkTransposition := g.IsEmpty()
	if m1.Equals(m2, checkTransposition, true) {
		return true
	}

	// Otherwise check if it's a single-tile move.
	if m1.TilesPlayed() == 1 && m2.TilesPlayed() == 1 &&
		uniqueSingleTileKey(m1) == uniqueSingleTileKey(m2) {
		return true
	}
	return false
}

func movesAreKindaTheSame(m1 *move.Move, m2 *move.Move, m2tiles []tilemapping.MachineLetter,
	g *board.GameBoard) bool {
	// This is a bit of a fuzzy equality function. We want to see if two
	// tile-play moves are "materially" the same.
	// If they're tile play moves, and they use the same tiles, we will
	// call them the same, even if the plays were in different places.
	// This is because the person we're inferring for may have missed
	// a play using the same tiles in a better spot.

	if movesAreTheSame(m1, m2, g) {
		return true
	}

	// Otherwise, check if they use the same tiles.
	m1tiles := []tilemapping.MachineLetter{}
	for _, t := range m1.Tiles() {
		if t == 0 {
			continue
		}
		ml := t.IntrinsicTileIdx()
		m1tiles = append(m1tiles, ml)
	}
	if len(m1tiles) != len(m2tiles) {
		return false
	}
	sort.Slice(m1tiles, func(i, j int) bool { return m1tiles[i] < m1tiles[j] })
	// m2tiles is already sorted.
	allEqual := true
	for i := range m1tiles {
		// Take into account the blank. If the player played a blank and
		// missed a better play with a blank being another letter, it's still
		// "the same play". This is handled by the calls to "IntrinsicTileIdx" above.
		if m1tiles[i] != m2tiles[i] {
			allEqual = false
			break
		}
	}
	return allEqual
}

func uniqueSingleTileKey(m *move.Move) int {
	// Find the tile.
	var idx int
	var tile tilemapping.MachineLetter
	for idx, tile = range m.Tiles() {
		if tile != 0 {
			break
		}
	}
	row, col, vert := m.CoordsAndVertical()
	// We want to get the coordinate of the tile that is on the board itself.
	if vert {
		row += idx
	} else {
		col += idx
	}
	// A unique, fast to compute key for this play.
	return row + tilemapping.MaxAlphabetSize*col +
		tilemapping.MaxAlphabetSize*tilemapping.MaxAlphabetSize*int(tile)
}

// SetForcedLeaves names leaves that must get a mini-sim, whether or not the
// proposal would ever have drawn them. They are measured once round 0 is done,
// so the first imputation model and every round after it are built knowing what
// those leaves are really worth.
//
// This is a diagnostic, not a setting: naming the leave the opponent actually
// held means using the answer, which no bot can do. It exists to separate two
// explanations of a bad read that look identical from the outside -- the model
// scored the true leave badly, or the model never looked at it. Forcing it in
// and seeing the read stay wrong rules the second one out.
//
// The forced measurements enter as ordinary prior-weight draws, which does
// slightly flatter the calibration constant: they are evidence the sampler did
// not pay for. The bias is small next to what the comparison is measuring.
func (r *RangeFinder) SetForcedLeaves(leaves [][]tilemapping.MachineLetter) {
	r.forcedLeaves = leaves
}

// measureForcedLeaves evaluates whatever SetForcedLeaves named, skipping any
// the sampler already happened to measure.
func (r *RangeFinder) measureForcedLeaves(ctx context.Context) {
	if len(r.forcedLeaves) == 0 || r.acc == nil {
		return
	}
	var todo [][]tilemapping.MachineLetter
	for _, leave := range r.forcedLeaves {
		sorted := make([]tilemapping.MachineLetter, len(leave))
		copy(sorted, leave)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		if ml, ok := r.measured[leaveKey(sorted)]; ok && ml.count > 0 {
			continue
		}
		todo = append(todo, sorted)
	}
	if len(todo) == 0 {
		return
	}
	var mu sync.Mutex
	err := r.evaluateLeaves(ctx, todo, func(leave []tilemapping.MachineLetter, lik float64) {
		mu.Lock()
		defer mu.Unlock()
		r.recordPlacementSample(leave, lik, 1)
		r.traceDraw(DrawRecord{Round: 0, U: 1, Mult: 1, Measured: lik}, leave)
	})
	if err != nil {
		log.Err(err).Msg("forced-leaf-evaluate-failed")
	}
}

// SetImputationLambda sets the shrinkage pseudo-count used when combining
// sub-leave marginals: a term estimated from c samples is scaled by c/(c+λ), so
// a larger λ pulls thin terms harder toward no effect. Pass 0 for the default.
func (r *RangeFinder) SetImputationLambda(l float64) { r.tuning.lambda = l }

// SetCalibrationShrink sets how far the imputation's calibration constant moves
// from its in-sample fit toward the cross-fitted one. 1 is the engine's
// behavior, 0 keeps the in-sample constant, and values between interpolate.
//
// The constant scales every imputed leave alike, so it decides how imputed
// leaves weigh against measured ones without changing their order among
// themselves -- and so barely touches which leaves the proposal draws.
func (r *RangeFinder) SetCalibrationShrink(s float64) {
	r.tuning.calibShrink = s
	r.tuning.calibSet = true
}

// SetMaxMarginalOrder caps the sub-leave expansion the imputation uses. The
// engine's rule is ceil(k/2) capped at 3 for a k-tile leave; raising it to 4
// lets the model carry four-way interactions, which is where a six-tile
// leave's bingo structure lives. Pass 0 for the engine's rule.
func (r *RangeFinder) SetMaxMarginalOrder(m int) { r.tuning.maxOrder = m }

// SetValueTerm pins how the static leave value enters the imputation, for
// every leave length; see ValueMode. Unpinned, the engine uses ValueFirst for
// leaves of five tiles and up and nothing below, which is where the idea
// holds: a player who lays down one or two tiles has given up points to keep
// the rest, so the leave they kept is likely a strong one, and the leave
// table already knows which leaves are strong.
func (r *RangeFinder) SetValueTerm(m ValueMode) {
	r.tuning.valueMode = m
	r.tuning.valueSet = true
}

// leaveValueFunc returns a function scoring a leave (as tile runs) with the
// static leave value, or nil when no calculator here can.
func (r *RangeFinder) leaveValueFunc() func([]tileRun) float64 {
	type valuer interface {
		LeaveValue(leave tilemapping.MachineWord) float64
	}
	for _, c := range r.equityCalculators {
		lv, ok := c.(valuer)
		if !ok {
			continue
		}
		return func(runs []tileRun) float64 {
			var buf [8]tilemapping.MachineLetter
			n := 0
			for _, ru := range runs {
				for k := 0; k < ru.c && n < len(buf); k++ {
					buf[n] = ru.t
					n++
				}
			}
			return lv.LeaveValue(tilemapping.MachineWord(buf[:n]))
		}
	}
	return nil
}

// ImputeStats reports the last imputation's calibration and value-term
// figures without needing a trace: the cross-fitted and in-sample calibration
// constants, the share of posterior mass on measured leaves, and the value
// slope. ok is false when nothing has been imputed.
func (r *RangeFinder) ImputeStats() (logCalib, logCalibInSample, measuredMass, valueBeta float64, ok bool) {
	res := r.imputeRes
	if res == nil {
		return 0, 0, 0, 0, false
	}
	return res.logCalib, res.logCalibInSample, res.measuredMass, res.valueBeta, true
}
