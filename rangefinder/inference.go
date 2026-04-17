package rangefinder

import (
	"context"
	"errors"
	"io"
	"math"
	"runtime"
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
	// higher values allow more weight for sub-optimal plays.
	// Softmax is applied over log-odds of win probabilities, so tau is on the
	// log-odds scale. Typical positions (20%-80% win prob) span roughly [-1.4, 1.4];
	// strongly won/lost positions (5%-95%) reach about [-3, 3].
	SoftmaxTemperature = 0.1

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
	seen          map[string]int // leaveKey -> index in InferredRacks
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

// logit converts a win probability to log-odds: ln(p / (1-p)).
// p is clamped to [logitEps, 1-logitEps] to avoid ±Inf.
func logit(p float64) float64 {
	if p < logitEps {
		p = logitEps
	} else if p > 1-logitEps {
		p = 1 - logitEps
	}
	return math.Log(p / (1 - p))
}

// softmaxLikelihood computes P(targetMove | leave) as a softmax over the
// log-odds of win probabilities of all simmed plays. Using log-odds undoes
// the implicit sigmoid in win probabilities, giving softmax unbounded inputs
// it is designed for. Returns (likelihood, targetWinProb); likelihood is 0
// if the target move is not found among the plays.
func softmaxLikelihood(plays []*montecarlo.SimmedPlay, targetMove *move.Move, b *board.GameBoard, tau float64) (float64, float64) {
	if len(plays) == 0 {
		return 0, 0
	}

	targetWinProb := math.NaN()
	maxLogOdds := math.Inf(-1)
	for _, sp := range plays {
		lo := logit(sp.WinProb())
		if lo > maxLogOdds {
			maxLogOdds = lo
		}
		if movesAreTheSame(sp.Move(), targetMove, b) {
			targetWinProb = sp.WinProb()
		}
	}
	if math.IsNaN(targetWinProb) {
		return 0, 0
	}

	// Softmax over log-odds with numerical stability (subtract max before exp).
	sum := 0.0
	for _, sp := range plays {
		sum += math.Exp((logit(sp.WinProb()) - maxLogOdds) / tau)
	}
	return math.Exp((logit(targetWinProb)-maxLogOdds)/tau) / sum, targetWinProb
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
	// simIters is the max mini-sim iterations per rack candidate.
	// 0 means use the SimpleSimmer default (200).
	simIters int
	// maxEnumeratedLeaves is the threshold for switching from Monte Carlo sampling
	// to exhaustive enumeration. When the number of distinct leaves drawable from
	// the bag (countMultisets) is ≤ this value, inferEnumerated is used instead of
	// inferSingle. 0 means use DefaultMaxEnumeratedLeaves.
	maxEnumeratedLeaves int

	working      bool
	readyToInfer bool

	inferenceBagMap []uint8
	cfg             *config.Config
	lastOppMove     *move.Move
	// tiles used by the last opponent's move, from their rack:
	lastOppMoveRackTiles []tilemapping.MachineLetter
	inference            *Inference

	logStream io.Writer
}

func (r *RangeFinder) Init(game *game.Game, eqCalcs []equity.EquityCalculator,
	cfg *config.Config) {

	r.origGame = game
	r.equityCalculators = eqCalcs
	r.threads = max(1, runtime.NumCPU())
	r.cfg = cfg
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

func (r *RangeFinder) Tau() float64 {
	if r.tau == 0 {
		return SoftmaxTemperature
	}
	return r.tau
}

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
	return nil
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
		maxLeaves := r.maxEnumeratedLeaves
		if maxLeaves == 0 {
			maxLeaves = DefaultMaxEnumeratedLeaves
		}
		m := countMultisets(r.inferenceBagMap, r.inference.RackLength)
		if m <= maxLeaves {
			log.Info().Int("leaf-count", m).Int("rack-length", r.inference.RackLength).
				Msg("using-exhaustive-enumeration")
			return r.inferEnumerated(ctx)
		}
		log.Info().Int("leaf-count", m).Int("max-leaves", maxLeaves).
			Msg("leaf-space-too-large-using-sampling")
	}

	logChan := make(chan []byte)
	syncExitChan := make(chan bool, r.threads)
	logDone := make(chan bool)

	ctrl := errgroup.Group{}
	writer := errgroup.Group{}

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
			for {
				iterMutex.Lock()
				r.iterationCount++
				iterNum := r.iterationCount
				iterMutex.Unlock()
				newRacks, err := r.inferSingle(t, iterNum, logChan)
				if err != nil {
					log.Err(err).Msg("infer-single-error")
					cancel()
				}
				if len(newRacks) > 0 {
					iterMutex.Lock()
					for _, ir := range newRacks {
						key := leaveKey(ir.Leave)
						if _, exists := r.inference.seen[key]; !exists {
							r.inference.InferredRacks = append(r.inference.InferredRacks, ir)
							r.inference.seen[key] = len(r.inference.InferredRacks) - 1
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

	// SetRandomRack already samples racks from the hypergeometric (prior)
	// distribution, so the IS weight is only the likelihood P(play | leave).
	// Multiplying by the prior again would double-count it.
	likelihoodP, targetWinProb := softmaxLikelihood(bestPlays, lastOppMove, g.Board(), r.Tau())
	bayesianWeight := likelihoodP

	if bayesianWeight <= 0 {
		return nil, nil
	}

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

	return []montecarlo.InferredRack{{Leave: tiles, Weight: bayesianWeight}}, nil
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
