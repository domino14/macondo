package negamax

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"sort"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"
	"lukechampine.com/frand"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/tinymove/conversions"
	"github.com/domino14/word-golib/tilemapping"
)

// thanks Wikipedia:
/*
function negamax(node, depth, α, β, color) is
    if depth = 0 or node is a terminal node then
        return color × the heuristic value of node

    childNodes := generateMoves(node)
    childNodes := orderMoves(childNodes)
    value := −∞
    foreach child in childNodes do
        value := max(value, −negamax(child, depth − 1, −β, −α, −color))
        α := max(α, value)
        if α ≥ β then
            break (* cut-off *)
    return value
(* Initial call for Player A's root node *)
negamax(rootNode, depth, −∞, +∞, 1)
**/

const HugeNumber = int16(32000)
const MaxVariantLength = 25

// Parallel algorithm constants
const (
	ParallelAlgoAuto      = "auto"
	ParallelAlgoLazySMP   = "lazysmp"
	ParallelAlgoABDADA    = "abdada"
	ParallelAlgoTreeSplit = "treesplit"
)

// Bitflags for move estimates.
const (
	EarlyPassBF = 1 << 13
	HashMoveBF  = 1 << 12
	GoingOutBF  = 1 << 11
	// Moves that score more than 256 pts in the endgame may have some
	// sorting issues. Can probably fix this later.
	TilesPlayedBFOffset = 8
)

var (
	ErrNoEndgameSolution = errors.New("no endgame solution found")
)

// Credit: MIT-licensed https://github.com/algerbrex/blunder/blob/main/engine/search.go
type PVLine struct {
	Moves    [MaxVariantLength]*move.Move
	g        *game.Game
	score    int16
	numMoves int
}

// Clear the principal variation line.
func (pvLine *PVLine) Clear() {
	pvLine.numMoves = 0
}

// Update the principal variation line with a new best move,
// and a new line of best play after the best move.
func (pvLine *PVLine) Update(m *move.Move, newPVLine PVLine, score int16) {
	pvLine.Clear()
	mc := &move.Move{}
	mc.CopyFrom(m)
	pvLine.Moves[0] = mc
	for i := 0; i < newPVLine.numMoves; i++ {
		pvLine.Moves[i+1] = newPVLine.Moves[i]
	}
	pvLine.numMoves = newPVLine.numMoves + 1
	pvLine.score = score
}

// Get the best move from the principal variation line.
func (pvLine *PVLine) GetPVMove() *move.Move {
	return pvLine.Moves[0]
}

// Convert the principal variation line to a string.
func (pvLine PVLine) String() string {
	var s string
	s = fmt.Sprintf("PV; val %d\n", pvLine.score)
	for i := 0; i < pvLine.numMoves; i++ {
		s += fmt.Sprintf("%d: %s (%d)\n",
			i+1,
			pvLine.Moves[i].ShortDescription(),
			pvLine.Moves[i].Score())
	}
	return s
}

func (pvLine PVLine) NLBString() string {
	// no line breaks
	var s string
	s = fmt.Sprintf("[Value: %+d] ", pvLine.score)
	for i := 0; i < pvLine.numMoves; i++ {
		finalSeparator := "|"
		if i == pvLine.numMoves-1 {
			finalSeparator = ""
		}
		s += fmt.Sprintf("%d) %s (%d) %s ",
			i+1,
			// XXX: this will only work if we are playing the moves and keeping
			// track of the playthrough
			// pvLine.g.Board().MoveDescriptionWithPlaythrough(pvLine.Moves[i]),
			pvLine.Moves[i].ShortDescription(),
			pvLine.Moves[i].Score(),
			finalSeparator,
		)
	}
	return s
}

// panic if pvline is invalid
// func (p PVLine) verify() {
// 	g := p.g.Copy()
// 	onturn := g.PlayerOnTurn()
// 	initSpread := g.SpreadFor(onturn)
// 	for i := 0; i < len(p.Moves); i++ {
// 		mc := &move.Move{}
// 		p.Moves[i].CopyToMove(mc)
// 		_, err := g.ValidateMove(mc)
// 		if err != nil {
// 			fmt.Println("error with pv", p)
// 			panic(err)
// 		}
// 		err = g.PlayMove(mc, false, 0)
// 		if err != nil {
// 			panic(err)
// 		}
// 	}
// 	// If the scores don't match, log a warning. This can be because
// 	// the transposition table cut off the PV.
// 	if g.SpreadFor(onturn)-initSpread != int(p.score) {
// 		log.Warn().
// 			Int("initSpread", initSpread).
// 			Int("nowSpread", g.SpreadFor(onturn)).
// 			Int("diffInSpreads", g.SpreadFor(onturn)-initSpread).
// 			Int16("expectedDiff", p.score).
// 			Msg("pv-cutoff-spreads-do-not-match")
// 	}
// }

type endgameLog struct {
	Ply   int       `yaml:"ply"`
	Plays []playLog `yaml:"plays"`
}

type playLog struct {
	Play      string    `yaml:"play"`
	Value     int16     `yaml:"value"`
	Alpha     int16     `yaml:"α"`
	Beta      int16     `yaml:"β"`
	TTNodeKey uint64    `yaml:"ttNodeKey"`
	Flag      uint8     `yaml:"ttFlag"`
	Depth     uint8     `yaml:"ttDepth"`
	Score     int16     `yaml:"ttScore"`
	Plays     []playLog `yaml:"plays"`
}

type Solver struct {
	stmMovegen   movegen.MoveGenerator
	game         *game.Game
	initialMoves [][]tinymove.SmallMove

	gameCopies []*game.Game
	movegens   []movegen.MoveGenerator

	initialSpread int
	solvingPlayer int // This is the player who we call this function for.

	iterativeDeepeningOptim bool
	firstWinOptim           bool
	nullWindowOptim         bool
	transpositionTableOptim bool
	negascoutOptim          bool
	// lazySMP is a way to optimize the endgame speed, but it doesn't work
	// super great with Scrabble
	lazySMPOptim bool
	// ABDADA is another alpha-beta parallelization algorithm
	abdadaOptim bool
	// treeSplit distributes root moves across threads with work-stealing
	treeSplitOptim bool

	// solveMultipleVariations will solve multiple variations in parallel.
	solveMultipleVariations int
	principalVariation      PVLine
	bestPVValue             int16
	preventSlowroll         bool

	ttable      *TranspositionTable
	abdadaTable *ABDADATable

	currentIDDepths []int
	requestedPlies  int
	threads         int
	nodes           atomic.Uint64

	logStream  io.Writer
	busy       bool
	threadLogs []playLog

	variations []PVLine

	// Metrics from last solve
	lastSolveTime   float64
	lastTTableStats string
}

// Init initializes the solver
func (s *Solver) Init(m movegen.MoveGenerator, game *game.Game) error {
	s.ttable = GlobalTranspositionTable
	s.stmMovegen = m
	s.game = game

	s.firstWinOptim = false
	s.transpositionTableOptim = true
	s.iterativeDeepeningOptim = true
	s.negascoutOptim = true
	s.threads = max(1, runtime.NumCPU())
	if s.stmMovegen != nil {
		s.stmMovegen.SetGenPass(true)
		s.stmMovegen.SetPlayRecorder(movegen.AllPlaysSmallRecorder)
	}

	return nil
}

// SetLogStream only prints coherent logs for single-threaded endgames for now.
func (s *Solver) SetLogStream(l io.Writer) {
	s.logStream = l
}

func (s *Solver) Movegen() movegen.MoveGenerator {
	return s.stmMovegen
}

func (s *Solver) Variations() []PVLine {
	return s.variations
}

func (s *Solver) SetThreads(threads int) {
	s.threads = max(1, threads)
	// Auto-enable parallel algorithm selection when using multiple threads
	if threads >= 2 && !s.lazySMPOptim && !s.abdadaOptim && !s.treeSplitOptim {
		// Default to "auto" mode - will be resolved based on root move count
		// No need to set any flag - the dispatcher will handle it
	}
}

func (s *Solver) SetLazySMP(enable bool) {
	s.lazySMPOptim = enable
	if enable {
		s.treeSplitOptim = false
		s.abdadaOptim = false
	}
}

func (s *Solver) SetTreeSplit(enable bool) {
	s.treeSplitOptim = enable
	if enable {
		s.lazySMPOptim = false
		s.abdadaOptim = false
	}
}

func (s *Solver) SetABDADA(enable bool) {
	s.abdadaOptim = enable
	if enable {
		s.lazySMPOptim = false
		s.treeSplitOptim = false
		if s.abdadaTable == nil {
			s.abdadaTable = NewABDADATable()
		}
	}
}

// SetParallelAlgorithm sets the parallelization algorithm to use.
// Options: ParallelAlgoAuto, ParallelAlgoLazySMP, ParallelAlgoABDADA, ParallelAlgoTreeSplit
// ParallelAlgoAuto uses a probe-based heuristic:
//   - < 50 root moves: LazySMP (not enough parallel work for ABDADA)
//   - > 500 root moves: ABDADA (high branching factor)
//   - Otherwise: runs a 4-ply probe (max 2s) to measure EBF (effective branching factor)
//   - EBF >= 17: ABDADA (bushy tree benefits from ABDADA's work distribution)
//   - EBF < 12: LazySMP (narrow tree, TT sharing works well)
//   - Default: LazySMP (simpler, lower overhead)
func (s *Solver) SetParallelAlgorithm(algo string) error {
	switch algo {
	case ParallelAlgoAuto:
		// Auto mode: will be resolved via probe in iterativelyDeepen
		s.lazySMPOptim = false
		s.abdadaOptim = false
		s.treeSplitOptim = false
	case ParallelAlgoLazySMP:
		s.SetLazySMP(true)
	case ParallelAlgoABDADA:
		s.SetABDADA(true)
	case ParallelAlgoTreeSplit:
		s.SetTreeSplit(true)
	default:
		return fmt.Errorf("unknown parallel algorithm: %s (options: %s, %s, %s, %s)",
			algo, ParallelAlgoAuto, ParallelAlgoLazySMP, ParallelAlgoABDADA, ParallelAlgoTreeSplit)
	}
	return nil
}

func (s *Solver) SetSolveMultipleVariations(i int) {
	s.solveMultipleVariations = i
}

func (s *Solver) makeGameCopies() error {
	log.Debug().Int("threads", s.threads).Msg("makeGameCopies")
	s.gameCopies = []*game.Game{}
	for i := 0; i < s.threads-1; i++ {
		s.gameCopies = append(s.gameCopies, s.game.Copy())
		s.gameCopies[i].SetBackupMode(game.SimulationMode)
	}
	s.movegens = []movegen.MoveGenerator{}
	gaddag := s.stmMovegen.(*movegen.GordonGenerator).GADDAG()
	for i := 0; i < s.threads-1; i++ {
		mg := movegen.NewGordonGenerator(gaddag, s.gameCopies[i].Board(), s.gameCopies[i].Bag().LetterDistribution())
		mg.SetSortingParameter(movegen.SortByNone)
		mg.SetGenPass(true)
		mg.SetPlayRecorder(movegen.AllPlaysSmallRecorder)
		s.movegens = append(s.movegens, mg)
	}
	return nil
}

func (s *Solver) generateSTMPlays(depth, thread int) []tinymove.SmallMove {
	// STM means side-to-move
	g := s.game
	mg := s.stmMovegen

	if thread > 0 {
		mg = s.movegens[thread-1]
		g = s.gameCopies[thread-1]
	}
	var genPlays []tinymove.SmallMove

	stmRack := g.RackFor(g.PlayerOnTurn())
	if s.currentIDDepths[thread] == depth {
		genPlays = s.initialMoves[thread]
	} else {
		mg.GenAll(stmRack, false)
		plays := mg.SmallPlays()
		// movegen owns the plays array. Make a copy of these.
		genPlays = make([]tinymove.SmallMove, len(plays))
		copy(genPlays, plays)
		movegen.SmallPlaySlicePool.Put(&plays)
	}
	return genPlays
}

func (s *Solver) assignEstimates(moves []tinymove.SmallMove, depth, thread int, ttMove tinymove.TinyMove) {
	g := s.game

	if thread > 0 {
		g = s.gameCopies[thread-1]
	}

	stmRack := g.RackFor(g.PlayerOnTurn())
	pnot := (g.PlayerOnTurn() + 1) % g.NumPlayers()
	otherRack := g.RackFor(pnot)
	numTilesOnRack := int(stmRack.NumTiles())
	ld := g.Bag().LetterDistribution()

	lastMoveWasPass := g.ScorelessTurns() > g.LastScorelessTurns()
	// var pvMove tinymove.TinyMove

	// if depth > 0 && s.principalVariation.Moves[depth] != nil {
	// 	pvMove = conversions.MoveToTinyMove(s.principalVariation.Moves[depth])
	// }

	for idx := range moves {
		if moves[idx].TilesPlayed() == numTilesOnRack {
			moves[idx].SetEstimatedValue(int16(moves[idx].Score()+2*otherRack.ScoreOn(ld)) + GoingOutBF)
			// } else if thread == 4 {
			// 	// Some handwavy LazySMP thing.
			// 	p.SetEstimatedValue(int16(7 - p.TilesPlayed()))
			// } else if depth > 2 {
			// 	moves[idx].SetEstimatedValue(int16(moves[idx].Score() - 5*moves[idx].TilesPlayed()))
			// } else {
			// 	moves[idx].SetEstimatedValue(int16(moves[idx].Score()))
			// }
		} else if depth > 2 {
			if thread >= 6 {
				// add some more jitter for lazysmp
				moves[idx].SetEstimatedValue(int16(moves[idx].Score() + 3*moves[idx].TilesPlayed()))
			} else {
				moves[idx].SetEstimatedValue(int16(moves[idx].Score() - 5*moves[idx].TilesPlayed()))
			}
		} else {
			moves[idx].SetEstimatedValue(int16(moves[idx].Score()))
		}

		// XXX: should also verify validity of ttMove later.
		if moves[idx].TinyMove() == ttMove {
			moves[idx].AddEstimatedValue(HashMoveBF)
		}

		if lastMoveWasPass && moves[idx].IsPass() {
			moves[idx].AddEstimatedValue(EarlyPassBF)
		}
	}
	sort.Slice(moves, func(i int, j int) bool {
		return moves[i].EstimatedValue() > moves[j].EstimatedValue()
	})
}

func (s *Solver) iterativelyDeepenLazySMP(ctx context.Context, plies int) error {
	// Generate first layer of moves.
	if plies < 2 {
		return errors.New("use at least 2 plies")
	}
	s.makeGameCopies()
	log.Info().Int("threads", s.threads).Msg("using-lazy-smp")
	s.currentIDDepths = make([]int, s.threads)
	initialHashKey := s.ttable.Zobrist().Hash(
		s.game.Board().GetSquares(),
		s.game.RackFor(s.solvingPlayer),
		s.game.RackFor(1-s.solvingPlayer),
		false, s.game.ScorelessTurns(),
	)

	α := -HugeNumber
	β := HugeNumber
	if s.firstWinOptim {
		// Search a very small window centered around 0. We're just trying
		// to find something that surpasses it.
		α = -1
		β = 1
	} else if s.nullWindowOptim {
		α = -1
		β = 0
	}

	// Generate first layer of moves (unless already set by wrapper for multiple variations).
	movesAlreadyGenerated := s.initialMoves != nil && len(s.initialMoves) > 0 && s.initialMoves[0] != nil && len(s.initialMoves[0]) > 0

	var lastIteration int16
	if !movesAlreadyGenerated {
		s.currentIDDepths[0] = -1 // so that generateSTMPlays generates all moves first properly.
		s.initialMoves = make([][]tinymove.SmallMove, s.threads)
		s.initialMoves[0] = s.generateSTMPlays(0, 0)
		// assignEstimates for the very first time around.
		s.assignEstimates(s.initialMoves[0], 0, 0, tinymove.InvalidTinyMove)

		pv := PVLine{g: s.game}
		// Do initial search so that we can have a good estimate for
		// move ordering.
		s.currentIDDepths[0] = 1
		lastIteration, _ = s.negamax(ctx, initialHashKey, 1, α, β, &pv, 0, true)
		// Sort the moves by valuation.
		sort.Slice(s.initialMoves[0], func(i, j int) bool {
			return s.initialMoves[0][i].EstimatedValue() > s.initialMoves[0][j].EstimatedValue()
		})
	} else {
		// Moves already set and sorted by wrapper, just ensure array is sized for threads
		if len(s.initialMoves) < s.threads {
			newInitialMoves := make([][]tinymove.SmallMove, s.threads)
			newInitialMoves[0] = s.initialMoves[0]
			s.initialMoves = newInitialMoves
		}
		// Use a default lastIteration value
		lastIteration = 0
	}

	// copy these moves to per-thread subarrays. This will also copy
	// the initial estimated valuation and order.
	for t := 1; t < s.threads; t++ {
		s.initialMoves[t] = make([]tinymove.SmallMove, len(s.initialMoves[0]))
		copy(s.initialMoves[t], s.initialMoves[0])
	}

	var lastPlyWinner *move.Move
	var lastPlyWinnerMatches int
	for p := 2; p <= plies; p++ {

		log.Info().Int("plies", p).Msg("deepening-iteratively")
		err := s.lazySMP(ctx, &lastIteration, p, initialHashKey)
		if err != nil {
			// could be a timeout, most likely.
			log.Err(err).Msg("lazySMP-possible-error")
			return err
		}
		justWon := s.principalVariation.Moves[0]

		if s.preventSlowroll {
			if lastPlyWinner != nil {
				if lastPlyWinner.ShortDescription() == justWon.ShortDescription() {
					lastPlyWinnerMatches++
					if lastPlyWinnerMatches > 4 {
						log.Info().Msg("preventing slowroll; exiting iterative deepening early")
						break
					} else {
						finalSpread := s.principalVariation.score + int16(s.game.CurrentSpread())
						if finalSpread > 0 && len(lastPlyWinner.Leave()) == 0 {
							// This move goes out, and it is a sure win.
							// So don't slowroll and just play it quickly.
							log.Info().Int16("final-spread", finalSpread).
								Msg("preventing slowroll on sure win")
							break
						}
					}
				} else {
					// they don't match.
					lastPlyWinnerMatches = 0
				}
			}
		}
		lastPlyWinner = justWon
	}

	s.bestPVValue = s.principalVariation.score
	return nil
}

func (s *Solver) iterativelyDeepenABDADA(ctx context.Context, plies int) error {
	// ABDADA (Alpha-Beta Distributed Asynchronous Distributed Asynchronous)
	// https://dl.acm.org/doi/pdf/10.1145/228329.228345
	if plies < 2 {
		return errors.New("use at least 2 plies")
	}
	s.makeGameCopies()
	log.Info().Int("threads", s.threads).Msg("using-abdada")
	s.currentIDDepths = make([]int, s.threads)
	initialHashKey := s.ttable.Zobrist().Hash(
		s.game.Board().GetSquares(),
		s.game.RackFor(s.solvingPlayer),
		s.game.RackFor(1-s.solvingPlayer),
		false, s.game.ScorelessTurns(),
	)

	α := -HugeNumber
	β := HugeNumber
	if s.firstWinOptim {
		α = -1
		β = 1
	} else if s.nullWindowOptim {
		α = -1
		β = 0
	}

	// Generate first layer of moves (unless already set by wrapper for multiple variations)
	movesAlreadyGenerated := s.initialMoves != nil && len(s.initialMoves) > 0 && s.initialMoves[0] != nil && len(s.initialMoves[0]) > 0

	var lastIteration int16
	if !movesAlreadyGenerated {
		s.currentIDDepths[0] = -1
		s.initialMoves = make([][]tinymove.SmallMove, s.threads)
		s.initialMoves[0] = s.generateSTMPlays(0, 0)
		s.assignEstimates(s.initialMoves[0], 0, 0, tinymove.InvalidTinyMove)

		// Do initial 1-ply search for move ordering
		pv := PVLine{g: s.game}
		s.currentIDDepths[0] = 1
		lastIteration, _ = s.negamax(ctx, initialHashKey, 1, α, β, &pv, 0, true)
		sort.Slice(s.initialMoves[0], func(i, j int) bool {
			return s.initialMoves[0][i].EstimatedValue() > s.initialMoves[0][j].EstimatedValue()
		})
	} else {
		// Moves already set and sorted by wrapper, just ensure array is sized for threads
		if len(s.initialMoves) < s.threads {
			newInitialMoves := make([][]tinymove.SmallMove, s.threads)
			newInitialMoves[0] = s.initialMoves[0]
			s.initialMoves = newInitialMoves
		}
		// Use a default lastIteration value
		lastIteration = 0
	}

	// Copy moves to all threads
	for t := 1; t < s.threads; t++ {
		s.initialMoves[t] = make([]tinymove.SmallMove, len(s.initialMoves[0]))
		copy(s.initialMoves[t], s.initialMoves[0])
	}

	// Iterative deepening loop
	for p := 2; p <= plies; p++ {
		log.Info().Int("plies", p).Msg("deepening-iteratively-abdada")

		// Set current depth for all threads
		for t := 0; t < s.threads; t++ {
			s.currentIDDepths[t] = p
		}

		// Aspiration search
		window := int16(8)
		aspα := lastIteration - window
		aspβ := lastIteration + window
		if s.firstWinOptim {
			aspα = -1
			aspβ = 1
		} else if s.nullWindowOptim {
			aspα = -1
			aspβ = 0
		}

	aspirationLoop:
		for {
			aspα = max(-HugeNumber, aspα)
			aspβ = min(HugeNumber, aspβ)
			log.Info().Int16("α", aspα).Int16("β", aspβ).Msg("abdada-search")

			// Start helper threads
			g := errgroup.Group{}
			cancels := make([]context.CancelFunc, s.threads-1)

			for t := 1; t < s.threads; t++ {
				helperCtx, cancel := context.WithCancel(ctx)
				cancels[t-1] = cancel
				threadID := t

				g.Go(func() error {
					pv := PVLine{g: s.gameCopies[threadID-1]}
					_, err := s.negamax(helperCtx, initialHashKey, p, aspα, aspβ, &pv, threadID, true)
					return err
				})
			}

			// Main thread search
			mainPV := PVLine{g: s.game}
			val, err := s.negamax(ctx, initialHashKey, p, aspα, aspβ, &mainPV, 0, true)

			// Stop helper threads
			for _, c := range cancels {
				if c != nil {
					c()
				}
			}
			g.Wait()

			if err != nil {
				log.Err(err).Msg("abdada-error")
				return err
			}

			sort.Slice(s.initialMoves[0], func(i, j int) bool {
				return s.initialMoves[0][i].EstimatedValue() > s.initialMoves[0][j].EstimatedValue()
			})

			s.principalVariation = mainPV
			s.bestPVValue = val - int16(s.initialSpread)
			log.Info().Int16("spread", val).Int("ply", p).Str("pv", mainPV.NLBString()).Msg("abdada-best-val")

			// Check if we're within the aspiration window
			if val > aspα && val < aspβ {
				lastIteration = val
				break aspirationLoop
			}

			// Widen window
			window *= 2
			if val <= aspα {
				for aspα-window < -HugeNumber {
					window /= 2
				}
				aspα -= window
				aspβ -= window / 3
			} else {
				for aspβ+window > HugeNumber {
					window /= 2
				}
				aspβ += window
				aspα += window / 3
			}
			log.Debug().Int16("new-α", aspα).Int16("new-β", aspβ).Msg("abdada-re-iterating")
		}
	}
	return nil
}

func (s *Solver) iterativelyDeepenTreeSplit(ctx context.Context, plies int) error {
	// Tree splitting with work-stealing: distribute root moves across threads
	// Each thread has a deque of moves, steals from others when empty

	// Initialize game copies and movegens for threads
	s.makeGameCopies()
	log.Info().Int("threads", s.threads).Msg("using-tree-split")

	s.currentIDDepths = make([]int, s.threads)
	g := s.game

	initialHashKey := uint64(0)
	if s.transpositionTableOptim {
		initialHashKey = s.ttable.Zobrist().Hash(
			g.Board().GetSquares(),
			g.RackFor(s.solvingPlayer),
			g.RackFor(1-s.solvingPlayer),
			false, g.ScorelessTurns(),
		)
	}

	α := -HugeNumber
	β := HugeNumber
	if s.firstWinOptim {
		α = -1
		β = 1
	} else if s.nullWindowOptim {
		α = -1
		β = 0
	}

	// Generate first layer of moves (unless already set by wrapper for multiple variations)
	movesAlreadyGenerated := s.initialMoves != nil && len(s.initialMoves) > 0 && s.initialMoves[0] != nil && len(s.initialMoves[0]) > 0

	if !movesAlreadyGenerated {
		s.currentIDDepths[0] = -1 // so that generateSTMPlays generates all moves first properly
		s.initialMoves = make([][]tinymove.SmallMove, s.threads)
		s.initialMoves[0] = s.generateSTMPlays(0, 0)
		// assignEstimates for the very first time around
		s.assignEstimates(s.initialMoves[0], 0, 0, tinymove.InvalidTinyMove)
	} else {
		// Moves already set, but need to ensure array is sized for threads
		if len(s.initialMoves) < s.threads {
			newInitialMoves := make([][]tinymove.SmallMove, s.threads)
			newInitialMoves[0] = s.initialMoves[0]
			s.initialMoves = newInitialMoves
		}
	}

	start := 1
	if !s.iterativeDeepeningOptim {
		start = plies
	}

	for p := start; p <= plies; p++ {
		if s.iterativeDeepeningOptim {
			log.Info().Int("plies", p).Msg("deepening-iteratively")
		}

		rootMoves := s.initialMoves[0]
		if len(rootMoves) == 0 {
			return errors.New("no moves generated")
		}

		// Create work deques for each thread with move indices
		deques := make([]*WorkDeque, s.threads)
		movesPerThread := (len(rootMoves) + s.threads - 1) / s.threads

		for t := 0; t < s.threads; t++ {
			start := t * movesPerThread
			end := min((t+1)*movesPerThread, len(rootMoves))
			if start < len(rootMoves) {
				// Give each thread a slice of indices
				indices := make([]int, end-start)
				for i := 0; i < len(indices); i++ {
					indices[i] = start + i
				}
				deques[t] = NewWorkDeque(indices)
			} else {
				// Thread has no initial work
				deques[t] = NewWorkDeque(nil)
			}
		}

		// Shared best result
		var bestValue atomic.Int32
		bestValue.Store(int32(-HugeNumber))

		// Launch worker threads
		eg := errgroup.Group{}

		for t := 0; t < s.threads; t++ {
			threadID := t
			s.currentIDDepths[threadID] = p

			eg.Go(func() error {
				threadGame := s.game
				if threadID > 0 {
					threadGame = s.gameCopies[threadID-1]
				}
				pv := PVLine{g: threadGame}
				onTurn := threadGame.PlayerOnTurn()
				stmRack := threadGame.RackFor(onTurn)

				for {
					// Try to get work from own deque
					idx, ok := deques[threadID].Pop()

					// If no work, try to steal from others
					if !ok {
						stolen := false
						for victimID := 0; victimID < s.threads; victimID++ {
							if victimID == threadID {
								continue
							}
							idx, ok = deques[victimID].Steal()
							if ok {
								stolen = true
								break
							}
						}
						if !stolen {
							// No work anywhere, exit
							break
						}
					}

					// Get the move from rootMoves
					sm := rootMoves[idx]

					// Search this move
					moveTiles, err := threadGame.PlaySmallMove(&sm)
					if err != nil {
						return err
					}
					s.nodes.Add(1)

					childKey := uint64(0)
					if s.transpositionTableOptim {
						childKey = s.ttable.Zobrist().AddMove(initialHashKey, &sm, stmRack, moveTiles,
							onTurn == s.solvingPlayer, threadGame.ScorelessTurns(), threadGame.LastScorelessTurns())
					}

					val, err := s.negamax(
						ctx,
						childKey,
						p-1,
						-β, -α,
						&pv, threadID, true)

					threadGame.UnplayLastMove()

					if err != nil {
						return err
					}

					val = -val

					// Update the move in rootMoves
					rootMoves[idx].SetEstimatedValue(val)

					// Update shared best
					for {
						current := bestValue.Load()
						if int32(val) <= current {
							break
						}
						if bestValue.CompareAndSwap(current, int32(val)) {
							break
						}
					}
				}

				return nil
			})
		}

		err := eg.Wait()
		if err != nil {
			return err
		}

		// Sort moves by their estimated values
		sort.Slice(rootMoves, func(i, j int) bool {
			return rootMoves[i].EstimatedValue() > rootMoves[j].EstimatedValue()
		})

		// Build PV from best move
		finalBest := int16(bestValue.Load())
		if len(rootMoves) > 0 {
			pv := PVLine{g: g}
			bestMove := &move.Move{}
			conversions.SmallMoveToMove(rootMoves[0], bestMove, g.Alphabet(), g.Board(), g.RackFor(g.PlayerOnTurn()))
			pv.Update(bestMove, PVLine{g: g}, finalBest-int16(s.initialSpread))
			s.principalVariation = pv
			s.bestPVValue = finalBest - int16(s.initialSpread)
		}

		nodes := s.nodes.Load()
		log.Info().Int16("spread", finalBest-int16(s.initialSpread)).Int("ply", p).Str("pv", s.principalVariation.NLBString()).Uint64("total-nodes", nodes).Msg("best-val")
	}

	return nil
}

func (s *Solver) lazySMP(ctx context.Context, lastIteration *int16, ply int,
	initialHashKey uint64) error {

	s.currentIDDepths[0] = ply
	if s.logStream != nil {
		// fmt.Fprintf(s.logStream, "- ply: %d\n", p)
	}

	// aspiration search
	window := int16(8)
	α := *lastIteration - window
	β := *lastIteration + window

	if s.firstWinOptim {
		α = -1
		β = 1
	} else if s.nullWindowOptim {
		α = -1
		β = 0
	}

aspirationLoop:
	for {
		α = max(-HugeNumber, α)
		β = min(HugeNumber, β)
		log.Info().Int16("α", α).Int16("β", β).Int16("window", window).Int16("lastIteration", *lastIteration).
			Msg("starting")
		if !(α < β) {
			panic("unexpected alphabeta")
		}

		// start helper threads
		g := errgroup.Group{}
		cancels := make([]context.CancelFunc, s.threads-1)

		for t := 1; t < s.threads; t++ {
			// search to different plies for different threads
			s.currentIDDepths[t] = ply + t%3
			helperCtx, cancel := context.WithCancel(ctx)
			cancels[t-1] = cancel

			g.Go(func() error {
				defer func() {
					log.Debug().Msgf("Thread %d exiting", t)
				}()
				log.Debug().Msgf("Thread %d starting; searching %d deep", t, ply+t%2)

				// ignore the score for these helper threads; we're just
				// using them to help build up the transposition table.
				pv := PVLine{g: s.gameCopies[t-1]} // not being used for anything
				val, err := s.negamax(
					helperCtx, initialHashKey, s.currentIDDepths[t],
					α, β, &pv, t, true)
				if err != nil {
					log.Debug().Msgf("Thread %d error %v", t, err)
				}
				log.Debug().Msgf("Thread %d done; val returned %d, pv %s", t, val, pv.NLBString())
				// Try a few schemes to really randomize stuff.
				if t == 1 {
					sort.Slice(s.initialMoves[t], func(i, j int) bool {
						return s.initialMoves[t][i].EstimatedValue() > s.initialMoves[t][j].EstimatedValue()
					})
				} else if t == 2 {
					// do nothing, use original order
				} else if t > 2 && t <= 7 {
					// Shuffle the order of root nodes
					frand.Shuffle(len(s.initialMoves[t]), func(i, j int) {
						s.initialMoves[t][i], s.initialMoves[t][j] = s.initialMoves[t][j], s.initialMoves[t][i]
					})
				} else if t > 7 {
					// A sort of restricted shuffle?
					topfew := len(s.initialMoves[t]) / 3

					frand.Shuffle(topfew, func(i, j int) {
						s.initialMoves[t][i], s.initialMoves[t][j] = s.initialMoves[t][j], s.initialMoves[t][i]
					})
					frand.Shuffle(len(s.initialMoves[t])-topfew, func(i, j int) {
						s.initialMoves[t][i+topfew], s.initialMoves[t][j+topfew] = s.initialMoves[t][j+topfew], s.initialMoves[t][i+topfew]
					})
				}
				return err
			})
		}

		// This is the main thread. All other threads just help update the
		// transposition table, but this one actually edits the principal
		// variation.
		pv := PVLine{g: s.game}
		val, err := s.negamax(ctx, initialHashKey, ply, α, β, &pv, 0, true)

		if err != nil {
			log.Err(err).Msg("negamax-error-most-likely-timeout")
			// we keep the last known variation; don't change it.
		} else {
			sort.Slice(s.initialMoves[0], func(i, j int) bool {
				return s.initialMoves[0][i].EstimatedValue() > s.initialMoves[0][j].EstimatedValue()
			})

			s.principalVariation = pv
			s.bestPVValue = val - int16(s.initialSpread)
			log.Info().
				Int16("α", α).
				Int16("β", β).
				Int16("window", window).
				Int16("spread", val).
				Int("ply", ply).
				Str("pv", pv.NLBString()).Msg("best-val")
		}
		// stop helper threads cleanly
		for _, c := range cancels {
			if c != nil {
				c()
			}
		}

		err = g.Wait()
		if err != nil {
			if err.Error() == "context canceled" {
				log.Debug().Msg("helper threads exited with a canceled context")
			} else {
				log.Err(err).Msg("returning an error after waiting for helper threads")
				return err
			}
		}
		log.Debug().Int16("α", α).Int16("β", β).Int16("val", val).Msg("iteration-done")

		if val > α && val < β {
			*lastIteration = val
			break aspirationLoop
		}
		window *= 2

		if val <= α {
			// fail low
			for α-window < -HugeNumber {
				window /= 2
			}
			α -= window
			β -= window / 3
		} else {
			// fail high
			for β+window > HugeNumber {
				window /= 2
			}
			β += window
			α += window / 3
		}
		log.Debug().Int16("new-α", α).Int16("new-β", β).Int16("new-window", window).Msg("re-iterating")
	}
	return nil
}

// solveWithMultipleVariations wraps any algorithm to solve for multiple variations.
// It repeatedly calls the provided algorithm, each time excluding the previous winner.
func (s *Solver) solveWithMultipleVariations(ctx context.Context, plies int,
	algorithmFunc func(context.Context, int) error) error {

	if s.solveMultipleVariations == 0 {
		s.solveMultipleVariations = 1
	}

	// Generate initial moves ONCE before the loop, like the original code did
	s.currentIDDepths = make([]int, s.threads)
	s.currentIDDepths[0] = -1
	s.initialMoves = make([][]tinymove.SmallMove, s.threads)
	s.initialMoves[0] = s.generateSTMPlays(0, 0)
	s.assignEstimates(s.initialMoves[0], 0, 0, tinymove.InvalidTinyMove)

	// Do initial 1-ply search ONCE for move ordering, just like the original code
	initialHashKey := s.ttable.Zobrist().Hash(
		s.game.Board().GetSquares(),
		s.game.RackFor(s.solvingPlayer),
		s.game.RackFor(1-s.solvingPlayer),
		false, s.game.ScorelessTurns(),
	)
	α := -HugeNumber
	β := HugeNumber
	if s.firstWinOptim {
		α = -1
		β = 1
	} else if s.nullWindowOptim {
		α = -1
		β = 0
	}
	pv := PVLine{g: s.game}
	s.currentIDDepths[0] = 1
	_, _ = s.negamax(context.Background(), initialHashKey, 1, α, β, &pv, 0, true)
	sort.Slice(s.initialMoves[0], func(i, j int) bool {
		return s.initialMoves[0][i].EstimatedValue() > s.initialMoves[0][j].EstimatedValue()
	})

	variationsFound := 0
	var lastWinner *move.Move

searchLoop:
	for variationsFound < s.solveMultipleVariations {
		toDelete := -1
		if lastWinner != nil {
			// Delete the last winner from s.initialMoves[0]
			tinyWinner := conversions.MoveToTinyMove(lastWinner)
			for i := range s.initialMoves[0] {
				if s.initialMoves[0][i].TinyMove() == tinyWinner {
					toDelete = i
					log.Info().Str("last-winner", s.game.Board().MoveDescriptionWithPlaythrough(lastWinner)).
						Msg("finding-new-variation-deleting-last-winner")
					break
				}
			}
		}

		if toDelete != -1 {
			ret := make([]tinymove.SmallMove, 0)
			ret = append(ret, s.initialMoves[0][:toDelete]...)
			ret = append(ret, s.initialMoves[0][toDelete+1:]...)
			s.initialMoves[0] = ret
		}

		if len(s.initialMoves[0]) == 0 {
			// No more moves to try
			log.Info().Msg("no more moves available for additional variations")
			break searchLoop
		}

		// Run the algorithm - it will use the filtered s.initialMoves[0]
		err := algorithmFunc(ctx, plies)
		if err != nil {
			log.Err(err).Msg("algorithm-error-in-multiple-variations")
			break searchLoop
		}

		// Save this variation
		lastWinner = s.principalVariation.Moves[0]
		variationsFound++
		s.variations = append(s.variations, s.principalVariation)

		log.Info().Int("variation", variationsFound).
			Int("requested", s.solveMultipleVariations).
			Str("pv", s.principalVariation.NLBString()).
			Msg("found-variation")
	}

	// If we solved many variations, the best is the first one
	if len(s.variations) > 0 {
		s.principalVariation = s.variations[0]
		s.bestPVValue = s.principalVariation.score
	}

	return nil
}

// iterativelyDeepen is single-threaded version.
func (s *Solver) iterativelyDeepen(ctx context.Context, plies int) error {
	// Auto-select parallel algorithm if multi-threaded but no algorithm chosen
	if s.threads >= 2 && !s.lazySMPOptim && !s.abdadaOptim && !s.treeSplitOptim {
		// Generate initial moves to determine branching factor
		s.currentIDDepths = make([]int, 1)
		s.currentIDDepths[0] = -1
		s.initialMoves = make([][]tinymove.SmallMove, 1)
		s.initialMoves[0] = s.generateSTMPlays(0, 0)

		rootMoves := len(s.initialMoves[0])

		// Quick decisions without probe
		if rootMoves < 50 {
			log.Info().Int("root-moves", rootMoves).Msg("auto-selecting LazySMP (few root moves)")
			s.SetLazySMP(true)
		} else if rootMoves > 500 {
			log.Info().Int("root-moves", rootMoves).Msg("auto-selecting ABDADA (very high branching factor)")
			s.SetABDADA(true)
		} else {
			// Run a 4-ply probe with 2-second timeout to measure EBF
			probePlies := 4
			if probePlies > plies {
				probePlies = plies
			}

			// Create timeout context for probe
			probeCtx, probeCancel := context.WithTimeout(ctx, 2*time.Second)

			// Temporarily enable LazySMP for probe
			s.lazySMPOptim = true
			s.ttable.ResetStats()

			log.Info().Int("probe-plies", probePlies).Int("root-moves", rootMoves).Msg("running probe for auto-selection")
			err := s.iterativelyDeepenLazySMP(probeCtx, probePlies)
			probeCancel()

			probeTimedOut := err == context.DeadlineExceeded
			probeCompleted := err == nil

			// Calculate EBF from probe stats
			created := s.ttable.Created()
			var ebf float64
			if created > 0 && probePlies > 0 {
				ebf = math.Pow(float64(created), 1.0/float64(probePlies))
			}

			log.Info().
				Uint64("created", created).
				Float64("ebf", ebf).
				Int("root-moves", rootMoves).
				Int("threads", s.threads).
				Bool("timed-out", probeTimedOut).
				Msg("probe stats for auto-selection")

			// If probe completed the full solve and we don't need multiple variations, we're done
			if probeCompleted && probePlies >= plies && s.solveMultipleVariations <= 1 {
				log.Info().Msg("probe completed full solve")
				return nil
			}

			// Apply heuristic based on EBF
			// EBF >= 17: ABDADA (bushy tree)
			// EBF < 12: LazySMP (narrow tree)
			// 12-17 with many threads and EBF >= 15: try ABDADA
			// Otherwise: LazySMP (safe default)
			if ebf >= 17.0 {
				log.Info().Float64("ebf", ebf).Msg("auto-selecting ABDADA (high EBF)")
				s.lazySMPOptim = false
				s.SetABDADA(true)
			} else if ebf < 12.0 {
				log.Info().Float64("ebf", ebf).Msg("auto-selecting LazySMP (low EBF)")
				// Already set to LazySMP
			} else if s.threads >= 16 && ebf >= 15.0 {
				log.Info().Float64("ebf", ebf).Int("threads", s.threads).Msg("auto-selecting ABDADA (high thread count, moderate EBF)")
				s.lazySMPOptim = false
				s.SetABDADA(true)
			} else {
				log.Info().Float64("ebf", ebf).Msg("auto-selecting LazySMP (default)")
				// Already set to LazySMP
			}

			// Continue with full solve - TT entries from probe are kept
			// Reset stats for clean metrics on full solve
			s.ttable.ResetStats()
			// Reset variations from probe - they should come from the actual solve
			s.variations = []PVLine{}
		}
	}

	// Dispatch to the appropriate algorithm, wrapping with multiple variations if needed
	if s.solveMultipleVariations > 1 {
		// Multiple variations requested - wrap the algorithm
		if s.lazySMPOptim {
			return s.solveWithMultipleVariations(ctx, plies, s.iterativelyDeepenLazySMP)
		} else if s.abdadaOptim {
			return s.solveWithMultipleVariations(ctx, plies, s.iterativelyDeepenABDADA)
		} else if s.treeSplitOptim {
			return s.solveWithMultipleVariations(ctx, plies, s.iterativelyDeepenTreeSplit)
		} else {
			// Single-threaded doesn't support multiple variations
			return errors.New("you must use multi-threading to solve multiple variations")
		}
	}

	// Single variation - call algorithm directly
	if s.lazySMPOptim {
		return s.iterativelyDeepenLazySMP(ctx, plies)
	} else if s.abdadaOptim {
		return s.iterativelyDeepenABDADA(ctx, plies)
	} else if s.treeSplitOptim {
		return s.iterativelyDeepenTreeSplit(ctx, plies)
	}
	s.currentIDDepths = make([]int, 1)
	g := s.game

	initialHashKey := uint64(0)
	if s.transpositionTableOptim {
		initialHashKey = s.ttable.Zobrist().Hash(
			g.Board().GetSquares(),
			g.RackFor(s.solvingPlayer),
			g.RackFor(1-s.solvingPlayer),
			false, g.ScorelessTurns(),
		)
	}

	α := -HugeNumber
	β := HugeNumber
	if s.firstWinOptim {
		// Search a very small window centered around 0. We're just trying
		// to find something that surpasses it.
		α = -1
		β = 1
	} else if s.nullWindowOptim {
		α = -1
		β = 0
	}

	// Generate first layer of moves.
	s.currentIDDepths[0] = -1 // so that generateSTMPlays generates all moves first properly.
	s.initialMoves = make([][]tinymove.SmallMove, 1)
	s.initialMoves[0] = s.generateSTMPlays(0, 0)
	// assignEstimates for the very first time around.
	s.assignEstimates(s.initialMoves[0], 0, 0, tinymove.InvalidTinyMove)
	start := 1
	if !s.iterativeDeepeningOptim {
		start = plies
	}

	for p := start; p <= plies; p++ {
		if s.iterativeDeepeningOptim {
			log.Info().Int("plies", p).Msg("deepening-iteratively")
		}
		s.currentIDDepths[0] = p
		if s.logStream != nil {
			// fmt.Fprintf(s.logStream, "- ply: %d\n", p)
		}
		pv := PVLine{g: g}
		val, err := s.negamax(ctx, initialHashKey, p, α, β, &pv, 0, true)
		if err != nil {
			return err
		}
		nodes := s.nodes.Load()
		log.Info().Int16("spread", val).Int("ply", p).Str("pv", pv.NLBString()).Uint64("total-nodes", nodes).Msg("best-val")
		// Sort top layer of moves by value for the next time around.
		sort.Slice(s.initialMoves[0], func(i, j int) bool {
			return s.initialMoves[0][i].EstimatedValue() > s.initialMoves[0][j].EstimatedValue()
		})
		s.principalVariation = pv
		s.bestPVValue = val - int16(s.initialSpread)
	}
	return nil

}

func (s *Solver) negamax(ctx context.Context, nodeKey uint64, depth int, α, β int16, pv *PVLine, thread int, pvNode bool) (int16, error) {
	if ctx.Err() != nil {
		return 0, ctx.Err()
	}
	g := s.game
	if thread > 0 {
		g = s.gameCopies[thread-1]
	}
	onTurn := g.PlayerOnTurn()
	ourSpread := g.SpreadFor(onTurn)

	alphaOrig := α
	ttMove := tinymove.InvalidTinyMove
	if !(pvNode || α == β-1) {
		panic("bad conditions")
	}
	if s.transpositionTableOptim {
		ttEntry := s.ttable.lookup(nodeKey)
		if ttEntry.valid() && ttEntry.depth() >= uint8(depth) {
			score := ttEntry.getScore()
			flag := ttEntry.flag()
			// add spread back in; we subtract them when storing.
			score += int16(ourSpread)
			if flag == TTExact {
				if !pvNode {
					return score, nil
				}
			} else if flag == TTLower {
				α = max(α, score)
			} else if flag == TTUpper {
				β = min(β, score)
			}
			if α >= β {
				if !pvNode { // don't cut off PV node
					return score, nil
				}
			}
			// search hash move first.
			ttMove = ttEntry.getPlay()
		}
	}

	if depth == 0 || g.Playing() != pb.PlayState_PLAYING {
		// Evaluate the state.
		// A very simple evaluation function for now. Just the current spread,
		// even if the game is not over yet.
		spreadNow := g.SpreadFor(g.PlayerOnTurn())
		return int16(spreadNow), nil
	}
	childPV := PVLine{g: g}

	children := s.generateSTMPlays(depth, thread)
	stmRack := g.RackFor(g.PlayerOnTurn())
	if s.currentIDDepths[thread] != depth {
		// If we're not at the top level, assign estimates. Otherwise,
		// we assign the values as estimates in the loop below.
		s.assignEstimates(children, depth, thread, ttMove)
	}

	bestValue := -HugeNumber
	// logIndent := strings.Repeat(" ", max(2*(s.currentIDDepths[thread]-depth), 0))
	if s.logStream != nil {
		// fmt.Fprintf(s.logStream, "  %vplays:\n", logIndent)
	}
	var bestMove tinymove.SmallMove
	var err error
	var moveTiles *[21]tilemapping.MachineLetter

	// ABDADA: track deferred moves for second pass
	var deferredMoves []int

	// First pass: search non-deferred moves
	for idx := range children {
		// ABDADA: check if this move should be deferred
		if s.abdadaOptim && s.abdadaTable.deferMove(nodeKey, children[idx], depth) {
			deferredMoves = append(deferredMoves, idx)
			continue
		}

		// ABDADA: register that we're starting to search this move
		if s.abdadaOptim {
			s.abdadaTable.startingSearch(nodeKey, children[idx], depth)
		}
		if s.logStream != nil {
			// fmt.Fprintf(s.logStream, "  %v- play: %v\n", logIndent, children[idx].ShortDescription())
		}
		moveTiles, err = g.PlaySmallMove(&children[idx])
		if err != nil {
			return 0, err
		}
		s.nodes.Add(1)
		childKey := uint64(0)
		if s.transpositionTableOptim {
			childKey = s.ttable.Zobrist().AddMove(nodeKey, &children[idx], stmRack, moveTiles,
				onTurn == s.solvingPlayer, g.ScorelessTurns(), g.LastScorelessTurns())
		}
		var value int16
		// negascout
		if idx == 0 || !s.negascoutOptim {
			value, err = s.negamax(ctx, childKey, depth-1, -β, -α, &childPV, thread, pvNode)
			if err != nil {
				g.UnplayLastMove()
				// ABDADA: clean up on error
				if s.abdadaOptim {
					s.abdadaTable.finishedSearch(nodeKey, children[idx])
				}
				return value, err
			}
		} else {
			value, err = s.negamax(ctx, childKey, depth-1, -α-1, -α, &childPV, thread, false)
			if err != nil {
				g.UnplayLastMove()
				// ABDADA: clean up on error
				if s.abdadaOptim {
					s.abdadaTable.finishedSearch(nodeKey, children[idx])
				}
				return value, err
			}
			if α < -value && -value < β {
				// re-search with wider window
				value, err = s.negamax(ctx, childKey, depth-1, -β, -α, &childPV, thread, pvNode)
			}
		}
		if err != nil {
			g.UnplayLastMove()
			// ABDADA: clean up on error
			if s.abdadaOptim {
				s.abdadaTable.finishedSearch(nodeKey, children[idx])
			}
			return value, err
		}
		g.UnplayLastMove()

		// ABDADA: finished searching this move
		if s.abdadaOptim {
			s.abdadaTable.finishedSearch(nodeKey, children[idx])
		}
		if s.logStream != nil {
			// fmt.Fprintf(s.logStream, "  %v  value: %v\n", logIndent, value)
		}
		if -value > bestValue {
			bestValue = -value
			bestMove = children[idx]
			// allocate a move to update the pv
			m := &move.Move{}
			conversions.SmallMoveToMove(bestMove, m, g.Alphabet(), g.Board(), stmRack)
			pv.Update(m, childPV, bestValue-int16(s.initialSpread))
		}
		if s.currentIDDepths[thread] == depth {
			children[idx].SetEstimatedValue(-value)
		}

		α = max(α, bestValue)
		if s.logStream != nil {
			// fmt.Fprintf(s.logStream, "  %v  α: %v\n", logIndent, α)
			// fmt.Fprintf(s.logStream, "  %v  β: %v\n", logIndent, β)
		}
		if bestValue >= β {
			break // beta cut-off
		}
		childPV.Clear() // clear the child node's pv for the next child node
	}

	// ABDADA: Second pass - search deferred moves
	if s.abdadaOptim && bestValue < β {
		for _, idx := range deferredMoves {
			// Don't check defer again, just search
			s.abdadaTable.startingSearch(nodeKey, children[idx], depth)

			if s.logStream != nil {
				// fmt.Fprintf(s.logStream, "  %v- play (deferred): %v\n", logIndent, children[idx].ShortDescription())
			}
			moveTiles, err = g.PlaySmallMove(&children[idx])
			if err != nil {
				return 0, err
			}
			s.nodes.Add(1)
			childKey := uint64(0)
			if s.transpositionTableOptim {
				childKey = s.ttable.Zobrist().AddMove(nodeKey, &children[idx], stmRack, moveTiles,
					onTurn == s.solvingPlayer, g.ScorelessTurns(), g.LastScorelessTurns())
			}
			var value int16
			// negascout - for deferred moves, always use full window first
			if idx == 0 || !s.negascoutOptim {
				value, err = s.negamax(ctx, childKey, depth-1, -β, -α, &childPV, thread, pvNode)
				if err != nil {
					g.UnplayLastMove()
					s.abdadaTable.finishedSearch(nodeKey, children[idx])
					return value, err
				}
			} else {
				value, err = s.negamax(ctx, childKey, depth-1, -α-1, -α, &childPV, thread, false)
				if err != nil {
					g.UnplayLastMove()
					s.abdadaTable.finishedSearch(nodeKey, children[idx])
					return value, err
				}
				if α < -value && -value < β {
					// re-search with wider window
					value, err = s.negamax(ctx, childKey, depth-1, -β, -α, &childPV, thread, pvNode)
				}
			}
			if err != nil {
				g.UnplayLastMove()
				s.abdadaTable.finishedSearch(nodeKey, children[idx])
				return value, err
			}
			g.UnplayLastMove()
			s.abdadaTable.finishedSearch(nodeKey, children[idx])

			if -value > bestValue {
				bestValue = -value
				bestMove = children[idx]
				m := &move.Move{}
				conversions.SmallMoveToMove(bestMove, m, g.Alphabet(), g.Board(), stmRack)
				pv.Update(m, childPV, bestValue-int16(s.initialSpread))
			}
			if s.currentIDDepths[thread] == depth {
				children[idx].SetEstimatedValue(-value)
			}

			α = max(α, bestValue)
			if bestValue >= β {
				break // beta cut-off
			}
			childPV.Clear()
		}
	}

	if s.transpositionTableOptim {
		// We store this value without our spread to make it spread-independent.
		// Without this, we need to hash the spread as well and this results
		// in many more TT misses.

		score := bestValue - int16(ourSpread)
		var flag uint8
		if bestValue <= alphaOrig {
			flag = TTUpper
		} else if bestValue >= β {
			flag = TTLower
		} else {
			flag = TTExact
		}
		flagAndDepth := flag<<6 + uint8(depth)
		play := bestMove.TinyMove()
		s.ttable.store(nodeKey, score, flagAndDepth, play)
		if s.logStream != nil {
			// fmt.Fprintf(s.logStream, "  %vttnodeKey: %v\n", logIndent, nodeKey)
			// fmt.Fprintf(s.logStream, "  %vttflag: %v\n", logIndent, flag)
			// fmt.Fprintf(s.logStream, "  %vttdepth: %v\n", logIndent, depth)
			// fmt.Fprintf(s.logStream, "  %vttscore: %v\n", logIndent, score)
			// fmt.Fprintf(s.logStream, "  %vttplay: %v\n", logIndent, play)
			fmt.Fprintf(s.logStream, "%d (d: %d s: %d)\n", nodeKey, depth, score)
			fmt.Fprintf(s.logStream, " cgp: %v\n\n", g.ToCGP(false))
		}
	}
	return bestValue, nil

}

func (s *Solver) Solve(ctx context.Context, plies int) (int16, []*move.Move, error) {
	s.busy = true
	defer func() {
		s.busy = false
	}()
	s.solvingPlayer = s.game.PlayerOnTurn()
	// Make sure the other player's rack isn't empty.
	if s.game.RackFor(1-s.solvingPlayer).NumTiles() == 0 {
		_, err := s.game.SetRandomRack(1-s.solvingPlayer, nil)
		if err != nil {
			return 0, nil, err
		}
	}
	if s.game.Bag().TilesRemaining() > 0 {
		return 0, nil, errors.New("bag is not empty; cannot use endgame solver")
	}
	log.Debug().Int("plies", plies).Msg("alphabeta-solve-config")
	s.requestedPlies = plies
	s.variations = []PVLine{}

	tstart := time.Now()
	s.stmMovegen.SetSortingParameter(movegen.SortByNone)
	defer s.stmMovegen.SetSortingParameter(movegen.SortByScore)
	if s.lazySMPOptim && s.abdadaOptim {
		return 0, nil, errors.New("you can only use lazySMP OR solve multiple variants, but not both")
	}
	if s.lazySMPOptim && s.treeSplitOptim {
		return 0, nil, errors.New("you can only use lazySMP OR treeSplit, but not both")
	}
	if s.lazySMPOptim || s.abdadaOptim || s.treeSplitOptim {
		if s.transpositionTableOptim {
			s.ttable.SetMultiThreadedMode()
		} else {
			return 0, nil, errors.New("cannot parallelize alphabeta without transposition table")
		}
	} else {
		s.ttable.SetSingleThreadedMode()
	}
	if s.transpositionTableOptim {
		s.ttable.Reset(s.game.Config().GetFloat64(config.ConfigTtableMemFraction), s.game.Board().Dim())
	}
	s.game.SetEndgameMode(true)
	defer s.game.SetEndgameMode(false)

	s.initialSpread = s.game.CurrentSpread()
	log.Debug().Msgf("Player %v spread at beginning of endgame: %v (%d)", s.solvingPlayer, s.initialSpread, s.game.ScorelessTurns())
	s.nodes.Store(0)
	var bestV int16
	var bestSeq []*move.Move
	// + 2 since lazysmp can search at a higher ply count
	s.game.SetStateStackLength(plies + 2)

	g := &errgroup.Group{}
	done := make(chan bool)

	g.Go(func() error {
		ticker := time.NewTicker(1 * time.Second)
		var lastNodes uint64
		for {
			select {
			case <-done:
				return nil
			case <-ticker.C:
				nodes := s.nodes.Load()
				log.Debug().Uint64("nps", nodes-lastNodes).Msg("nodes-per-second")
				lastNodes = nodes
			}
		}

	})

	g.Go(func() error {
		if s.lazySMPOptim && !s.iterativeDeepeningOptim {
			done <- true
			return errors.New("cannot use lazySMP if iterative deepening is off")
		}
		log.Debug().Msgf("Using iterative deepening with %v max plies", plies)
		err := s.iterativelyDeepen(ctx, plies)
		done <- true
		return err
	})

	err := g.Wait()
	// Go down tree and find best variation:

	bestSeq = s.principalVariation.Moves[:s.principalVariation.numMoves]
	bestV = s.bestPVValue

	// Store metrics
	s.lastSolveTime = time.Since(tstart).Seconds()
	s.lastTTableStats = s.ttable.Stats()

	log.Info().Str("ttable-stats", s.lastTTableStats).
		Float64("time-elapsed-sec", s.lastSolveTime).
		Msg("solve-returning")
	if err != nil {
		if err == context.Canceled || err == context.DeadlineExceeded {
			// ignore
			err = nil
		}
	}
	return bestV, bestSeq, err
}

func (s *Solver) ShortDetails() string {
	return s.principalVariation.NLBString()
}

// GetMetrics returns the timing and transposition table stats from the last solve
func (s *Solver) GetMetrics() string {
	return fmt.Sprintf("time-elapsed-sec:%f ttable-stats:%s", s.lastSolveTime, s.lastTTableStats)
}

// QuickAndDirtySolve is meant for a pre-endgame engine to call this function
// without having to initialize everything. The caller is responsible for
// initializations of data structures. It is single-threaded as well.
func (s *Solver) QuickAndDirtySolve(ctx context.Context, plies, thread int) (int16, []*move.Move, error) {
	s.busy = true
	defer func() {
		s.busy = false
	}()
	// Make sure the other player's rack isn't empty.
	s.solvingPlayer = s.game.PlayerOnTurn()
	if s.game.RackFor(1-s.solvingPlayer).NumTiles() == 0 {
		_, err := s.game.SetRandomRack(1-s.solvingPlayer, nil)
		if err != nil {
			return 0, nil, err
		}
	}
	if s.game.Bag().TilesRemaining() > 0 {
		return 0, nil, errors.New("bag is not empty; cannot use endgame solver")
	}
	// log.Debug().
	// 	Int("thread", thread).
	// 	Str("ourRack", s.game.RackLettersFor(s.solvingPlayer)).
	// 	Str("theirRack", s.game.RackLettersFor(1-s.solvingPlayer)).
	// 	Int("plies", plies).Msg("qdsolve-alphabeta-solve-config")
	s.requestedPlies = plies

	// tstart := time.Now()
	s.stmMovegen.SetSortingParameter(movegen.SortByNone)
	defer s.stmMovegen.SetSortingParameter(movegen.SortByScore)

	s.initialSpread = s.game.CurrentSpread()
	log.Debug().Int("thread", thread).Msgf("Player %v spread at beginning of endgame: %v (%d)", s.solvingPlayer, s.initialSpread, s.game.ScorelessTurns())

	var bestV int16
	var bestSeq []*move.Move

	// err := s.iterativelyDeepen(ctx, plies)
	// if err != nil {
	// 	log.Debug().AnErr("err", err).Msg("error iteratively deepening")
	// }
	initialHashKey := uint64(0)
	if s.transpositionTableOptim {
		initialHashKey = s.ttable.Zobrist().Hash(
			s.game.Board().GetSquares(),
			s.game.RackFor(s.solvingPlayer),
			s.game.RackFor(1-s.solvingPlayer),
			false, s.game.ScorelessTurns(),
		)
	}
	α := -HugeNumber
	β := HugeNumber
	if s.firstWinOptim {
		// Search a very small window centered around 0. We're just trying
		// to find something that surpasses it.
		α = -1
		β = 1
	}
	s.currentIDDepths = make([]int, 1) // a hack
	pv := PVLine{g: s.game}
	val, err := s.negamax(ctx, initialHashKey, plies, α, β, &pv, 0, true)
	s.principalVariation = pv
	s.bestPVValue = val - int16(s.initialSpread)

	bestSeq = s.principalVariation.Moves[:s.principalVariation.numMoves]
	bestV = s.bestPVValue
	// log.Debug().
	// 	Int("thread", thread).
	// 	Uint64("ttable-created", s.ttable.created.Load()).
	// 	Uint64("ttable-lookups", s.ttable.lookups.Load()).
	// 	Uint64("ttable-hits", s.ttable.hits.Load()).
	// 	Uint64("ttable-t2collisions", s.ttable.t2collisions.Load()).
	// 	Float64("time-elapsed-sec", time.Since(tstart).Seconds()).
	// 	Int16("bestV", bestV).
	// 	Str("bestSeq", s.principalVariation.NLBString()).
	// 	Msg("solve-returning")

	return bestV, bestSeq, err
}

func (s *Solver) SetIterativeDeepening(id bool) {
	s.iterativeDeepeningOptim = id
}

func (s *Solver) SetTranspositionTableOptim(tt bool) {
	s.transpositionTableOptim = tt
}

func (s *Solver) SetFirstWinOptim(w bool) {
	s.firstWinOptim = w
}

func (s *Solver) SetNullWindowOptim(nw bool) {
	s.nullWindowOptim = nw
}

func (s *Solver) SetNegascoutOptim(n bool) {
	s.negascoutOptim = n
}

func (s *Solver) IsSolving() bool {
	return s.busy
}

func (s *Solver) Game() *game.Game {
	return s.game
}

func (s *Solver) SetPreventSlowroll(p bool) {
	s.preventSlowroll = p
}
