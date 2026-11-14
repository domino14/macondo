package negamax

import (
	"context"
	"errors"
	"fmt"
	"io"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"
	"lukechampine.com/frand"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/movegen/wordprune"
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

// Constants for the greedy leaf playout and the stuck-tile conservation
// heuristics, ported from MAGPIE (src/impl/endgame.c). Both projects are GPL.
const (
	// Conservation bonus weights: penalize playing tiles when the opponent is
	// stuck. MAGPIE CONSERVATION_TILE_WEIGHT / CONSERVATION_VALUE_WEIGHT.
	conservationTileWeight  = 7
	conservationValueWeight = 2
	// MaxGreedyPlayoutPlies caps how many moves the greedy leaf playout may
	// play past the negamax ply limit. Two full racks can only produce 14 tile
	// plays, and each voluntary pass must be followed by a tile play (two
	// consecutive scoreless turns end the game in endgame mode), so ~23 is the
	// true worst case. MAGPIE's equivalent bound is MAX_SEARCH_DEPTH + 1 - plies.
	MaxGreedyPlayoutPlies = MaxVariantLength
	// maxNonGoingOutEstimate clamps non-going-out estimates so an accumulated
	// build-chain value can never reach GoingOutBF and masquerade as a move
	// that plays out the whole rack. macondo's estimate is an int16 with flags
	// at 1<<11..1<<13; MAGPIE has an int32 with flags at 1<<27..1<<29 and so
	// needs no clamp.
	maxNonGoingOutEstimate = GoingOutBF - 1
	// stuckMemoSize is the per-thread direct-mapped memo for oppStuckFraction.
	// Must be a power of two.
	stuckMemoSize = 1024
)

// EstimatedMoveMarker prefixes moves in a principal variation that came from
// the greedy playout rather than from the search itself.
const EstimatedMoveMarker = "~"

var (
	ErrNoEndgameSolution = errors.New("no endgame solution found")
)

// Credit: MIT-licensed https://github.com/algerbrex/blunder/blob/main/engine/search.go
type PVLine struct {
	Moves     [MaxVariantLength]*move.Move
	tinyMoves [MaxVariantLength]tinymove.SmallMove
	g         *game.Game
	score     int16
	numMoves  int
	// numSearched is how many of the leading moves came out of the alpha-beta
	// search proper. Any moves past that are the greedy playout's estimate of
	// how the rest of the game goes, and are displayed with a "~" prefix.
	numSearched int
}

// Clear the principal variation line.
func (pvLine *PVLine) Clear() {
	pvLine.numMoves = 0
	pvLine.numSearched = 0
}

// Update the principal variation line with a new best move and child PV.
// Takes a SmallMove directly to avoid allocations in the hot path.
func (pvLine *PVLine) Update(sm tinymove.SmallMove, childPV *PVLine, score int16) {
	pvLine.numMoves = 0
	pvLine.tinyMoves[0] = sm
	n := min(childPV.numMoves, MaxVariantLength-1)
	for i := 0; i < n; i++ {
		pvLine.tinyMoves[i+1] = childPV.tinyMoves[i]
	}
	pvLine.numMoves = n + 1
	pvLine.numSearched = min(childPV.numSearched+1, pvLine.numMoves)
	pvLine.score = score
}

// setPlayout records the greedy playout's moves as the whole of this line.
// None of them came from the search.
func (pvLine *PVLine) setPlayout(moves []tinymove.SmallMove) {
	n := min(len(moves), MaxVariantLength)
	copy(pvLine.tinyMoves[:n], moves[:n])
	pvLine.numMoves = n
	pvLine.numSearched = 0
}

// NumSearchedMoves returns how many of the leading moves in this variation were
// found by the search itself; the rest are the greedy playout's estimate.
func (pvLine *PVLine) NumSearchedMoves() int {
	return pvLine.numSearched
}

// MaterializeFull populates Moves[] by replaying the PV from the current
// (root) game state. Must be called after search completes, when the game
// is back at the root position.
func (pvLine *PVLine) MaterializeFull() {
	if pvLine.numMoves == 0 || pvLine.g == nil {
		return
	}
	g := pvLine.g.Copy()
	for i := 0; i < pvLine.numMoves; i++ {
		sm := pvLine.tinyMoves[i]
		rack := g.RackFor(g.PlayerOnTurn())
		m := &move.Move{}
		conversions.SmallMoveToMove(sm, m, g.Alphabet(), g.Board(), rack)
		pvLine.Moves[i] = m
		if i < pvLine.numMoves-1 {
			_ = g.PlayMove(m, false, 0)
		}
	}
}

// GetPVMove returns the best (first) move in the PV, materializing if needed.
func (pvLine *PVLine) GetPVMove() *move.Move {
	if pvLine.Moves[0] == nil && pvLine.numMoves > 0 {
		pvLine.MaterializeFull()
	}
	return pvLine.Moves[0]
}

// NumMoves returns the number of moves in this variation
func (pvLine *PVLine) NumMoves() int {
	return pvLine.numMoves
}

// Score returns the score (spread) for this variation
func (pvLine *PVLine) Score() int16 {
	return pvLine.score
}

// writePV writes the PV to sb. It replays moves onto a board copy so that
// pass-through tiles are shown with their actual letters (e.g. F(Y)KE).
func (pvLine *PVLine) writePV(sb *strings.Builder, linebreaks bool) {
	if pvLine.numMoves == 0 || pvLine.g == nil {
		return
	}
	bd := pvLine.g.Board().Copy()
	alph := pvLine.g.Alphabet()
	var m move.Move
	for i := 0; i < pvLine.numMoves; i++ {
		sm := pvLine.tinyMoves[i]
		conversions.TinyMoveToMove(sm.TinyMove(), bd, &m)
		m.SetAlphabet(alph)
		desc := bd.MoveDescriptionWithPlaythrough(&m)
		if i >= pvLine.numSearched {
			// Past the search horizon: this is the greedy playout's guess, not
			// a solved move.
			desc = EstimatedMoveMarker + desc
		}
		if linebreaks {
			fmt.Fprintf(sb, "%d: %s (%d)\n", i+1, desc, sm.Score())
		} else {
			sep := "| "
			if i == pvLine.numMoves-1 {
				sep = ""
			}
			fmt.Fprintf(sb, "%d) %s (%d) %s", i+1, desc, sm.Score(), sep)
		}
		if i < pvLine.numMoves-1 {
			bd.PlaceMoveTiles(&m)
		}
	}
}

// Convert the principal variation line to a string.
func (pvLine *PVLine) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "PV; val %d\n", pvLine.score)
	pvLine.writePV(&sb, true)
	return sb.String()
}

func (pvLine *PVLine) NLBString() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "[Value: %+d] ", pvLine.score)
	pvLine.writePV(&sb, false)
	return sb.String()
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
	skipMaterialize         bool
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
	// prunedKWGOptim builds a smaller word graph before solving, containing
	// only words playable with the tiles remaining in the endgame position.
	prunedKWGOptim bool
	// lineMismatches counts the lines of the last solve that ran to the end
	// of the game without adding up to their value (see completeLine). It
	// should stay 0.
	lineMismatches int
	// crossSetCacheOptim gives each thread a memory of the cross-sets it
	// has worked out (see cross_set.CachedGenerator). It changes only the
	// speed of a solve.
	crossSetCacheOptim bool
	// crossSetCaches are the threads' cross-set generators, kept between
	// solves: 0 is the main game's, i+1 game copy i's.
	crossSetCaches []*cross_set.CachedGenerator
	// useHeuristics turns on the MAGPIE-derived endgame heuristics: the greedy
	// leaf playout (instead of a static spread at depth 0) and the stuck-tile
	// aware move ordering (conservation bonus + build chains). On by default.
	useHeuristics bool
	// magpieOrdering selects MAGPIE's move-ordering estimate over macondo's
	// older one. Only consulted when useHeuristics is on.
	//
	// Off by default, because it measures worse here. Ordering never changes
	// the minimax value, only the node count, and macondo's existing ordering
	// needs 20-25% fewer nodes on the wide positions (VsRoy 8 plies: 14.8M vs
	// 19.3M; VsJoey 14 plies: 14.8M vs 18.7M) while tying everywhere else --
	// including the stuck EldarVsNigel position, where MAGPIE's build-chain
	// logic is supposed to earn its keep but comes out within 6 nodes of the
	// old ordering. The code is kept and tested so the choice can be revisited
	// on other positions; the greedy leaf playout and its stuck-tile
	// conservation (which do pay off) are unaffected by this flag.
	magpieOrdering bool

	// incrementalLists derives each position's move list from an ancestor's
	// instead of generating it from scratch (see pathlists.go). pathLists
	// holds each thread's lists.
	incrementalLists bool
	pathLists        []pathLists
	// childShortcuts values children that need no board without playing
	// them (see valueChildUnplayed).
	childShortcuts bool
	// keepOneOptim skips moves the keep-one bound shows cannot beat alpha
	// (see keepOneBound).
	keepOneOptim bool
	// outReplyOptim skips moves the out-reply bound shows cannot beat alpha
	// (see outReplyBound).
	outReplyOptim bool

	// solveMultipleVariations will solve multiple variations in parallel.
	solveMultipleVariations int
	principalVariation      PVLine
	bestPVValue             int16
	preventSlowroll         bool

	ttable         *TranspositionTable
	ttableFraction float64 // if > 0, overrides config value in Reset
	abdadaTable    *ABDADATable

	currentIDDepths []int
	// impureEvents counts, per thread, what made its search depend on depth:
	// leaves of an unfinished game and hits on impure table entries.
	impureEvents   []impureCounter
	requestedPlies int
	threads        int
	// nodes counts the moves made (or valued without being made), and how
	// many of those were moves of leaf playouts rather than positions the
	// search visited.
	nodes nodeCounter

	logStream  io.Writer
	busy       bool
	threadLogs []playLog

	variations []PVLine

	alsoSolveMove         tinymove.TinyMove
	alsoSolveMoveOriginal *move.Move

	// Track the last successfully completed ply depth
	lastCompletedPly int

	// Metrics from last solve
	lastSolveTime   float64
	lastTTableStats string

	// Per-thread bump allocators for SmallMove slices used during recursive
	// search. Using arenas avoids per-node make+GC overhead.
	arenas []*tinymove.SmallMoveArena

	// Per-thread scratch for the heuristics. All of these are sized by
	// ensureArenas, alongside the arenas above.
	//
	// stuckMemo is a direct-mapped cache of oppStuckFraction keyed by the
	// node's Zobrist hash. The stuck fraction is a pure function of the
	// position (board + opponent rack), which the hash subsumes, so memoizing
	// on it is safe and kills repeat work across iterative-deepening
	// iterations, negascout re-searches and transpositions.
	stuckMemo [][]stuckEntry
	// stuckGen invalidates the whole memo in O(1) when the game or lexicon
	// underneath it changes, the way Reset clears the transposition table.
	stuckGen uint32
	// playoutTails is each thread's cache of the rest of a playout from a
	// position one has passed through (see greedyPlayout). stuckGen is in
	// its keys, so it is forgotten along with the stuck-fraction memo.
	playoutTails [][]playoutTail
	// playoutTailOptim lets a playout stop at a position whose rest is
	// cached. It changes only the speed of a solve.
	playoutTailOptim bool
	// Build-chain scratch, only touched when the opponent is stuck.
	bcValues [][]int
	bcOrder  [][]int
	bcMoveA  []move.Move
	bcMoveB  []move.Move
	// Candidate scratch for the playout's one-tile fast path: the best
	// placement plus the pass. Per-thread and solver-owned so that taking a
	// pointer into it does not force a heap allocation at every leaf.
	singleTileBuf [][2]tinymove.SmallMove
}

// stuckEntry is one slot of the per-thread oppStuckFraction memo. A key of 0
// means "empty"; node key 0 is also how we signal "no transposition table", in
// which case the memo is bypassed entirely.
type stuckEntry struct {
	key  uint64
	frac float32
	gen  uint32
}

// Init initializes the solver
func (s *Solver) Init(m movegen.MoveGenerator, g *game.Game) error {
	if g != nil && game.IsWordSmog(g.Rules().Variant()) {
		// The endgame solver generates with the small-move recorders and
		// prunes the KWG, neither of which has an alpha dawg equivalent yet.
		// Refuse rather than silently solving with classic moves.
		return errors.New("the endgame solver does not support WordSmog yet")
	}
	if s.ttable == nil {
		s.ttable = &TranspositionTable{}
	}
	s.stmMovegen = m
	s.game = g

	s.firstWinOptim = false
	s.transpositionTableOptim = true
	s.iterativeDeepeningOptim = true
	s.negascoutOptim = true
	s.prunedKWGOptim = true
	s.crossSetCacheOptim = true
	s.useHeuristics = true
	s.magpieOrdering = false
	s.incrementalLists = true
	s.childShortcuts = true
	s.keepOneOptim = true
	s.outReplyOptim = true
	s.playoutTailOptim = true
	s.stuckGen++
	s.threads = max(1, runtime.NumCPU())
	s.alsoSolveMove = tinymove.InvalidTinyMove
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

// SetTTable injects an external transposition table. Must be called before Init
// or immediately after; the injected table survives Init (Init only allocates
// a new table when s.ttable == nil).
func (s *Solver) SetTTable(t *TranspositionTable) {
	s.ttable = t
}

// SetTTableFraction sets the memory fraction used when the TT is Reset inside
// Solve. When > 0 this overrides the config value, allowing per-game tables
// to be sized at (totalFraction / numConcurrentGames).
func (s *Solver) SetTTableFraction(f float64) {
	s.ttableFraction = f
}

func (s *Solver) Movegen() movegen.MoveGenerator {
	return s.stmMovegen
}

func (s *Solver) Variations() []PVLine {
	for i := range s.variations {
		s.variations[i].MaterializeFull()
	}
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
// ParallelAlgoAuto uses ABDADA when there is more than one thread.
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

func (s *Solver) SetAlsoSolveMove(m *move.Move) {
	s.alsoSolveMove = conversions.MoveToTinyMove(m)
	s.alsoSolveMoveOriginal = m
}

// ensureArenas creates per-thread SmallMoveArenas if not already sized for the
// current thread count. Call this before any negamax search begins.
func (s *Solver) ensureArenas() {
	if len(s.pathLists) != s.threads {
		s.releasePathLists()
		s.pathLists = make([]pathLists, s.threads)
	}
	if len(s.arenas) == s.threads {
		return
	}
	s.arenas = make([]*tinymove.SmallMoveArena, s.threads)
	for i := range s.arenas {
		s.arenas[i] = tinymove.NewSmallMoveArena(tinymove.DefaultSmallMoveArenaSize)
	}
	s.stuckMemo = make([][]stuckEntry, s.threads)
	for i := range s.stuckMemo {
		s.stuckMemo[i] = make([]stuckEntry, stuckMemoSize)
	}
	s.playoutTails = make([][]playoutTail, s.threads)
	for i := range s.playoutTails {
		s.playoutTails[i] = make([]playoutTail, playoutTailCacheSize)
	}
	s.bcValues = make([][]int, s.threads)
	s.bcOrder = make([][]int, s.threads)
	s.bcMoveA = make([]move.Move, s.threads)
	s.bcMoveB = make([]move.Move, s.threads)
	s.singleTileBuf = make([][2]tinymove.SmallMove, s.threads)
}

func (s *Solver) makeGameCopies() error {
	log.Debug().Int("threads", s.threads).Msg("makeGameCopies")
	s.gameCopies = []*game.Game{}
	for i := 0; i < s.threads-1; i++ {
		s.gameCopies = append(s.gameCopies, s.game.Copy())
		s.gameCopies[i].SetBackupMode(game.SimulationMode)
		// Copies are made inside a solve; Solve turns this off again.
		s.gameCopies[i].SetIncrementalUndo(true)
		s.gameCopies[i].SetSkipExtensionSets(true)
		// The copy shares the main game's cross-set generator; a caching
		// one is for a single thread, so the copy gets its own.
		if c, ok := s.game.CrossSetGen().(*cross_set.CachedGenerator); ok {
			s.gameCopies[i].SetCrossSetGen(s.crossSetCacheFor(i+1, c.Dist, c.Gaddag))
		}
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

// childKeyUnplayed is the table key of the position the tile play m leads to
// from the game's current position, whose key is nodeKey, without playing
// the move. onTurn is the player to move.
func (s *Solver) childKeyUnplayed(g *game.Game, m *tinymove.SmallMove, nodeKey uint64,
	onTurn int) uint64 {
	var tiles [board.MaxBoardDim]tilemapping.MachineLetter
	smallMoveTiles(g.Board(), m, &tiles)
	return s.ttable.Zobrist().AddTilePlayUnplayed(nodeKey, m, g.RackFor(onTurn), &tiles,
		onTurn == s.solvingPlayer, g.ScorelessTurns())
}

// smallMoveTiles fills out with the tiles m places on b, by position along
// the play, with 0 for a square that already holds a tile: what PlaySmallMove
// returns, without playing the move.
func smallMoveTiles(b *board.GameBoard, m *tinymove.SmallMove,
	out *[board.MaxBoardDim]tilemapping.MachineLetter) {
	row, col, vertical := m.CoordsAndVertical()
	ri, ci := 0, 1
	if vertical {
		ri, ci = 1, 0
	}
	tm := uint64(m.TinyMove())
	tidx := uint(0)
	for i := 0; i < m.PlayLength(); i++ {
		if b.HasLetter(row+ri*i, col+ci*i) {
			out[i] = 0
			continue
		}
		tile := tilemapping.MachineLetter((tm >> (20 + 6*tidx)) & 63)
		if tm&(1<<(12+tidx)) != 0 {
			tile = tile.Blank()
		}
		out[i] = tile
		tidx++
	}
}

// valueChildUnplayed values two kinds of child exactly as searching them
// would, without playing the move, which costs far more than either answer:
//
//   - a play of the whole rack ends the game at the mover's spread plus the
//     play's score and twice the opponent's tiles;
//   - a child whose value the transposition table holds returns it, when
//     the child's node would have returned from the table (an exact value
//     at a non-PV node, or a bound that closes the window). The child's key
//     is worked out from the move, so the move is never put on the board.
//
// firstOrFullWindow says the child gets the full window (the first move, or
// no negascout); otherwise it gets the null window, and a value inside
// (α, β) would be searched again as a PV node, which the table does not
// answer, so that child is played. The value is from the child's side, as
// negamax returns it. It leaves the thread's reach and impure count as the
// child's search would.
//
// childKey is the child's table key if haveKey is set (see
// childKeyUnplayed); otherwise it is worked out here.
func (s *Solver) valueChildUnplayed(g *game.Game, m *tinymove.SmallMove, nodeKey uint64,
	depth int, α, β int16, firstOrFullWindow, pvNode bool, ourSpread, onTurn, thread int,
	childKey uint64, haveKey bool) (int16, bool) {
	if !s.childShortcuts || m.IsPass() {
		return 0, false
	}
	stmRack := g.RackFor(onTurn)
	if m.TilesPlayed() == int(stmRack.NumTiles()) {
		oppTiles := g.RackFor(1 - onTurn).ScoreOn(g.Bag().LetterDistribution())
		s.impureEvents[thread].reach = 0
		return -int16(ourSpread + m.Score() + 2*oppTiles), true
	}
	if !s.transpositionTableOptim || nodeKey == 0 {
		return 0, false
	}
	if !haveKey {
		childKey = s.childKeyUnplayed(g, m, nodeKey, onTurn)
	}
	entry := s.ttable.lookup(childKey, thread)
	// The same test the child's node makes of its entry (see negamax).
	childDepth := uint8(depth - 1)
	entryDepth := entry.depth()
	if !entry.valid() || !(entryDepth == childDepth ||
		(entry.pure() && entryDepth <= childDepth)) {
		return 0, false
	}
	score := entry.getScore() - int16(ourSpread+m.Score())
	childα, childβ, childPV := -β, -α, pvNode
	if !firstOrFullWindow {
		childα, childβ, childPV = -α-1, -α, false
	}
	switch entry.flag() {
	case TTExact:
		if childPV {
			return 0, false
		}
	case TTLower:
		if max(childα, score) < childβ {
			return 0, false
		}
	case TTUpper:
		if childα < min(childβ, score) {
			return 0, false
		}
	}
	if !firstOrFullWindow && α < -score && -score < β {
		return 0, false
	}
	if entry.pure() {
		s.impureEvents[thread].reach = uint64(entryDepth)
	} else {
		// The entry holds an estimate, so the value depends on the depth.
		s.impureEvents[thread].n++
		s.impureEvents[thread].reach = 0
	}
	return score, true
}

// checkMoveLists makes every incremental move list verify itself against a
// full generation. For tests.
var checkMoveLists = false

// listsFor returns the thread's incremental move lists, bound to its game, or
// nil when they are off.
func (s *Solver) listsFor(thread int, g *game.Game, mg movegen.MoveGenerator) *pathLists {
	if !s.incrementalLists || thread >= len(s.pathLists) {
		return nil
	}
	p := &s.pathLists[thread]
	if p.g != g || p.mg != mg {
		if g.Bag().LetterDistribution().TileMapping().NumLetters()+1 > maxListTileTypes {
			return nil
		}
		p.bind(g, mg, checkMoveLists)
	}
	return p
}

// setIncrementalUndo switches the solver's games between undoing moves from
// the board's journal (on, for the length of a solve) and from whole-board
// copies. While it is on the boards also stop keeping their extension sets,
// which only shadow move generation reads and the solver never runs.
func (s *Solver) setIncrementalUndo(on bool) {
	s.game.SetIncrementalUndo(on)
	s.game.SetSkipExtensionSets(on)
	for _, g := range s.gameCopies {
		g.SetIncrementalUndo(on)
		g.SetSkipExtensionSets(on)
	}
}

// crossSetCacheBits sizes a thread's cross-set cache: 2^14 entries of 32
// bytes, half a megabyte.
const crossSetCacheBits = 14

// crossSetCacheFor returns cross-set generator i, emptied and set to answer
// from the given lexicon.
func (s *Solver) crossSetCacheFor(i int, dist *tilemapping.LetterDistribution,
	gd gaddag.WordGraph) *cross_set.CachedGenerator {
	for len(s.crossSetCaches) <= i {
		s.crossSetCaches = append(s.crossSetCaches, nil)
	}
	c := s.crossSetCaches[i]
	if c == nil {
		c = cross_set.NewCachedGenerator(dist, gd, crossSetCacheBits)
		s.crossSetCaches[i] = c
	} else {
		c.Reset(dist, gd)
	}
	c.Verify = checkMoveLists
	return c
}

// useCrossSetCache swaps the main game's cross-set generator for a caching
// one that answers from the same lexicon, and returns a function that puts
// the old one back. Game copies made meanwhile get caches of their own (see
// makeGameCopies). Call it once the generator has the solve's lexicon.
func (s *Solver) useCrossSetCache() func() {
	orig := s.game.CrossSetGen()
	csg, ok := orig.(*cross_set.GaddagCrossSetGenerator)
	if !ok || !s.crossSetCacheOptim {
		return func() {}
	}
	s.game.SetCrossSetGen(s.crossSetCacheFor(0, csg.Dist, csg.Gaddag))
	return func() { s.game.SetCrossSetGen(orig) }
}

// SetCrossSetCache turns the threads' cross-set caches on or off. They are
// on by default and change only the speed of a solve.
func (s *Solver) SetCrossSetCache(v bool) {
	s.crossSetCacheOptim = v
}

// releasePathLists detaches every thread's incremental lists from its game.
func (s *Solver) releasePathLists() {
	for i := range s.pathLists {
		s.pathLists[i].release()
	}
}

// SearchedNodes is the number of positions the last solve's search visited:
// its node count without the moves of the leaf playouts.
func (s *Solver) SearchedNodes() uint64 {
	return s.nodes.Load() - s.nodes.playoutMoves()
}

// SetIncrementalMoveLists turns the incremental move lists on or off. They
// are on by default and change only the speed of a solve.
func (s *Solver) SetIncrementalMoveLists(v bool) {
	s.incrementalLists = v
}

// SetOutReplyBound turns the out-reply bound on or off. It is on by default
// and changes only the speed of a solve.
func (s *Solver) SetOutReplyBound(v bool) {
	s.outReplyOptim = v
}

// SetPlayoutTailCache turns the cache of playout tails on or off. It is on
// by default and changes only the speed of a solve.
func (s *Solver) SetPlayoutTailCache(v bool) {
	s.playoutTailOptim = v
}

// SetKeepOneBound turns the keep-one bound on or off. It is on by default
// and changes only the speed of a solve.
func (s *Solver) SetKeepOneBound(v bool) {
	s.keepOneOptim = v
}

// SetChildShortcuts turns valuing children without playing them on or off.
// It is on by default and changes only the speed of a solve.
func (s *Solver) SetChildShortcuts(v bool) {
	s.childShortcuts = v
}

// generateSTMPlays returns the moves to search at this node.
// fromArena is true when the returned slice was bump-allocated from the
// per-thread arena and must be Dealloc'd by the caller after use.
// It is false when returning s.initialMoves[thread] (root depth) or when
// generating the initial move list (currentIDDepths == -1, uses make).
func (s *Solver) generateSTMPlays(depth, thread int) ([]tinymove.SmallMove, bool) {
	// STM means side-to-move
	g := s.game
	mg := s.stmMovegen

	if thread > 0 {
		mg = s.movegens[thread-1]
		g = s.gameCopies[thread-1]
	}

	stmRack := g.RackFor(g.PlayerOnTurn())
	if s.currentIDDepths[thread] == depth {
		return s.initialMoves[thread], false
	}

	if s.currentIDDepths[thread] >= 0 {
		if lists := s.listsFor(thread, g, mg); lists != nil {
			l := lists.get(g.PlayerOnTurn())
			n := len(l.moves)
			genPlays := s.arenas[thread].Alloc(n + 1)
			copy(genPlays, l.moves)
			genPlays[n] = tinymove.PassMove()
			return genPlays, true
		}
	}
	mg.GenAll(stmRack, false)
	plays := mg.SmallPlays()

	// Use the arena for recursive (in-search) move generation.
	// Use make for initial move population (currentIDDepths == -1).
	if s.currentIDDepths[thread] >= 0 {
		genPlays := s.arenas[thread].Alloc(len(plays))
		copy(genPlays, plays)
		return genPlays, true
	}
	genPlays := make([]tinymove.SmallMove, len(plays))
	copy(genPlays, plays)
	return genPlays, false
}

// assignEstimates scores each move for move ordering and sorts descending.
// Estimates never change the minimax value, only the order in which alpha-beta
// visits moves, and thus how quickly it can cut off.
//
// nodeKey is the position's Zobrist hash, used to memoize the opponent's stuck
// fraction. Note this may generate moves for the opponent, which clobbers the
// thread's movegen play list -- `moves` must already be a copy (which
// generateSTMPlays guarantees).
func (s *Solver) assignEstimates(moves []tinymove.SmallMove, depth, thread int, ttMove tinymove.TinyMove,
	nodeKey uint64) {
	g := s.game

	if thread > 0 {
		g = s.gameCopies[thread-1]
	}
	if s.useHeuristics && s.magpieOrdering {
		mg := s.stmMovegen
		if thread > 0 {
			mg = s.movegens[thread-1]
		}
		s.assignEstimatesHeuristic(moves, g, mg, thread, ttMove, nodeKey)
		return
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
			if thread >= 6 && s.lazySMPOptim {
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
	slices.SortFunc(moves, func(a, b tinymove.SmallMove) int {
		return int(b.EstimatedValue()) - int(a.EstimatedValue())
	})
}

// assignEstimatesHeuristic is the MAGPIE move ordering (assign_estimates,
// endgame.c:1641). Compared to the older macondo ordering it replaces the
// "-5 per tile played" bias with MAGPIE's static endgame equity (score minus
// the face value of the tiles played), and, when the opponent has stuck tiles,
// adds the conservation bonus and the build-chain boost.
func (s *Solver) assignEstimatesHeuristic(moves []tinymove.SmallMove, g *game.Game,
	mg movegen.MoveGenerator, thread int, ttMove tinymove.TinyMove, nodeKey uint64) {

	stmRack := g.RackFor(g.PlayerOnTurn())
	pnot := (g.PlayerOnTurn() + 1) % g.NumPlayers()
	otherRack := g.RackFor(pnot)
	numTilesOnRack := int(stmRack.NumTiles())
	ld := g.Bag().LetterDistribution()
	lastMoveWasPass := g.ScorelessTurns() > g.LastScorelessTurns()

	frac := s.oppStuckFraction(g, mg, nodeKey, thread)
	buildValues := s.computeBuildChainValues(moves, g.Board(), frac, thread)

	for idx := range moves {
		m := &moves[idx]
		tilesPlayed := m.TilesPlayed()
		isNonPassPartial := !m.IsPass() && tilesPlayed < numTilesOnRack

		bonus := 0
		if frac > 0 && isNonPassPartial {
			bonus = conservationBonus(m, ld, frac)
		}

		if tilesPlayed == numTilesOnRack {
			// Going out. Credit twice the opponent's rack, as the real
			// end-of-game scoring does.
			m.SetEstimatedValue(int16(m.Score()+2*otherRack.ScoreOn(ld)) + GoingOutBF)
		} else {
			var estimate int
			if buildValues != nil {
				// buildValues[idx] already includes the move's own score plus
				// any extension chain value; prorate the boost by how stuck the
				// opponent actually is.
				estimate = m.Score()
				if buildValues[idx] > estimate {
					estimate += int(frac * float32(buildValues[idx]-estimate))
				}
				estimate -= bonus
			} else {
				estimate = m.Score() - bonus
				if isNonPassPartial {
					// Static endgame equity: keep the high-value tiles.
					estimate -= playedTilesFaceValue(m, ld)
				}
			}
			estimate += threadJitter(s.threads, thread, tilesPlayed)
			// Clamp so a long build chain can never reach GoingOutBF and get
			// mistaken for a move that plays out the whole rack.
			estimate = min(max(estimate, -maxNonGoingOutEstimate), maxNonGoingOutEstimate)
			m.SetEstimatedValue(int16(estimate))
		}

		// XXX: should also verify validity of ttMove later.
		if m.TinyMove() == ttMove {
			m.AddEstimatedValue(HashMoveBF)
		}

		if lastMoveWasPass && m.IsPass() {
			m.AddEstimatedValue(EarlyPassBF)
		}
	}
	slices.SortFunc(moves, func(a, b tinymove.SmallMove) int {
		return int(b.EstimatedValue()) - int(a.EstimatedValue())
	})
}

// threadJitter gives each parallel search thread a slightly different move
// order so they explore different parts of the tree. Odd threads lean toward
// playing more tiles, even threads toward playing fewer. Translated from
// MAGPIE's compute_thread_jitter (endgame.c:1484), minus its random component,
// so that node counts stay reproducible.
func threadJitter(threads, thread, tilesPlayed int) int {
	if threads <= 1 || thread == 0 {
		return 0
	}
	bias := thread * tilesPlayed
	if thread%2 == 1 {
		return bias
	}
	return -bias
}

func (s *Solver) iterativelyDeepenLazySMP(ctx context.Context, plies int) error {
	// Generate first layer of moves.
	if plies < 2 {
		return errors.New("use at least 2 plies")
	}
	s.makeGameCopies()
	log.Info().Int("threads", s.threads).Msg("using-lazy-smp")
	s.currentIDDepths = make([]int, s.threads)
	s.impureEvents = make([]impureCounter, s.threads)
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
		s.initialMoves[0], _ = s.generateSTMPlays(0, 0)
		// assignEstimates for the very first time around.
		s.assignEstimates(s.initialMoves[0], 0, 0, tinymove.InvalidTinyMove, initialHashKey)

		pv := PVLine{g: s.game}
		// Do initial search so that we can have a good estimate for
		// move ordering.
		s.currentIDDepths[0] = 1
		lastIteration, _ = s.negamax(ctx, initialHashKey, 1, α, β, &pv, 0, true)
		// Sort the moves by valuation.
		slices.SortFunc(s.initialMoves[0], func(a, b tinymove.SmallMove) int {
			return int(b.EstimatedValue()) - int(a.EstimatedValue())
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

	var lastPlyWinner tinymove.SmallMove
	var lastPlyWinnerSet bool
	var lastPlyWinnerMatches int
	for p := 2; p <= plies; p++ {

		log.Info().Int("plies", p).Msg("deepening-iteratively")
		err := s.lazySMP(ctx, &lastIteration, p, initialHashKey)
		if err != nil {
			// could be a timeout, most likely.
			log.Err(err).Msg("lazySMP-possible-error")
			return err
		}
		// Successfully completed this ply
		s.lastCompletedPly = p

		if s.preventSlowroll && s.principalVariation.numMoves > 0 {
			justWon := s.principalVariation.tinyMoves[0]
			if lastPlyWinnerSet {
				if lastPlyWinner.TinyMove() == justWon.TinyMove() {
					lastPlyWinnerMatches++
					if lastPlyWinnerMatches > 4 {
						log.Info().Msg("preventing slowroll; exiting iterative deepening early")
						break
					} else {
						finalSpread := s.principalVariation.score + int16(s.game.CurrentSpread())
						rackSize := int(s.game.RackFor(s.solvingPlayer).NumTiles())
						if finalSpread > 0 && lastPlyWinner.TilesPlayed() == rackSize {
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
			lastPlyWinner = justWon
			lastPlyWinnerSet = true
		}
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
	s.impureEvents = make([]impureCounter, s.threads)
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
		s.initialMoves[0], _ = s.generateSTMPlays(0, 0)
		s.assignEstimates(s.initialMoves[0], 0, 0, tinymove.InvalidTinyMove, initialHashKey)

		// Do initial 1-ply search for move ordering
		pv := PVLine{g: s.game}
		s.currentIDDepths[0] = 1
		lastIteration, _ = s.negamax(ctx, initialHashKey, 1, α, β, &pv, 0, true)
		slices.SortFunc(s.initialMoves[0], func(a, b tinymove.SmallMove) int {
			return int(b.EstimatedValue()) - int(a.EstimatedValue())
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
	rootPure := false
	for p := s.nextIDDepth(1, plies); p <= plies && !rootPure; p = s.nextIDDepth(p, plies) {
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
			impureBefore := s.impureEvents[0].n
			val, err := s.negamax(ctx, initialHashKey, p, aspα, aspβ, &mainPV, 0, true)
			rootPure = s.impureEvents[0].n == impureBefore

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

			slices.SortFunc(s.initialMoves[0], func(a, b tinymove.SmallMove) int {
				return int(b.EstimatedValue()) - int(a.EstimatedValue())
			})

			s.principalVariation = mainPV
			s.bestPVValue = val - int16(s.initialSpread)
			log.Info().Int16("spread", val).Int("ply", p).Str("pv", mainPV.NLBString()).Msg("abdada-best-val")

			// Check if we're within the aspiration window
			if val > aspα && val < aspβ {
				lastIteration = val
				// Successfully completed this ply
				s.lastCompletedPly = p
				break aspirationLoop
			}
			rootPure = false

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
	if rootPure {
		// The main thread's search reached only finished games, so its value
		// is the game's and holds at the requested depth.
		s.lastCompletedPly = plies
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
	s.impureEvents = make([]impureCounter, s.threads)
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
		s.initialMoves[0], _ = s.generateSTMPlays(0, 0)
		// assignEstimates for the very first time around
		s.assignEstimates(s.initialMoves[0], 0, 0, tinymove.InvalidTinyMove, initialHashKey)
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
					s.nodes.add(threadID, 1)

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
		slices.SortFunc(rootMoves, func(a, b tinymove.SmallMove) int {
			return int(b.EstimatedValue()) - int(a.EstimatedValue())
		})

		// Build PV from best move
		finalBest := int16(bestValue.Load())
		if len(rootMoves) > 0 {
			pv := PVLine{g: g}
			emptyChild := PVLine{g: g}
			pv.Update(rootMoves[0], &emptyChild, finalBest-int16(s.initialSpread))
			s.principalVariation = pv
			s.bestPVValue = finalBest - int16(s.initialSpread)
		}

		// Successfully completed this ply
		s.lastCompletedPly = p

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
				if e := log.Debug(); e.Enabled() {
					e.Msgf("Thread %d done; val returned %d, pv %s", t, val, pv.NLBString())
				}
				// Try a few schemes to really randomize stuff.
				if t == 1 {
					slices.SortFunc(s.initialMoves[t], func(a, b tinymove.SmallMove) int {
						return int(b.EstimatedValue()) - int(a.EstimatedValue())
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
			slices.SortFunc(s.initialMoves[0], func(a, b tinymove.SmallMove) int {
				return int(b.EstimatedValue()) - int(a.EstimatedValue())
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
	s.impureEvents = make([]impureCounter, s.threads)
	s.currentIDDepths[0] = -1
	s.initialMoves = make([][]tinymove.SmallMove, s.threads)
	s.initialMoves[0], _ = s.generateSTMPlays(0, 0)

	// Do initial 1-ply search ONCE for move ordering, just like the original code
	initialHashKey := s.ttable.Zobrist().Hash(
		s.game.Board().GetSquares(),
		s.game.RackFor(s.solvingPlayer),
		s.game.RackFor(1-s.solvingPlayer),
		false, s.game.ScorelessTurns(),
	)
	s.assignEstimates(s.initialMoves[0], 0, 0, tinymove.InvalidTinyMove, initialHashKey)
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
	slices.SortFunc(s.initialMoves[0], func(a, b tinymove.SmallMove) int {
		return int(b.EstimatedValue()) - int(a.EstimatedValue())
	})

	variationsFound := 0
	lastWinner := tinymove.InvalidTinyMove

searchLoop:
	for variationsFound < s.solveMultipleVariations {
		toDelete := -1
		if lastWinner != tinymove.InvalidTinyMove {
			// Delete the last winner from s.initialMoves[0]
			for i := range s.initialMoves[0] {
				if s.initialMoves[0][i].TinyMove() == lastWinner {
					toDelete = i
					log.Info().Uint64("last-winner-tinymove", uint64(lastWinner)).
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
			// Save the best move found so far before breaking
			if s.principalVariation.numMoves > 0 {
				s.variations = append(s.variations, s.principalVariation)
				log.Info().Str("partial-result", s.principalVariation.NLBString()).
					Msg("saved-partial-result-before-timeout")
			}
			break searchLoop
		}

		// Save this variation
		if s.principalVariation.numMoves > 0 {
			lastWinner = s.principalVariation.tinyMoves[0].TinyMove()
		}
		variationsFound++
		s.variations = append(s.variations, s.principalVariation)

		log.Info().Int("variation", variationsFound).
			Int("requested", s.solveMultipleVariations).
			Str("pv", s.principalVariation.NLBString()).
			Msg("found-variation")
	}

	// Handle also-solve-var: solve specified move if not already found
	if s.alsoSolveMove != tinymove.InvalidTinyMove {
		found := false
		for _, v := range s.variations {
			if v.numMoves > 0 && v.tinyMoves[0].TinyMove() == s.alsoSolveMove {
				found = true
				break
			}
		}
		if !found {
			// Filter initialMoves to only the specified move
			filtered := []tinymove.SmallMove{}
			for _, sm := range s.initialMoves[0] {
				if sm.TinyMove() == s.alsoSolveMove {
					filtered = append(filtered, sm)
					break
				}
			}
			if len(filtered) == 0 && s.alsoSolveMoveOriginal != nil {
				// The move was not in the generated legal moves (e.g. a phony word).
				// PlaySmallMove does no lexicon validation, so we can still solve the
				// continuation by constructing a SmallMove directly from the original move.
				sm := tinymove.TilePlayMove(
					s.alsoSolveMove,
					int16(s.alsoSolveMoveOriginal.Score()),
					uint8(s.alsoSolveMoveOriginal.TilesPlayed()),
					uint8(s.alsoSolveMoveOriginal.PlayLength()),
				)
				filtered = []tinymove.SmallMove{sm}
				log.Info().Str("move", s.alsoSolveMoveOriginal.ShortDescription()).
					Msg("also-solve-move-not-in-legal-moves-solving-anyway")
			}
			if len(filtered) > 0 {
				s.initialMoves[0] = filtered

				// Use the same ply depth that was successfully completed for the best move
				// to ensure comparable results
				pliesToUse := plies
				if s.lastCompletedPly > 0 && s.lastCompletedPly < plies {
					pliesToUse = s.lastCompletedPly
					log.Info().
						Int("requested-plies", plies).
						Int("using-plies", pliesToUse).
						Msg("using-last-completed-ply-for-also-solve-move")
				}

				// Create fresh context with 3 minute timeout for solving just this move
				freshCtx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
				defer cancel()

				err := algorithmFunc(freshCtx, pliesToUse)
				if err == nil {
					s.variations = append(s.variations, s.principalVariation)
					log.Info().Str("also-solve-move", s.principalVariation.NLBString()).
						Msg("solved-also-solve-var")
				} else {
					log.Warn().Err(err).Msg("failed-to-solve-also-solve-move")
				}
			}
		}
	}

	// If we solved many variations, the best is the first one
	if len(s.variations) > 0 {
		s.principalVariation = s.variations[0]
		s.bestPVValue = s.principalVariation.score
	}

	return nil
}

// nextIDDepth is the iteration after depth p in a solve to plies. Iterations
// cost in pairs: depths 2k+1 and 2k+2 search about as many nodes, and each pair
// costs several times the last. One whose parity differs from the target's
// mostly orders moves for the next, so a solve to a fixed depth skips it
// (1, 2, 4, 6 for six plies; 1, 3, 5 for five), as MAGPIE does (e84f23cb).
// A solve that runs out of time may so report a depth one less than it
// otherwise would, but it gets there sooner. A solve for several variations
// keeps every depth: its intermediate results are used.
func (s *Solver) nextIDDepth(p, plies int) int {
	if s.solveMultipleVariations > 1 || s.alsoSolveMove != tinymove.InvalidTinyMove ||
		p+1 >= plies || (plies-(p+1))%2 == 0 {
		return p + 1
	}
	return p + 2
}

// iterativelyDeepen is single-threaded version.
func (s *Solver) iterativelyDeepen(ctx context.Context, plies int) error {
	// Auto-select parallel algorithm if multi-threaded but no algorithm chosen.
	// ABDADA: lazy SMP's speed came from the main thread taking the values its
	// deeper-searching helpers stored, which a fixed-depth search cannot use,
	// and without them ABDADA was faster at every thread count we measured.
	if s.threads >= 2 && !s.lazySMPOptim && !s.abdadaOptim && !s.treeSplitOptim {
		log.Info().Int("threads", s.threads).Msg("auto-selecting ABDADA")
		s.initialMoves = nil
		s.SetABDADA(true)
	}

	// Dispatch to the appropriate algorithm, wrapping with multiple variations if needed
	if s.solveMultipleVariations > 1 || s.alsoSolveMove != tinymove.InvalidTinyMove {
		// Multiple variations requested or also-solve-var specified - wrap the algorithm
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

	// Single variation - call algorithm directly. The parallel searches need
	// at least two plies, and a one-ply search gains nothing from threads, so
	// it runs single-threaded.
	if plies < 2 {
		// fall through to the single-threaded search
	} else if s.lazySMPOptim {
		return s.iterativelyDeepenLazySMP(ctx, plies)
	} else if s.abdadaOptim {
		return s.iterativelyDeepenABDADA(ctx, plies)
	} else if s.treeSplitOptim {
		return s.iterativelyDeepenTreeSplit(ctx, plies)
	}
	s.currentIDDepths = make([]int, 1)
	s.impureEvents = make([]impureCounter, 1)
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
	s.initialMoves[0], _ = s.generateSTMPlays(0, 0)
	// assignEstimates for the very first time around.
	s.assignEstimates(s.initialMoves[0], 0, 0, tinymove.InvalidTinyMove, initialHashKey)
	start := 1
	if !s.iterativeDeepeningOptim {
		start = plies
	}

	for p := start; p <= plies; p = s.nextIDDepth(p, plies) {
		if s.iterativeDeepeningOptim {
			log.Info().Int("plies", p).Msg("deepening-iteratively")
		}
		s.currentIDDepths[0] = p
		if s.logStream != nil {
			// fmt.Fprintf(s.logStream, "- ply: %d\n", p)
		}
		pv := PVLine{g: g}
		impureBefore := s.impureEvents[0].n
		val, err := s.negamax(ctx, initialHashKey, p, α, β, &pv, 0, true)
		if err != nil {
			return err
		}
		rootPure := s.impureEvents[0].n == impureBefore
		nodes := s.nodes.Load()
		log.Info().Int16("spread", val).Int("ply", p).Str("pv", pv.NLBString()).Uint64("total-nodes", nodes).Msg("best-val")
		// Sort top layer of moves by value for the next time around.
		slices.SortFunc(s.initialMoves[0], func(a, b tinymove.SmallMove) int {
			return int(b.EstimatedValue()) - int(a.EstimatedValue())
		})
		s.principalVariation = pv
		s.bestPVValue = val - int16(s.initialSpread)
		s.lastCompletedPly = p
		if rootPure {
			// The search reached only finished games, so its value is the
			// game's and every deeper iteration would return it again.
			s.lastCompletedPly = plies
			break
		}
	}
	return nil

}

// impureCounter is one thread's count of the leaves of unfinished games and
// the hits on impure table entries its search has met. A node whose search
// leaves the count unchanged reached only finished games, so it is stored
// pure. reach is how many plies the node just searched needed to reach them
// all: its value holds at that depth and every greater one, but not at a
// smaller one, where some of those lines would be cut off and estimated. The
// padding keeps each thread's counter on its own cache line.
// nodeStripes is the number of per-thread node counters; a thread uses the
// one at its index modulo this.
const nodeStripes = 256

// nodeCounter counts nodes per search thread. One shared counter would be
// written by every thread at every node, and they would spend their time
// passing its cache line around.
type nodeCounter struct {
	_       [120]byte
	stripes [nodeStripes]nodeStripe
}

type nodeStripe struct {
	nodes   atomic.Uint64
	playout atomic.Uint64
	_       [112]byte
}

func (c *nodeCounter) add(thread int, n uint64) {
	c.stripes[thread&(nodeStripes-1)].nodes.Add(n)
}

func (c *nodeCounter) addPlayout(thread int, n uint64) {
	st := &c.stripes[thread&(nodeStripes-1)]
	st.nodes.Add(n)
	st.playout.Add(n)
}

// Load returns the total node count.
func (c *nodeCounter) Load() uint64 {
	var sum uint64
	for i := range c.stripes {
		sum += c.stripes[i].nodes.Load()
	}
	return sum
}

func (c *nodeCounter) playoutMoves() uint64 {
	var sum uint64
	for i := range c.stripes {
		sum += c.stripes[i].playout.Load()
	}
	return sum
}

// Store sets the total node count, and forgets the playout share.
func (c *nodeCounter) Store(v uint64) {
	for i := range c.stripes {
		c.stripes[i].nodes.Store(0)
		c.stripes[i].playout.Store(0)
	}
	c.stripes[0].nodes.Store(v)
}

type impureCounter struct {
	n     uint64
	reach uint64
	_     [112]byte
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
	impure := &s.impureEvents[thread].n
	impureAtEntry := *impure
	// The plies this node's value needs, if it turns out pure.
	reach := &s.impureEvents[thread].reach
	maxReach := uint64(0)
	if s.transpositionTableOptim {
		ttEntry := s.ttable.lookup(nodeKey, thread)
		// An entry's value is this node's value only if it was searched to
		// the same depth, or if it is pure (its search reached only finished
		// games) and needed no more plies than this node has left: a node
		// with fewer cuts some of those lines off and estimates them. Nodes meet
		// entries from other depths through passes (the moves A, Z, B and
		// pass, Z, A, pass, B reach the same position, the second with two
		// fewer plies left) and through threads searching other depths, and
		// which ones a solve meets depends on thread timing. A deeper entry's
		// move is still the best guess for ordering.
		entryDepth := ttEntry.depth()
		usable := ttEntry.valid() && (entryDepth == uint8(depth) ||
			(ttEntry.pure() && entryDepth <= uint8(depth)))
		if !usable && ttEntry.valid() && entryDepth > uint8(depth) {
			ttMove = ttEntry.getPlay()
		} else if usable {
			if ttEntry.pure() {
				maxReach = max(maxReach, uint64(entryDepth))
			} else {
				*impure++
			}
			score := ttEntry.getScore()
			flag := ttEntry.flag()
			// add spread back in; we subtract them when storing.
			score += int16(ourSpread)
			// Only a node searched with a null window takes its value or its
			// bounds from the table. A principal-variation node has to say
			// which move is best, and it finds that out only by searching
			// with its window open: once another thread's lower bound lifts
			// α to the node's true value no move can beat α, every child
			// fails low, and the "best" move is whichever one happened to
			// return that bound first. The value would be right and the
			// line wrong. The entry still orders the moves.
			if !pvNode {
				if flag == TTExact {
					*reach = maxReach
					return score, nil
				}
				if flag == TTLower {
					α = max(α, score)
				} else if flag == TTUpper {
					β = min(β, score)
				}
				if α >= β {
					*reach = maxReach
					return score, nil
				}
			}
			// search hash move first.
			ttMove = ttEntry.getPlay()
		}
	}

	if depth == 0 || g.Playing() != pb.PlayState_PLAYING {
		if g.Playing() == pb.PlayState_PLAYING {
			*impure++
		}
		// Evaluate the state. If the game is still going, play it out greedily
		// from here; otherwise the spread is already final.
		if s.useHeuristics && g.Playing() == pb.PlayState_PLAYING {
			mg := s.stmMovegen
			if thread > 0 {
				mg = s.movegens[thread-1]
			}
			val := s.greedyPlayout(g, mg, pv, nodeKey, thread, pvNode)
			if s.transpositionTableOptim && nodeKey != 0 {
				// Cache the playout, so that revisiting this leaf -- on the next
				// iterative-deepening pass, in a re-search, or through a
				// transposition -- doesn't pay for it again. Only store when
				// there is nothing deeper to clobber; a real search result is
				// always worth more than an estimate. Same guard MAGPIE uses
				// (endgame.c:1957).
				existing := s.ttable.lookup(nodeKey, thread)
				if !existing.valid() || existing.depth() == 0 {
					s.ttable.store(nodeKey, val-int16(ourSpread), TTExact<<6,
						tinymove.InvalidTinyMove, thread)
				}
			}
			*reach = 0
			return val, nil
		}
		// Nothing follows this node, so don't leave a sibling's variation
		// sitting in the caller's line.
		pv.Clear()
		spreadNow := g.SpreadFor(g.PlayerOnTurn())
		*reach = 0
		return int16(spreadNow), nil
	}
	childPV := PVLine{g: g}

	children, fromArena := s.generateSTMPlays(depth, thread)
	if fromArena {
		defer s.arenas[thread].Dealloc(len(children))
	}
	stmRack := g.RackFor(g.PlayerOnTurn())
	if s.currentIDDepths[thread] != depth {
		// If we're not at the top level, assign estimates. Otherwise,
		// we assign the values as estimates in the loop below.
		s.assignEstimates(children, depth, thread, ttMove, nodeKey)
	}

	bestValue := -HugeNumber
	// logIndent := strings.Repeat(" ", max(2*(s.currentIDDepths[thread]-depth), 0))
	if s.logStream != nil {
		// fmt.Fprintf(s.logStream, "  %vplays:\n", logIndent)
	}
	var bestMove tinymove.SmallMove
	var err error
	var moveTiles *[21]tilemapping.MachineLetter

	// At depth==1 the child call immediately evaluates the spread without
	// generating any moves, so cross-sets computed during PlaySmallMove would
	// be discarded by UnplayLastMove without ever being read. Skip them.
	// With heuristics on, though, the leaf does generate moves (for the greedy
	// playout and the stuck-tile scan), so the cross-sets must be valid there.
	playMove := g.PlaySmallMove
	if depth == 1 && !s.useHeuristics {
		playMove = g.PlaySmallMoveNoCrossSet
	}

	// ABDADA: track deferred moves for second pass
	var deferredMoves []int
	// Moves are only ever deferred at depth DEFER_DEPTH and up, so below
	// that there is no point telling the other threads which move this one
	// is searching: it would be two writes to shared memory per child for
	// nobody to read.
	abdadaHere := s.abdadaOptim && depth >= DEFER_DEPTH
	// leavesPlayed counts the children played at depth 1.
	leavesPlayed := 0

	// See keepOneBound: a move that keeps one tile and cannot beat alpha
	// needs no search. Not at the root of a search, which needs each move's
	// own value, and only where the opponent's reply is searched.
	var keepOne keepOneBound
	keepOneActive := s.keepOneOptim && depth >= 2 && s.currentIDDepths[thread] != depth &&
		stmRack.NumTiles() >= 2
	if keepOneActive {
		mg := s.stmMovegen
		if thread > 0 {
			mg = s.movegens[thread-1]
		}
		keepOne.init(g, s.listsFor(thread, g, mg), onTurn, ourSpread)
	}

	// See outReplyBound: a move the opponent can answer by going out, for a
	// result that does not beat alpha, needs no search. Set up only once a
	// first move has failed to cut off.
	var outReply outReplyBound
	outReplyActive := s.outReplyOptim && depth >= 2 && s.currentIDDepths[thread] != depth
	stmTiles := int(stmRack.NumTiles())

	// See the look-ahead in the loop below.
	lookAhead := depth == 1 && s.childShortcuts && s.transpositionTableOptim && nodeKey != 0
	aheadIdx := -1
	aheadKey := uint64(0)

	// First pass: search non-deferred moves
	for idx := range children {
		// checkBound is set when the self-check searches a move a bound
		// would have skipped, to see that it does fail low.
		checkBound := false
		checkAlpha := α
		if outReplyActive && idx > 0 {
			if !outReply.ready {
				mg := s.stmMovegen
				if thread > 0 {
					mg = s.movegens[thread-1]
				}
				if lists := s.listsFor(thread, g, mg); lists != nil {
					outReply.init(g, lists, onTurn, ourSpread)
				} else {
					outReplyActive = false
				}
			}
			// Not a move that goes out, nor a pass that ends the game:
			// neither is replied to.
			m := &children[idx]
			if outReplyActive && m.TilesPlayed() != stmTiles &&
				(!m.IsPass() || g.ScorelessTurns() == 0) {
				if bound, ok := outReply.bound(m, int(α)); ok {
					if !checkMoveLists {
						if int16(bound) > bestValue {
							bestValue = int16(bound)
							bestMove = children[idx]
						}
						s.nodes.add(thread, 1)
						continue
					}
					checkBound = true
				}
			}
		}
		if keepOneActive && !children[idx].IsPass() &&
			children[idx].TilesPlayed() == keepOne.rackSize-1 {
			if bound := keepOne.bound(&children[idx], int(α)); bound <= int(α) {
				if !checkMoveLists {
					if int16(bound) > bestValue {
						bestValue = int16(bound)
						bestMove = children[idx]
					}
					s.nodes.add(thread, 1)
					continue
				}
				checkBound = true
			}
		}
		// ABDADA: check if this move should be deferred
		if abdadaHere && s.abdadaTable.deferMove(nodeKey, children[idx], depth) {
			deferredMoves = append(deferredMoves, idx)
			continue
		}

		// ABDADA: register that we're starting to search this move
		if abdadaHere {
			s.abdadaTable.startingSearch(nodeKey, children[idx], depth)
		}
		if s.logStream != nil {
			// fmt.Fprintf(s.logStream, "  %v- play: %v\n", logIndent, children[idx].ShortDescription())
		}
		// A child that goes out, or a leaf the table already holds, is valued
		// without being played (see valueChildUnplayed).
		// One ply above the leaves nearly every child is looked up in the
		// table and, if it is there, valued without being played; the wait
		// is for the table's memory. So work out the next child's key now
		// and start its bucket loading while this child is dealt with.
		// MAGPIE does the same.
		childKey, haveKey := uint64(0), false
		if lookAhead {
			if aheadIdx == idx {
				childKey, haveKey = aheadKey, true
			}
			if idx+1 < len(children) {
				next := &children[idx+1]
				if !next.IsPass() && next.TilesPlayed() != stmTiles {
					aheadKey = s.childKeyUnplayed(g, next, nodeKey, onTurn)
					aheadIdx = idx + 1
					s.ttable.prefetch(aheadKey)
				}
			}
		}
		value, valued := s.valueChildUnplayed(g, &children[idx], nodeKey, depth, α, β,
			idx == 0 || !s.negascoutOptim, pvNode, ourSpread, onTurn, thread, childKey, haveKey)
		if valued {
			s.nodes.add(thread, 1)
			childPV.Clear()
		} else {
			if depth == 1 && s.useHeuristics {
				// The children are leaves, each played out starting with the
				// other player's best play. Once a second one needs playing
				// out, build that player's list here, so that every further
				// leaf derives its own from this one -- regenerating only
				// what its own move changes -- instead of from the list two
				// moves up.
				leavesPlayed++
				if leavesPlayed == 2 && g.RackFor(1-onTurn).NumTiles() > 1 {
					mg := s.stmMovegen
					if thread > 0 {
						mg = s.movegens[thread-1]
					}
					if lists := s.listsFor(thread, g, mg); lists != nil {
						lists.get(1 - onTurn)
					}
				}
			}
			moveTiles, err = playMove(&children[idx])
			if err != nil {
				return 0, err
			}
			s.nodes.add(thread, 1)
			childKey := uint64(0)
			if s.transpositionTableOptim {
				childKey = s.ttable.Zobrist().AddMove(nodeKey, &children[idx], stmRack, moveTiles,
					onTurn == s.solvingPlayer, g.ScorelessTurns(), g.LastScorelessTurns())
			}
			// negascout
			if idx == 0 || !s.negascoutOptim {
				value, err = s.negamax(ctx, childKey, depth-1, -β, -α, &childPV, thread, pvNode)
				if err != nil {
					g.UnplayLastMove()
					// ABDADA: clean up on error
					if abdadaHere {
						s.abdadaTable.finishedSearch(nodeKey, children[idx])
					}
					return value, err
				}
			} else {
				value, err = s.negamax(ctx, childKey, depth-1, -α-1, -α, &childPV, thread, false)
				if err != nil {
					g.UnplayLastMove()
					// ABDADA: clean up on error
					if abdadaHere {
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
				if abdadaHere {
					s.abdadaTable.finishedSearch(nodeKey, children[idx])
				}
				return value, err
			}
			g.UnplayLastMove()
		}
		maxReach = max(maxReach, *reach+1)

		// ABDADA: finished searching this move
		if abdadaHere {
			s.abdadaTable.finishedSearch(nodeKey, children[idx])
		}
		if s.logStream != nil {
			// fmt.Fprintf(s.logStream, "  %v  value: %v\n", logIndent, value)
		}
		if checkBound && -value > checkAlpha {
			panic(fmt.Sprintf("a bound said %s could not beat %d, but it is worth %d (depth %d)",
				children[idx].ShortDescription(), checkAlpha, -value, depth))
		}
		if -value > bestValue {
			bestValue = -value
			bestMove = children[idx]
			pv.Update(bestMove, &childPV, bestValue-int16(s.initialSpread))
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
			moveTiles, err = playMove(&children[idx])
			if err != nil {
				return 0, err
			}
			s.nodes.add(thread, 1)
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
			maxReach = max(maxReach, *reach+1)
			s.abdadaTable.finishedSearch(nodeKey, children[idx])

			if -value > bestValue {
				bestValue = -value
				bestMove = children[idx]
				pv.Update(bestMove, &childPV, bestValue-int16(s.initialSpread))
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
		pure := *impure == impureAtEntry
		storedDepth := uint8(min(depth, depthMask))
		if pure {
			storedDepth = uint8(maxReach)
		}
		flagAndDepth := flag<<6 + storedDepth
		play := bestMove.TinyMove()
		s.ttable.storeEntry(nodeKey, score, flagAndDepth, play, pure, thread)
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
	*reach = maxReach
	return bestValue, nil

}

func (s *Solver) Solve(ctx context.Context, plies int) (int16, []*move.Move, error) {
	s.busy = true
	defer func() {
		s.busy = false
	}()
	defer s.releasePathLists()
	s.setIncrementalUndo(true)
	defer s.setIncrementalUndo(false)
	s.lineMismatches = 0
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

	// Build a pruned KWG containing only words playable in this endgame
	// position. This shrinks the GADDAG the move generator traverses and
	// typically cuts solve time by ~20%.
	if s.prunedKWGOptim {
		gg := s.stmMovegen.(*movegen.GordonGenerator)
		origGADDAG := gg.GADDAG()
		t0 := time.Now()
		if prunedKWG, err := wordprune.GeneratePrunedKWG(
			s.game.Board(),
			s.game.RackFor(0),
			s.game.RackFor(1),
			origGADDAG,
		); err == nil && prunedKWG != nil {
			gg.SetGADDAG(prunedKWG)
			defer gg.SetGADDAG(origGADDAG)
			if csg, ok := s.game.CrossSetGen().(*cross_set.GaddagCrossSetGenerator); ok {
				origCSGaddag := csg.Gaddag
				csg.Gaddag = prunedKWG
				defer func() { csg.Gaddag = origCSGaddag }()
			}
			fullNodes := len(origGADDAG.Nodes())
			prunedNodes := len(prunedKWG.Nodes())
			log.Info().
				Int("full-nodes", fullNodes).
				Int("pruned-nodes", prunedNodes).
				Int("reduction-pct", 100*(fullNodes-prunedNodes)/fullNodes).
				Dur("build-ms", time.Since(t0)).
				Msg("endgame-using-pruned-kwg")
		}
	}
	defer s.useCrossSetCache()()

	s.ensureArenas()
	log.Debug().Int("plies", plies).Msg("alphabeta-solve-config")
	s.requestedPlies = plies
	s.variations = []PVLine{}
	s.lastCompletedPly = 0

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
		fraction := s.ttableFraction
		if fraction == 0 {
			fraction = s.game.Config().GetFloat64(config.ConfigTtableMemFraction)
		}
		s.ttable.Reset(fraction, s.game.Board().Dim())
	}
	// Invalidate the stuck-fraction memo along with the transposition table.
	s.stuckGen++
	s.game.SetEndgameMode(true)
	defer s.game.SetEndgameMode(false)

	s.initialSpread = s.game.CurrentSpread()
	log.Debug().Msgf("Player %v spread at beginning of endgame: %v (%d)", s.solvingPlayer, s.initialSpread, s.game.ScorelessTurns())
	s.nodes.Store(0)
	var bestV int16
	var bestSeq []*move.Move
	// + 2 since lazysmp can search at a higher ply count, and room on top of
	// that for the greedy leaf playout, which plays moves past the ply limit.
	// Thread copies inherit this length via game.Copy (makeGameCopies runs
	// later). Sized unconditionally so that SetUseHeuristics never has to be
	// called before Solve.
	s.game.SetStateStackLength(plies + 2 + MaxGreedyPlayoutPlies)

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

	// Lines can come back cut short by transposition-table hits; finish them
	// while the table still holds this search.
	tcomplete := time.Now()
	for i := range s.variations {
		s.completeLine(ctx, &s.variations[i])
	}
	s.completeLine(ctx, &s.principalVariation)
	log.Debug().Dur("took", time.Since(tcomplete)).Int("variations", len(s.variations)).
		Msg("completed-variations")
	s.principalVariation.MaterializeFull()
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

// NumSearchedMoves returns how many moves at the front of the sequence returned
// by Solve were found by the search itself. Any moves past that came from the
// greedy playout and are only an estimate of how the game finishes.
func (s *Solver) NumSearchedMoves() int {
	return s.principalVariation.NumSearchedMoves()
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
	// The caller moves the game to another position between solves, which
	// the lists would not see.
	defer s.releasePathLists()
	s.setIncrementalUndo(true)
	defer s.setIncrementalUndo(false)
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
	s.ensureArenas()
	s.requestedPlies = plies

	// tstart := time.Now()
	s.stmMovegen.SetSortingParameter(movegen.SortByNone)
	defer s.stmMovegen.SetSortingParameter(movegen.SortByScore)

	// Build a pruned KWG for this endgame position (same optimisation as Solve).
	if s.prunedKWGOptim {
		ggQD := s.stmMovegen.(*movegen.GordonGenerator)
		origGADDAGQD := ggQD.GADDAG()
		if prunedKWG, err := wordprune.GeneratePrunedKWG(
			s.game.Board(),
			s.game.RackFor(0),
			s.game.RackFor(1),
			origGADDAGQD,
		); err == nil && prunedKWG != nil {
			ggQD.SetGADDAG(prunedKWG)
			defer ggQD.SetGADDAG(origGADDAGQD)
			if csg, ok := s.game.CrossSetGen().(*cross_set.GaddagCrossSetGenerator); ok {
				origCSGaddag := csg.Gaddag
				csg.Gaddag = prunedKWG
				defer func() { csg.Gaddag = origCSGaddag }()
			}
		}
	}
	defer s.useCrossSetCache()()

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
	if s.currentIDDepths == nil {
		s.currentIDDepths = make([]int, 1) // a hack; allocate once per solver lifetime
		s.impureEvents = make([]impureCounter, 1)
	}
	pv := PVLine{g: s.game}
	val, err := s.negamax(ctx, initialHashKey, plies, α, β, &pv, 0, true)
	s.principalVariation = pv
	s.bestPVValue = val - int16(s.initialSpread)

	if !s.skipMaterialize {
		s.principalVariation.MaterializeFull()
	}
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

func (s *Solver) SetSkipMaterialize(v bool) {
	s.skipMaterialize = v
}

func (s *Solver) SetNullWindowOptim(nw bool) {
	s.nullWindowOptim = nw
}

func (s *Solver) SetNegascoutOptim(n bool) {
	s.negascoutOptim = n
}

func (s *Solver) SetPrunedKWGOptim(v bool) {
	s.prunedKWGOptim = v
}

// SetUseHeuristics turns the MAGPIE-derived endgame heuristics on or off. When
// off, the solver behaves exactly as it did before they were added: a static
// spread at the leaf and the older move-ordering estimates.
func (s *Solver) SetUseHeuristics(v bool) {
	s.useHeuristics = v
}

// SetMagpieOrdering switches move ordering to MAGPIE's estimate (static endgame
// equity, plus conservation and build chains when the opponent is stuck).
// Off by default -- see the field comment for the measurements. Only has an
// effect while heuristics are on.
func (s *Solver) SetMagpieOrdering(v bool) {
	s.magpieOrdering = v
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
