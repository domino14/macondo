package negamax

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/samber/lo"

	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tilemapping"
	"github.com/domino14/macondo/zobrist"
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

const HugeNumber = int16(32767)
const MaxVariantLength = 25
const MaxKillers = 2
const Killer0Offset = 20000
const Killer1Offset = 19000
const EarlyPassOffset = 21000

var (
	ErrNoEndgameSolution = errors.New("no endgame solution found")
)

// Credit: MIT-licensed https://github.com/algerbrex/blunder/blob/main/engine/search.go
type PVLine struct {
	Moves []*move.MinimalMove
	g     *game.Game
	score int16
}

// Clear the principal variation line.
func (pvLine *PVLine) Clear() {
	pvLine.Moves = nil
}

// Update the principal variation line with a new best move,
// and a new line of best play after the best move.
func (pvLine *PVLine) Update(move *move.MinimalMove, newPVLine PVLine, score int16) {
	pvLine.Clear()
	pvLine.Moves = append(pvLine.Moves, move)
	pvLine.Moves = append(pvLine.Moves, newPVLine.Moves...)
	pvLine.score = score
}

// Get the best move from the principal variation line.
func (pvLine *PVLine) GetPVMove() *move.MinimalMove {
	return pvLine.Moves[0]
}

// Convert the principal variation line to a string.
func (pvLine PVLine) String() string {
	var s string
	s = fmt.Sprintf("PV; val %d\n", pvLine.score)
	for i := 0; i < len(pvLine.Moves); i++ {
		s += fmt.Sprintf("%d: %s (%d)\n",
			i+1,
			pvLine.Moves[i].ShortDescription(pvLine.g.Alphabet()),
			pvLine.Moves[i].Score())
	}
	return s
}

// panic if pvline is invalid
func (p PVLine) verify() {
	g := p.g.Copy()
	onturn := g.PlayerOnTurn()
	initSpread := g.SpreadFor(onturn)
	for i := 0; i < len(p.Moves); i++ {
		mc := &move.Move{}
		p.Moves[i].CopyToMove(mc)
		_, err := g.ValidateMove(mc)
		if err != nil {
			fmt.Println("error with pv", p)
			panic(err)
		}
		err = g.PlayMove(mc, false, 0)
		if err != nil {
			panic(err)
		}
	}
	// If the scores don't match, log a warning. This can be because
	// the transposition table cut off the PV.
	if g.SpreadFor(onturn)-initSpread != int(p.score) {
		log.Warn().
			Int("initSpread", initSpread).
			Int("nowSpread", g.SpreadFor(onturn)).
			Int("diffInSpreads", g.SpreadFor(onturn)-initSpread).
			Int16("expectedDiff", p.score).
			Msg("pv-cutoff-spreads-do-not-match")
	}
}

type Solver struct {
	zobrist     *zobrist.Zobrist
	stmMovegen  movegen.MoveGenerator
	game        *game.Game
	gameBackup  *game.Game
	killerCache map[uint64]*move.MinimalMove

	initialSpread      int
	initialTurnNum     int
	solvingPlayer      int // This is the player who we call this function for.
	solverInitialScore int
	oppInitialScore    int

	// earlyPassOptim: if the last move was a pass, try a pass first to end
	// the game. It costs very little to check this case first and results
	// in a modest speed boost.
	earlyPassOptim          bool
	iterativeDeepeningOptim bool
	killerPlayOptim         bool
	firstWinOptim           bool // this is nothing yet
	transpositionTableOptim bool
	principalVariation      PVLine
	bestPVValue             int16

	killers [MaxVariantLength][MaxKillers]*move.MinimalMove

	currentIDDepth int
	requestedPlies int

	logStream io.Writer
}

type solution struct {
	m     *move.MinimalMove
	score int16
}

func (s *solution) String() string {
	// debug purposes only
	return fmt.Sprintf("<score: %d move: %s>", s.score,
		s.m.ShortDescription(tilemapping.EnglishAlphabet()))
}

// max returns the larger of x or y.
func max(x, y int16) int16 {
	if x < y {
		return y
	}
	return x
}

func min(x, y int16) int16 {
	if x < y {
		return x
	}
	return y
}

// Init initializes the solver
func (s *Solver) Init(m movegen.MoveGenerator, game *game.Game) error {
	s.zobrist = &zobrist.Zobrist{}
	s.stmMovegen = m
	s.game = game
	s.earlyPassOptim = true
	// Can't get killer play optimization to work properly; it's typically
	// significantly slower with it on. Oh well.
	s.killerPlayOptim = false
	s.firstWinOptim = false
	s.transpositionTableOptim = true
	s.iterativeDeepeningOptim = true

	if s.stmMovegen != nil {
		s.stmMovegen.SetGenPass(true)
		s.stmMovegen.SetPlayRecorder(movegen.AllMinimalPlaysRecorder)
	}
	return nil
}

type playSorter struct {
	estimates []int16
	moves     []*move.MinimalMove
}

func (p playSorter) Len() int { return len(p.moves) }
func (p playSorter) Swap(i, j int) {
	p.estimates[i], p.estimates[j] = p.estimates[j], p.estimates[i]
	p.moves[i], p.moves[j] = p.moves[j], p.moves[i]
}
func (p playSorter) Less(i, j int) bool {
	return p.estimates[j] < p.estimates[i]
}

func (s *Solver) generateSTMPlays(depth int) []*move.MinimalMove {
	// STM means side-to-move
	stmRack := s.game.RackFor(s.game.PlayerOnTurn())
	pnot := (s.game.PlayerOnTurn() + 1) % s.game.NumPlayers()
	otherRack := s.game.RackFor(pnot)
	numTilesOnRack := stmRack.NumTiles()
	ld := s.game.Bag().LetterDistribution()

	genPlays := s.stmMovegen.GenAll(stmRack, false)

	moves := make([]*move.MinimalMove, len(genPlays))
	estimates := make([]int16, len(genPlays))
	lastMoveWasPass := s.game.ScorelessTurns() > s.game.LastScorelessTurns()
	for idx := range genPlays {
		p := genPlays[idx].(*move.MinimalMove).Copy()
		moves[idx] = p
		if p.TilesPlayed() == int(numTilesOnRack) {
			estimates[idx] = int16(p.Score() + 2*otherRack.ScoreOn(ld))
		} else if depth > 2 {
			estimates[idx] = int16(p.Score() + 3*p.TilesPlayed())
		} else {
			estimates[idx] = int16(p.Score())
		}
		if s.killerPlayOptim {
			// Force killer plays to be searched first.
			if p.Equals(s.killers[depth][0]) {
				estimates[idx] += Killer0Offset
			} else if p.Equals(s.killers[depth][1]) {
				estimates[idx] += Killer1Offset
			}
		}
		if s.earlyPassOptim && lastMoveWasPass && p.Type() == move.MoveTypePass {
			estimates[idx] += EarlyPassOffset
		}
	}
	sorter := &playSorter{estimates: estimates, moves: moves}
	sort.Sort(sorter)

	// if len(nodes) == 1 && nodes[0].Type() == move.MoveTypePass {
	// 	nodes[0].onlyPassPossible = true
	// }
	// return nodes
	return sorter.moves
}

func (s *Solver) iterativelyDeepen(ctx context.Context, plies int) error {
	// Generate first layer of moves.
	plays := s.generateSTMPlays(0)
	var err error
	start := 1
	if !s.iterativeDeepeningOptim {
		start = plies
	}
	for p := start; p <= plies; p++ {
		log.Info().Int("plies", p).Msg("deepening-iteratively")
		s.currentIDDepth = p
		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "- ply: %d\n", p)
		}
		plays, err = s.searchMoves(ctx, plays, p)
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Solver) searchMoves(ctx context.Context, moves []*move.MinimalMove, plies int) ([]*move.MinimalMove, error) {
	// negamax for every move.

	initialHashKey := s.zobrist.Hash(
		s.game.Board().GetSquares(),
		s.game.RackFor(s.solvingPlayer),
		s.game.RackFor(1-s.solvingPlayer),
		false, s.game.ScorelessTurns(),
	)

	log.Info().Uint64("initialHashKey", initialHashKey).Msg("starting-zobrist-key")

	α := -HugeNumber
	β := HugeNumber
	bestValue := -HugeNumber
	sols := make([]*solution, len(moves))
	if s.logStream != nil {
		fmt.Fprint(s.logStream, "  plays:\n")
	}
	pv := PVLine{g: s.gameBackup}
	childPV := PVLine{g: s.gameBackup}
	for idx, m := range moves {
		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "  - play: %v\n", m.ShortDescription(s.game.Alphabet()))
		}

		sol := &solution{m: m, score: -HugeNumber}

		err := s.game.PlayMove(m, false, 0)
		if err != nil {
			return nil, err
		}
		childKey := s.zobrist.AddMove(initialHashKey, m, true, s.game.ScorelessTurns(), s.game.LastScorelessTurns())

		score, err := s.negamax(ctx, childKey, plies-1, -β, -α, &childPV)
		if err != nil {
			s.game.UnplayLastMove()
			return nil, err
		}
		sol.score = -score
		s.game.UnplayLastMove()

		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "    value: %v\n", score)
		}

		if sol.score > bestValue {
			bestValue = sol.score
			pv.Update(m, childPV, sol.score-int16(s.initialSpread))
			s.principalVariation = pv
			s.bestPVValue = sol.score - int16(s.initialSpread)
			fmt.Println(pv.String())
		}
		α = max(α, bestValue)
		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "    α: %v\n", α)
			fmt.Fprintf(s.logStream, "    β: %v\n", β)
		}
		childPV.Clear()
		if bestValue >= β {
			break
		}
		// log.Info().Msgf("Tried move %v, sol %f", m.ShortDescription(s.game.Alphabet()), sol.score)
		sols[idx] = sol
	}
	// biggest to smallest
	sort.Slice(sols, func(i, j int) bool {
		return sols[j].score < sols[i].score
	})

	return lo.Map(sols, func(item *solution, idx int) *move.MinimalMove {
		return item.m
	}), nil
}

func (s *Solver) storeKiller(ply int, move *move.MinimalMove) {
	if !move.Equals(s.killers[ply][0]) {
		s.killers[ply][1] = s.killers[ply][0]
		s.killers[ply][0] = move
	}
}

// Clear the killer moves table.
func (s *Solver) ClearKillers() {
	for ply := 0; ply < MaxVariantLength; ply++ {
		s.killers[ply][0] = nil
		s.killers[ply][1] = nil
	}
}

func (s *Solver) evaluate() int16 {
	// Evaluate the state.
	// A very simple evaluation function for now. Just the difference in spread,
	// even if the game is not over yet.
	spreadNow := s.game.SpreadFor(s.game.PlayerOnTurn())
	return int16(spreadNow)
}

func (s *Solver) negamax(ctx context.Context, nodeKey uint64, depth int, α, β int16, pv *PVLine) (int16, error) {

	if ctx.Err() != nil {
		return 0, ctx.Err()
	}
	onTurn := s.game.PlayerOnTurn()
	ourSpread := s.game.SpreadFor(onTurn)

	// Note: if I return early as in here, the PV might not be complete.
	// (the transposition table is cutting off the iterations)
	// The value should still be correct, though.
	// Something like PVS might do better at keeping the PV intact.
	alphaOrig := α
	if s.transpositionTableOptim {
		ttEntry := globalTranspositionTable.lookup(nodeKey)
		if ttEntry != nil && ttEntry.depth() >= uint8(depth) {
			score := ttEntry.score
			flag := ttEntry.flag()
			// add spread back in; we subtract them when storing.
			score += int16(ourSpread)
			if flag == TTExact {
				return score, nil
			} else if flag == TTLower {
				α = max(α, score)
			} else if flag == TTUpper {
				β = min(β, score)
			}
			if α >= β {
				return score, nil
			}
		}
	}

	if depth == 0 || s.game.Playing() != pb.PlayState_PLAYING {
		return s.evaluate(), nil
	}
	childPV := PVLine{g: s.gameBackup}

	children := s.generateSTMPlays(depth)
	bestValue := -HugeNumber
	indent := 2 * (s.currentIDDepth - depth)
	if s.logStream != nil {
		fmt.Fprintf(s.logStream, "  %vplays:\n", strings.Repeat(" ", indent))
	}
	for _, child := range children {
		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "  %v- play: %v\n", strings.Repeat(" ", indent), child.ShortDescription(s.game.Alphabet()))
		}
		err := s.game.PlayMove(child, false, 0)
		if err != nil {
			return 0, err
		}
		childKey := s.zobrist.AddMove(nodeKey, child, onTurn == s.solvingPlayer,
			s.game.ScorelessTurns(), s.game.LastScorelessTurns())
		value, err := s.negamax(ctx, childKey, depth-1, -β, -α, &childPV)
		if err != nil {
			s.game.UnplayLastMove()
			return value, err
		}
		s.game.UnplayLastMove()
		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "  %v  value: %v\n", strings.Repeat(" ", indent), value)
		}
		if -value > bestValue {
			bestValue = -value
			pv.Update(child, childPV, bestValue-int16(s.initialSpread))
		}
		α = max(α, bestValue)
		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "  %v  α: %v\n", strings.Repeat(" ", indent), α)
			fmt.Fprintf(s.logStream, "  %v  β: %v\n", strings.Repeat(" ", indent), β)
		}
		if bestValue >= β {
			if s.killerPlayOptim {
				s.storeKiller(depth, child)
			}
			break // beta cut-off
		}
		childPV.Clear() // clear the child node's pv for the next child node
	}
	if s.transpositionTableOptim {
		// We store this value without our spread to make it spread-independent.
		// Without this, we need to hash the spread as well and this results
		// in many more TT misses.

		score := bestValue - int16(ourSpread)
		var flag uint8
		entryToStore := TableEntry{
			score: score,
		}
		if bestValue <= alphaOrig {
			flag = TTUpper
		} else if bestValue >= β {
			flag = TTLower
		} else {
			flag = TTExact
		}
		entryToStore.flagAndDepth = flag<<6 + uint8(depth)
		globalTranspositionTable.store(nodeKey, entryToStore)
	}
	return bestValue, nil

}

func (s *Solver) Solve(ctx context.Context, plies int) (int16, []*move.Move, error) {
	if s.game.Bag().TilesRemaining() > 0 {
		return 0, nil, errors.New("bag is not empty; cannot use endgame solver")
	}
	log.Debug().Int("plies", plies).Msg("alphabeta-solve-config")
	s.requestedPlies = plies
	tstart := time.Now()
	s.zobrist.Initialize(s.game.Board().Dim())
	s.stmMovegen.SetSortingParameter(movegen.SortByNone)
	defer s.stmMovegen.SetSortingParameter(movegen.SortByScore)
	if s.transpositionTableOptim {
		globalTranspositionTable.reset(0.25)
	}
	// Set max scoreless turns to 2 in the endgame so we don't generate
	// unnecessary sequences of passes.
	s.game.SetMaxScorelessTurns(2)
	defer s.game.SetMaxScorelessTurns(game.DefaultMaxScorelessTurns)
	s.initialSpread = s.game.CurrentSpread()
	s.solverInitialScore = s.game.PointsFor(s.game.PlayerOnTurn())
	s.oppInitialScore = s.game.PointsFor(s.game.NextPlayer())
	s.initialTurnNum = s.game.Turn()
	s.solvingPlayer = s.game.PlayerOnTurn()
	log.Debug().Msgf("%v %d Spread at beginning of endgame: %v (%d)", s.solvingPlayer, s.initialTurnNum, s.initialSpread, s.game.ScorelessTurns())

	var bestV int16
	var bestSeq []*move.Move
	s.gameBackup = s.game.Copy()
	var wg sync.WaitGroup
	s.ClearKillers()

	go func(ctx context.Context) {
		defer wg.Done()
		log.Debug().Msgf("Using iterative deepening with %v max plies", plies)
		err := s.iterativelyDeepen(ctx, plies)
		if err != nil {
			log.Err(err).Msg("error iteratively deepening")
		}
	}(ctx)

	var err error
	wg.Wait()
	// Go down tree and find best variation:

	bestSeq = make([]*move.Move, len(s.principalVariation.Moves))
	for i := 0; i < len(bestSeq); i++ {
		m := &move.Move{}
		m.SetAlphabet(s.game.Alphabet())
		s.principalVariation.Moves[i].CopyToMove(m)
		bestSeq[i] = m
	}
	bestV = s.bestPVValue
	log.Info().
		Uint64("ttable-created", globalTranspositionTable.created).
		Uint64("ttable-lookups", globalTranspositionTable.lookups).
		Uint64("ttable-hits", globalTranspositionTable.hits).
		Uint64("ttable-t2collisions", globalTranspositionTable.t2collisions).
		Float64("time-elapsed-sec", time.Since(tstart).Seconds()).
		Msg("solve-returning")

	return bestV, bestSeq, err
}
