package negamax

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
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

var (
	ErrNoEndgameSolution = errors.New("no endgame solution found")
)

type Solver struct {
	zobrist     *zobrist.Zobrist
	stmMovegen  movegen.MoveGenerator
	game        *game.Game
	killerCache map[uint64]*move.MinimalMove

	initialSpread    int
	initialTurnNum   int
	maximizingPlayer int // This is the player who we call this function for.

	earlyPassOptim          bool
	stuckTileOrderOptim     bool
	killerPlayOptim         bool
	firstWinOptim           bool // this is nothing yet
	transpositionTableOptim bool

	lastPrincipalVariation []*move.Move
	currentIDDepth         int
	requestedPlies         int

	// Should we make this a linear array instead and use modulo?
	// ttable map[uint64]*TNode
}

// max returns the larger of x or y.
func max(x, y float32) float32 {
	if x < y {
		return y
	}
	return x
}

func min(x, y float32) float32 {
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
	s.killerCache = make(map[uint64]*move.MinimalMove)
	s.earlyPassOptim = true
	s.killerPlayOptim = true
	s.firstWinOptim = false
	s.transpositionTableOptim = false

	if s.stmMovegen != nil {
		s.stmMovegen.SetGenPass(true)
		s.stmMovegen.SetPlayRecorder(movegen.AllMinimalPlaysRecorder)
	}
	return nil
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
	estimates := make([]float32, len(genPlays))
	for idx := range genPlays {
		p := genPlays[idx].(*move.MinimalMove).Copy()
		moves[idx] = p
		if p.TilesPlayed() == int(numTilesOnRack) {
			estimates[idx] = float32(p.Score() + 2*otherRack.ScoreOn(ld))
		} else if depth > 2 {
			estimates[idx] = float32(p.Score() + 3*p.TilesPlayed())
		} else {
			estimates[idx] = float32(p.Score())
		}
	}
	sort.Slice(moves, func(i, j int) bool {
		return estimates[i] < estimates[j]
	})

	// if len(nodes) == 1 && nodes[0].Type() == move.MoveTypePass {
	// 	nodes[0].onlyPassPossible = true
	// }
	// return nodes
	return moves
}

func (s *Solver) Solve(ctx context.Context, plies int) (float32, []*move.Move, error) {
	if s.game.Bag().TilesRemaining() > 0 {
		return 0, nil, errors.New("bag is not empty; cannot use endgame solver")
	}
	log.Debug().Int("plies", plies).Msg("alphabeta-solve-config")
	s.requestedPlies = plies
	tstart := time.Now()
	s.zobrist.Initialize(s.game.Board().Dim())
	s.stmMovegen.SetSortingParameter(movegen.SortByNone)
	defer s.stmMovegen.SetSortingParameter(movegen.SortByScore)

	// Set max scoreless turns to 2 in the endgame so we don't generate
	// unnecessary sequences of passes.
	s.game.SetMaxScorelessTurns(2)
	defer s.game.SetMaxScorelessTurns(game.DefaultMaxScorelessTurns)
	s.initialSpread = s.game.CurrentSpread()
	s.initialTurnNum = s.game.Turn()
	s.maximizingPlayer = s.game.PlayerOnTurn()
	log.Debug().Msgf("%v %d Spread at beginning of endgame: %v (%d)", s.maximizingPlayer, s.initialTurnNum, s.initialSpread, s.game.ScorelessTurns())

	var bestV float32
	var bestNodeSoFar *GameNode
	var bestSeq []*move.Move

	initialHashKey := s.zobrist.Hash(s.game.Board().GetSquares(),
		s.game.RackFor(s.maximizingPlayer), s.game.RackFor(1-s.maximizingPlayer), false)

	log.Info().Uint64("initialHashKey", initialHashKey).Msg("starting-zobrist-key")
	var wg sync.WaitGroup
	wg.Add(1)

	go func(ctx context.Context) {
		defer wg.Done()
		log.Debug().Msgf("Using iterative deepening with %v max plies", plies)

		// Generate first layer of moves.
		plays := s.generateSTMPlays(0)

		for p := 1; p <= plies; p++ {

		}
	}(ctx)

	var err error
	wg.Wait()
	if bestNodeSoFar != nil {
		log.Debug().Msgf("Best spread found: %v", bestNodeSoFar.heuristicValue.value)
	} else {
		// This should never happen unless we gave it an absurdly low time or
		// node count?
		err = ErrNoEndgameSolution
	}
	// Go down tree and find best variation:
	log.Info().Msgf("Number of cached killer plays: %d", len(s.killerCache))
	log.Debug().Msgf("Best sequence: (len=%v) %v", len(bestSeq), bestSeq)

	log.Info().
		Float64("time-elapsed-sec", time.Since(tstart).Seconds()).
		Msg("solve-returning")
	return bestV, bestSeq, err
}
