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

const HugeNumber = float64(1e7)

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

	pvTable        []*move.Move
	currentIDDepth int
	requestedPlies int

	logStream io.Writer

	// Should we make this a linear array instead and use modulo?
	// ttable map[uint64]*TNode
}

type solution struct {
	m     *move.MinimalMove
	score float64
}

func (s *solution) String() string {
	// debug purposes only
	return fmt.Sprintf("<score: %f move: %s>", s.score,
		s.m.ShortDescription(tilemapping.EnglishAlphabet()))
}

// max returns the larger of x or y.
func max(x, y float64) float64 {
	if x < y {
		return y
	}
	return x
}

func min(x, y float64) float64 {
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
		return estimates[i] > estimates[j]
	})

	// if len(nodes) == 1 && nodes[0].Type() == move.MoveTypePass {
	// 	nodes[0].onlyPassPossible = true
	// }
	// return nodes
	return moves
}

func (s *Solver) iterativelyDeepen(ctx context.Context, plies int) error {
	// Generate first layer of moves.
	plays := s.generateSTMPlays(0)
	var err error
	for p := 1; p <= plies; p++ {
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
	α := -HugeNumber
	β := HugeNumber
	bestVal := -HugeNumber
	sols := make([]*solution, len(moves))
	if s.logStream != nil {
		fmt.Fprint(s.logStream, "  plays:\n")
	}

	for idx, m := range moves {
		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "  - play: %v\n", m.ShortDescription(s.game.Alphabet()))
		}

		sol := &solution{m: m, score: -HugeNumber}
		err := s.game.PlayMove(m, false, 0)
		if err != nil {
			return nil, err
		}
		score, err := s.negamax(ctx, plies-1, -β, -α, false)
		if err != nil {
			s.game.UnplayLastMove()
			return nil, err
		}
		sol.score = -score
		s.game.UnplayLastMove()

		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "    value: %v\n", score)
		}

		if sol.score > bestVal {
			bestVal = sol.score
		}
		α = max(α, bestVal)
		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "    α: %v\n", α)
			fmt.Fprintf(s.logStream, "    β: %v\n", β)
		}

		if bestVal >= β {
			break
		}
		// log.Info().Msgf("Tried move %v, sol %f", m.ShortDescription(s.game.Alphabet()), sol.score)
		sols[idx] = sol
	}
	// biggest to smallest
	sort.Slice(sols, func(i, j int) bool {
		return sols[j].score < sols[i].score
	})
	fmt.Println("plies", plies, "found sols", sols)

	return lo.Map(sols, func(item *solution, idx int) *move.MinimalMove {
		return item.m
	}), nil
}

func (s *Solver) evaluate(maximizingPlayer bool) float64 {
	// Evaluate the state.
	initialSpread := s.initialSpread
	// spreadNow is from the POV of the maximizing player
	spreadNow := s.game.PointsFor(s.maximizingPlayer) -
		s.game.PointsFor(1-s.maximizingPlayer)
		// A very simple evaluation function for now. Just the difference in spread,
		// even if the game is not over yet.
	val := float64(spreadNow - initialSpread)
	if !maximizingPlayer {
		return -val
	}
	// gameOver := s.game.Playing() != pb.PlayState_PLAYING
	// val := float64(0)

	// if gameOver {
	// 	// Technically no one is on turn, but the player NOT on turn is
	// 	// the one that just ended the game.
	// 	val = float64(spreadNow - initialSpread)
	// } else {
	// 	// The valuation is already an estimate of the overall gain or loss
	// 	// in spread for this move (if taken to the end of the game).

	// 	// `player` is NOT the one that just made a move.
	// 	ptValue := g.Score()
	// 	moveVal := g.initialEstimate - ptValue
	// 	val = float64(spreadNow) + moveVal - float64(initialSpread)
	// }

	return val
}

func (s *Solver) negamax(ctx context.Context, depth int, α, β float64, maximizingPlayer bool) (float64, error) {
	if ctx.Err() != nil {
		return 0, ctx.Err()
	}
	if depth == 0 || s.game.Playing() != pb.PlayState_PLAYING {
		// s.game.Playing() happens if the game is over; i.e. if the
		// parent node  is terminal.
		// node.calculateNValue(s)
		// if !maximizingPlayer {
		// 	node.heuristicValue.negate()
		// }
		// return node, nil
		return s.evaluate(maximizingPlayer), nil
	}

	children := s.generateSTMPlays(depth)
	value := -HugeNumber
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

		ws, err := s.negamax(ctx, depth-1, -β, -α, !maximizingPlayer)
		if err != nil {
			s.game.UnplayLastMove()
			return ws, err
		}
		s.game.UnplayLastMove()
		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "  %v  value: %v\n", strings.Repeat(" ", indent), ws)
		}
		if -ws > value {
			value = -ws
			// assign winning node?
		}
		α = max(α, value)
		if s.logStream != nil {
			fmt.Fprintf(s.logStream, "  %v  α: %v\n", strings.Repeat(" ", indent), α)
			fmt.Fprintf(s.logStream, "  %v  β: %v\n", strings.Repeat(" ", indent), β)
		}
		if α >= β {
			break // beta cut-off
		}
	}

	return value, nil

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
	var bestSeq []*move.Move

	initialHashKey := s.zobrist.Hash(s.game.Board().GetSquares(),
		s.game.RackFor(s.maximizingPlayer), s.game.RackFor(1-s.maximizingPlayer), false)

	log.Info().Uint64("initialHashKey", initialHashKey).Msg("starting-zobrist-key")
	var wg sync.WaitGroup
	wg.Add(1)

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
	// if bestNodeSoFar != nil {
	// 	log.Debug().Msgf("Best spread found: %v", bestNodeSoFar.heuristicValue.value)
	// } else {
	// 	// This should never happen unless we gave it an absurdly low time or
	// 	// node count?
	// 	err = ErrNoEndgameSolution
	// }
	// Go down tree and find best variation:
	log.Info().Msgf("Number of cached killer plays: %d", len(s.killerCache))
	log.Debug().Msgf("Best sequence: (len=%v) %v", len(bestSeq), bestSeq)

	log.Info().
		Float64("time-elapsed-sec", time.Since(tstart).Seconds()).
		Msg("solve-returning")
	return bestV, bestSeq, err
}
