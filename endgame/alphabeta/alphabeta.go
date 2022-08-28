// Package alphabeta implements an endgame solver using depth-limited
// minimax with alpha-beta pruning.
package alphabeta

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/zobrist"
	"github.com/rs/zerolog/log"
)

// thanks Wikipedia:
/**function alphabeta(node, depth, α, β, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
		for each child of node do
			play(child)
			value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
			unplayLastMove()
            α := max(α, value)
            if α ≥ β then
                break (* β cut-off *)
        return value
    else
        value := +∞
		for each child of node do
			play(child)
			value := min(value, alphabeta(child, depth − 1, α, β, TRUE))
			unplayLastMove()
            β := min(β, value)
            if α ≥ β then
                break (* α cut-off *)
        return value
(* Initial call *)
alphabeta(origin, depth, −∞, +∞, TRUE)
**/

const (
	// Infinity is 10 million.
	Infinity = 10000000
	// TwoPlyOppSearchLimit is how many plays to consider for opponent
	// for the evaluation function.
	TwoPlyOppSearchLimit = 50
	// FutureAdjustment potentially weighs the value of future points
	// less than present points, to allow for the possibility of being
	// blocked. We just make it 1 because our 2-ply evaluation function
	// is reasonably accurate.
	FutureAdjustment = float32(1)
)

var ErrNoEndgameSolution = errors.New("no endgame solution found")

// Solver implements the minimax + alphabeta algorithm.
type Solver struct {
	zobrist          zobrist.Zobrist
	stmMovegen       movegen.MoveGenerator
	otsMovegen       movegen.MoveGenerator
	game             *game.Game
	nodeCache        map[uint64]*GameNode
	totalNodes       int
	initialSpread    int
	initialTurnNum   int
	maximizingPlayer int // This is the player who we call this function for.

	complexEvaluation    bool
	iterativeDeepeningOn bool
	disablePruning       bool
	rootNode             *GameNode
	// Some helpful variables to avoid big allocations
	// stm: side-to-move  ots: other side
	stmPlayed []bool
	otsPlayed []bool
	// Rectangle lists for side-to-move and other-side
	stmBlockingRects []rect
	otsBlockingRects []rect
	stmRectIndex     int
	otsRectIndex     int
	// moveCache        map[int][]*minimalMove
	mmCount int

	lastPrincipalVariation []*move.Move
	currentIDDepth         int

	config *config.Config

	positionHash map[int64]*PositionResult
}

type PositionResult struct {
	move *move.Move
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
func (s *Solver) Init(m1 movegen.MoveGenerator, m2 movegen.MoveGenerator, game *game.Game, cfg *config.Config) error {
	s.zobrist = zobrist.Zobrist{}
	s.stmMovegen = m1
	s.otsMovegen = m2
	s.game = game
	s.nodeCache = make(map[uint64]*GameNode)
	s.totalNodes = 0
	s.iterativeDeepeningOn = true

	s.stmPlayed = make([]bool, alphabet.MaxAlphabetSize+1)
	s.otsPlayed = make([]bool, alphabet.MaxAlphabetSize+1)
	s.stmBlockingRects = make([]rect, 20)
	s.otsBlockingRects = make([]rect, 25)
	s.config = cfg
	return nil
}

func (s *Solver) clearStuckTables() {
	for i := 0; i < alphabet.MaxAlphabetSize+1; i++ {
		s.stmPlayed[i] = false
		s.otsPlayed[i] = false
	}
}

// Given the plays and the given rack, return a list of tiles that were
// never played.
func (s *Solver) computeStuck(plays []*move.Move, rack *alphabet.Rack,
	stuckBitArray []bool) []alphabet.MachineLetter {

	stuck := []alphabet.MachineLetter{}
	for _, play := range plays {
		for _, t := range play.Tiles() {
			idx, ok := t.IntrinsicTileIdx()
			if ok {
				stuckBitArray[idx] = true
			}
		}
	}
	for _, ml := range rack.TilesOn() {
		idx, ok := ml.IntrinsicTileIdx()
		if ok {
			if !stuckBitArray[idx] {
				// this tile was never played.
				stuck = append(stuck, idx)
			}
		}
	}
	return stuck
}

func leaveAdjustment(myLeave, oppLeave alphabet.MachineWord,
	myStuck, otherStuck []alphabet.MachineLetter, ld *alphabet.LetterDistribution) float32 {
	if len(myStuck) == 0 && len(otherStuck) == 0 {
		// Neither player is stuck so the adjustment is sum(stm)
		// minus 2 * sum(ots). This prioritizes moves as if the side to move
		// can play out in two.
		// XXX: this formula doesn't make sense to me. Change to
		// + instead of - for now.
		// log.Debug().Msgf("Calculating adjustment; myLeave, oppLeave (%v %v)",
		// 	myLeave.UserVisible(alphabet.EnglishAlphabet()),
		// 	oppLeave.UserVisible(alphabet.EnglishAlphabet()))
		var adjustment float32
		if len(oppLeave) == 0 {
			// The opponent went out with their play, so our adjustment
			// is negative:
			adjustment = -2 * float32(myLeave.Score(ld))
		} else {
			// Otherwise, the opponent did not go out. We pretend that
			// we are going to go out next turn (for face value, I suppose?),
			// and get twice our opp's rack.
			adjustment = float32(myLeave.Score(ld) + 2*oppLeave.Score(ld))
		}
		return adjustment
	}
	var oppAdjustment, myAdjustment float32
	// Otherwise at least one player is stuck.
	b := 2
	// Opp gets first dibs on next moves, so c > d here
	c := float32(1.75)
	d := float32(1.25)

	if len(myStuck) > 0 && len(otherStuck) > 0 {
		b = 1
	}

	if len(myStuck) > 0 {
		// Opp gets all my tiles
		stuckValue := alphabet.MachineWord(myStuck).Score(ld)
		oppAdjustment = float32(b * stuckValue)
		// Opp can also one-tile me:
		oppAdjustment += c * float32(oppLeave.Score(ld))
		// But I can also one-tile:
		oppAdjustment -= d * float32(myLeave.Score(ld)-stuckValue)
	}
	if len(otherStuck) > 0 {
		// Same as above in reverse. In practice a lot of this will end up
		// nearly canceling out.
		stuckValue := alphabet.MachineWord(otherStuck).Score(ld)
		myAdjustment = float32(b * stuckValue)
		myAdjustment += c * float32(myLeave.Score(ld))
		myAdjustment -= d * float32(oppLeave.Score(ld)-stuckValue)
	}
	return myAdjustment - oppAdjustment
}

func (s *Solver) addPass(plays []*move.Move, ponturn int) []*move.Move {
	if len(plays) > 0 && plays[0].Action() != move.MoveTypePass {
		// movegen doesn't generate a pass move if unneeded (actually, I'm not
		// totally sure why). So generate it here, as sometimes a pass is beneficial
		// in the endgame.
		plays = append(plays, move.NewPassMove(s.game.RackFor(ponturn).TilesOn(), s.game.Alphabet()))
	}
	return plays
}

func (s *Solver) generateSTMPlays(parentMove *move.Move, depth int, plies int) []*move.Move {
	// STM means side-to-move
	stmRack := s.game.RackFor(s.game.PlayerOnTurn())
	pnot := (s.game.PlayerOnTurn() + 1) % s.game.NumPlayers()
	otherRack := s.game.RackFor(pnot)
	numTilesOnRack := stmRack.NumTiles()
	board := s.game.Board()
	ld := s.game.Bag().LetterDistribution()

	sideToMovePlays := s.stmMovegen.GenAll(stmRack, false)
	if stmRack.NumTiles() > 1 && (plies > 1 || parentMove == nil || len(parentMove.Tiles()) == 0) {
		// If opponent just scored and depth is 1, "6-pass" scoring is not available.
		// Skip adding pass if player has an out play ("6-pass" scoring never outperforms an out play).
		// This is more about "don't search a dubious pass subtree" than about memory allocation.
		if !containsOutPlay(sideToMovePlays, int(numTilesOnRack)) {
			sideToMovePlays = s.addPass(sideToMovePlays, s.game.PlayerOnTurn())
		}
	}
	// log.Debug().Msgf("stm plays %v", sideToMovePlays)
	if !s.complexEvaluation {
		// Static evaluation must be fast and resource-efficient
		for _, m := range sideToMovePlays {
			if depth > 2 {
				m.SetValuation(float32(m.Score() + 3*m.TilesPlayed()))
			} else if m.TilesPlayed() == int(numTilesOnRack) {
				m.SetValuation(float32(m.Score() + 2*otherRack.ScoreOn(ld)))
			} else {
				m.SetValuation(float32(m.Score()))
			}
		}
		sort.Slice(sideToMovePlays, func(i, j int) bool {
			// if s.currentIDDepth
			return sideToMovePlays[i].Valuation() > sideToMovePlays[j].Valuation()
		})

		stmCopy := make([]*move.Move, len(sideToMovePlays))
		for idx := range stmCopy {
			stmCopy[idx] = new(move.Move)
			stmCopy[idx].CopyFrom(sideToMovePlays[idx])
		}

		return stmCopy
	}

	// log.Debug().Msgf("stm %v (%v), ots %v (%v)",
	// 	s.game.PlayerOnTurn(), stmRack.String(), pnot, otherRack.String())
	s.otsMovegen.SetSortingParameter(movegen.SortByScore)
	defer s.otsMovegen.SetSortingParameter(movegen.SortByNone)
	s.otsMovegen.GenAll(otherRack, false)

	toConsider := len(s.otsMovegen.Plays())
	if TwoPlyOppSearchLimit < toConsider {
		toConsider = TwoPlyOppSearchLimit
	}
	otherSidePlays := s.addPass(s.otsMovegen.Plays()[:toConsider], pnot)

	// Compute for which tiles we are stuck
	s.clearStuckTables()
	sideToMoveStuck := s.computeStuck(sideToMovePlays, stmRack,
		s.stmPlayed)
	otherSideStuck := s.computeStuck(otherSidePlays, otherRack,
		s.otsPlayed)

	for _, play := range sideToMovePlays {
		// log.Debug().Msgf("Evaluating play %v", play)
		if play.TilesPlayed() == int(numTilesOnRack) {
			// Value is the score of this play plus 2 * the score on
			// opponent's rack (we're going out; general Crossword Game rules)
			play.SetValuation(float32(play.Score() + 2*otherRack.ScoreOn(ld)))
		} else {
			// subtract off the score of the opponent's highest scoring move
			// that is not blocked.
			var oScore int
			var oLeave alphabet.MachineWord
			blockedAll := true
			for _, o := range otherSidePlays {
				if s.blocks(play, o, board) {
					continue
				}
				blockedAll = false
				// log.Debug().Msgf("Highest unblocked play: %v", o)
				oScore = o.Score()
				oLeave = o.Leave()
				break
			}
			if blockedAll {
				// If all the plays are blocked, then the other side's
				// leave is literally all the tiles they have.
				// we also count them as stuck with all their tiles then.
				oLeave = otherRack.TilesOn()
				otherSideStuck = oLeave
			}
			adjust := leaveAdjustment(play.Leave(), oLeave, sideToMoveStuck, otherSideStuck,
				ld)
			// if blockedAll {
			// 	// further reward one-tiling
			// 	adjust *= (float32(numTilesOnRack+1) - float32(play.TilesPlayed()))
			// }

			play.SetValuation(float32(play.Score()-oScore) + FutureAdjustment*adjust)
			// log.Debug().Msgf("Setting evaluation of %v to (%v - %v + %v) = %v",
			// 	play, play.Score(), oScore, adjust, play.Valuation())
		}
	}
	// Sort by valuation.
	sort.Slice(sideToMovePlays, func(i, j int) bool {
		return sideToMovePlays[i].Valuation() > sideToMovePlays[j].Valuation()
	})
	// Finally, we need to allocate here, so that we can save these plays
	// as we change context and recurse. Otherwise, the movegen is still
	// holding on to the slice of moves.
	stmCopy := make([]*move.Move, len(sideToMovePlays))
	for idx := range stmCopy {
		stmCopy[idx] = new(move.Move)
		stmCopy[idx].CopyFrom(sideToMovePlays[idx])
	}
	return stmCopy
}

func containsOutPlay(plays []*move.Move, numTilesOnRack int) bool {
	for _, m := range plays {
		if m.TilesPlayed() == numTilesOnRack {
		        return true
		}
	}
	return false
}

func (s *Solver) findBestSequence(endNode *GameNode) []*move.Move {
	// findBestSequence assumes we have already run alphabeta / iterative deepening
	seq := []*move.Move{}
	child := endNode
	for {
		seq = append([]*move.Move{child.move}, seq...)
		child = child.parent
		if child == nil || child.move == nil {
			break
		}
	}

	return seq
}

// Solve solves the endgame given the current state of s.game, for the
// current player whose turn it is in that state.
func (s *Solver) Solve(plies int) (float32, []*move.Move, error) {
	if s.game.Bag().TilesRemaining() > 0 {
		return 0, nil, errors.New("bag is not empty; cannot use endgame solver")
	}
	log.Debug().Int("plies", plies).
		Bool("iterative-deepening", s.iterativeDeepeningOn).
		Bool("complex-evaluation", s.complexEvaluation).
		Int("maxtimesecs", s.config.AlphaBetaTimeLimit).
		Msg("alphabeta-solve-config")

	tstart := time.Now()
	s.mmCount = 0
	s.zobrist.Initialize(s.game.Board().Dim(), alphabet.MaxAlphabetSize+1, alphabet.MaxAlphabetSize+1)
	// Generate children moves.
	s.stmMovegen.SetSortingParameter(movegen.SortByNone)
	defer s.stmMovegen.SetSortingParameter(movegen.SortByScore)

	// Set max scoreless turns to 2 in the endgame so we don't generate
	// unnecessary sequences of passes.
	s.game.SetMaxScorelessTurns(2)
	defer s.game.SetMaxScorelessTurns(game.DefaultMaxScorelessTurns)

	log.Debug().Msgf("Attempting to solve endgame with %v plies...", plies)

	// technically the children are the actual board _states_ but
	// we don't keep track of those exactly
	s.rootNode = &GameNode{}
	// the root node is basically the board state prior to making any moves.
	// the children of these nodes are the board states after every move.
	// however we treat the children as those actual moves themsselves.

	ctx := context.Background()
	var cancel context.CancelFunc
	if s.config.AlphaBetaTimeLimit > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(s.config.AlphaBetaTimeLimit)*time.Second)
		defer cancel()
	}

	s.initialSpread = s.game.CurrentSpread()
	s.initialTurnNum = s.game.Turn()
	s.maximizingPlayer = s.game.PlayerOnTurn()
	log.Debug().Msgf("Spread at beginning of endgame: %v", s.initialSpread)
	log.Debug().Msgf("Maximizing player is: %v", s.maximizingPlayer)
	var bestV float32
	var bestNodeSoFar *GameNode
	var bestSeq []*move.Move

	var wg sync.WaitGroup
	wg.Add(1)

	go func(ctx context.Context) {
		defer wg.Done()
		if s.iterativeDeepeningOn {

			log.Debug().Msgf("Using iterative deepening with %v max plies", plies)
			for p := 1; p <= plies; p++ {
				log.Debug().Msgf("scoreless turns: %v", s.game.ScorelessTurns())
				log.Debug().Msgf("Spread at beginning of endgame: %v", s.game.CurrentSpread())
				log.Debug().Msgf("Maximizing player is: %v", s.game.PlayerOnTurn())
				s.currentIDDepth = p
				bestValue, bestNode, err := s.alphabeta(ctx, s.rootNode, 0, p, plies, float32(-Infinity), float32(Infinity), true)
				if err != nil {
					log.Err(err).Msg("alphabeta-error")
					break
				} else {
					bestNodeSoFar = bestNode
					bestV = bestValue
					bestSeq = s.findBestSequence(bestNode)
					s.lastPrincipalVariation = bestSeq

					fmt.Printf("-- Spread swing estimate found after %d plies: %f", p, bestV)
					for idx, move := range bestSeq {
						fmt.Printf(" %d) %v", idx+1, move.ShortDescription())
					}
					fmt.Printf("\n")
				}
			}
		} else {
			s.currentIDDepth = 0
			s.lastPrincipalVariation = nil
			bestValue, bestNode, err := s.alphabeta(ctx, s.rootNode, 0, plies, plies, float32(-Infinity), float32(Infinity), true)
			if err != nil {
				log.Err(err).Msg("alphabeta-error")
			} else {
				bestNodeSoFar = bestNode
				bestV = bestValue
				bestSeq = s.findBestSequence(bestNode)
				s.lastPrincipalVariation = bestSeq

				fmt.Printf("-- Spread swing estimate found after %d plies: %f", plies, bestV)
				for idx, move := range bestSeq {
					fmt.Printf(" %d) %v", idx+1, move.ShortDescription())
				}
			}
		}

		log.Debug().Msg("exiting solver goroutine")
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
	log.Debug().Msgf("Size of node cache map: %d", len(s.nodeCache))
	log.Debug().Msgf("Number of expanded nodes: %d", s.totalNodes)
	log.Debug().Msgf("Allocated minimal moves: %d", s.mmCount)

	log.Debug().Msgf("Best sequence: (len=%v) %v", len(bestSeq), bestSeq)
	// for k, v := range s.moveCache {
	// 	fmt.Printf("%d: ", k)
	// 	for _, vv := range v {
	// 		fmt.Printf("[ %s ] ", vv.ShortDescription(s.game.Alphabet()))
	// 	}
	// 	fmt.Printf("\n")
	// }
	log.Info().
		Float64("time-elapsed-sec", time.Since(tstart).Seconds()).
		Msg("solve-returning")
	return bestV, bestSeq, err
}

func (s *Solver) alphabeta(ctx context.Context, parent *GameNode, parentKey uint64,
	depth int, plies int, α float32, β float32, maximizingPlayer bool) (float32, *GameNode, error) {

	select {
	case <-ctx.Done():
		return 0, nil, errors.New("context done -- time limit reached?")
	default:
	}

	if depth == 0 || s.game.Playing() != pb.PlayState_PLAYING {
		// s.game.Playing() happens if the game is over; i.e. if the
		// parent node is terminal.
		parent.calculateValue(s, maximizingPlayer)
		return parent.heuristicValue.value, parent, nil
	}

	if maximizingPlayer {
		// Maximizing
		plays := s.generateSTMPlays(parent.move, depth, plies)
		var maxLeafNode *GameNode
		for _, play := range plays {
			s.game.PlayMove(play, false, 0)
			nodeKey := parentKey ^ s.zobrist.Hash(s.game.Board().GetSquares(), play.Leave(), play.TilesPlayed() == 0)
			node := s.nodeCache[nodeKey]
			nodeValue := α
			// Favor deeper searches
			if !isEnglish(s.game.LexiconName()) || node == nil || node.GetDepth() < uint8(depth-1) {
				node = new(GameNode)
				node.move = play
				node.parent = parent
				node.depth = uint8(depth-1)
				childValue, leaf, err := s.alphabeta(ctx, node, nodeKey, depth-1, plies-1, α, β, false)
				if err != nil {
					s.game.UnplayLastMove()
					return α, nil, err
				}
				nodeValue = childValue
				// TODO: Differentiate between lower/upper/exact bounds somehow
				node.heuristicValue = leaf.heuristicValue
				node = leaf
				s.nodeCache[nodeKey] = node
			} else {
				nodeValue = node.heuristicValue.value
			}
			s.game.UnplayLastMove()

			if maxLeafNode == nil || nodeValue > α {
				maxLeafNode = node
			}
			α = max(α, nodeValue)
			if α >= β {
				return β, maxLeafNode, nil // beta cut-off
			}
		}
		return α, maxLeafNode, nil
	} else {
		// Minimizing
		plays := s.generateSTMPlays(parent.move, depth, plies)
		var minLeafNode *GameNode
		for _, play := range plays {
			s.game.PlayMove(play, false, 0)
			nodeKey := parentKey ^ s.zobrist.Hash(s.game.Board().GetSquares(), play.Leave(), play.TilesPlayed() == 0)
			node := s.nodeCache[nodeKey]
			nodeValue := β
			// Favor deeper searches
			if !isEnglish(s.game.LexiconName()) || node == nil || node.GetDepth() < uint8(depth-1) {
				node = new(GameNode)
				node.move = play
				node.parent = parent
				node.depth = uint8(depth-1)
				childValue, leaf, err := s.alphabeta(ctx, node, nodeKey, depth-1, plies-1, α, β, true)
				if err != nil {
					s.game.UnplayLastMove()
					return β, nil, err
				}
				nodeValue = childValue
				// TODO: Differentiate between lower/upper/exact bounds somehow
				node.heuristicValue = leaf.heuristicValue
				node = leaf
				s.nodeCache[nodeKey] = node
			} else {
				nodeValue = node.heuristicValue.value
			}
			s.game.UnplayLastMove()

			if minLeafNode == nil || nodeValue < β {
				minLeafNode = node
			}
			β = min(β, nodeValue)
			if β <= α {
				return α, minLeafNode, nil // alpha cut-off
			}
		}
		return β, minLeafNode, nil
	}
}

func isEnglish(lexiconName string) bool {
	return strings.HasPrefix(lexiconName, "CSW") ||
		strings.HasPrefix(lexiconName, "TWL") ||
		strings.HasPrefix(lexiconName, "NWL") ||
		strings.HasPrefix(lexiconName, "ECWL") || // obsolete name for CEL
		strings.HasPrefix(lexiconName, "CEL") || // common english words
		strings.HasPrefix(lexiconName, "NSWL")
}

func (s *Solver) SetIterativeDeepening(i bool) {
	s.iterativeDeepeningOn = i
}

func (s *Solver) SetComplexEvaluator(i bool) {
	s.complexEvaluation = i
}

func (s *Solver) SetPruningDisabled(i bool) {
	s.disablePruning = i
}

func (s *Solver) RootNode() *GameNode {
	return s.rootNode
}
