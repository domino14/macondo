// Package alphabeta implements an endgame solver using depth-limited
// minimax with alpha-beta pruning.
package alphabeta

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tilemapping"
	"github.com/domino14/macondo/zobrist"
	"github.com/rs/zerolog/log"
)

var (
	ErrEndEarly          = errors.New("ending early") // not an error
	ErrNoEndgameSolution = errors.New("no endgame solution found")
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

const (
	// Infinity is 10 million.
	Infinity = 10000000
	// TwoPlyOppSearchLimit is how many plays to consider for opponent
	// for the evaluation function.
	// XXX not getting used anymore
	TwoPlyOppSearchLimit = 50
	// FutureAdjustment potentially weighs the value of future points
	// less than present points, to allow for the possibility of being
	// blocked. We just make it 1 because our 2-ply evaluation function
	// is reasonably accurate.
	FutureAdjustment = float32(1)

	killersPerPly = 2
)

// Solver implements the minimax + alphabeta algorithm.
type Solver struct {
	zobrist          *zobrist.Zobrist
	stmMovegen       movegen.MoveGenerator
	otsMovegen       movegen.MoveGenerator
	game             *game.Game
	killerCache      map[uint64]*GameNode
	initialSpread    int
	initialTurnNum   int
	maximizingPlayer int // This is the player who we call this function for.

	complexEvaluation    bool
	iterativeDeepeningOn bool
	disablePruning       bool
	rootNode             *GameNode
	// early pass optimization - if opponent passed, examine a pass first,
	// to get an early value for the end of the game.
	earlyPassOptim          bool
	stuckTileOrderOptim     bool
	killerPlayOptim         bool
	firstWinOptim           bool // this is nothing yet
	transpositionTableOptim bool
	// Some helpful variables to avoid big allocations
	// stm: side-to-move  ots: other side
	stmPlayed []bool
	otsPlayed []bool
	// Rectangle lists for side-to-move and other-side
	stmBlockingRects []rect
	otsBlockingRects []rect
	stmRectIndex     int
	otsRectIndex     int

	lastPrincipalVariation      []*move.Move
	lastPrincipalVariationNodes []*GameNode
	currentIDDepth              int
	requestedPlies              int

	config       *config.Config
	skippedPlays int

	// topLevel is the top level of the tree; all the valid moves that I have
	// These never change; we can sort them by valuation for future iterations
	// of iterative deepening.
	topLevel []*GameNode

	// Should we make this a linear array instead and use modulo?
	ttable map[uint64]*TNode
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
	s.zobrist = &zobrist.Zobrist{}
	s.stmMovegen = m1
	s.otsMovegen = m2
	s.game = game
	s.killerCache = make(map[uint64]*GameNode)
	s.iterativeDeepeningOn = true
	s.earlyPassOptim = true
	s.killerPlayOptim = true
	s.firstWinOptim = false
	s.transpositionTableOptim = false

	s.stmPlayed = make([]bool, tilemapping.MaxAlphabetSize+1)
	s.otsPlayed = make([]bool, tilemapping.MaxAlphabetSize+1)
	s.stmBlockingRects = make([]rect, 20)
	s.otsBlockingRects = make([]rect, 25)
	s.config = cfg
	if s.stmMovegen != nil {
		s.stmMovegen.SetGenPass(true)
		s.stmMovegen.SetPlayRecorder(movegen.AllMinimalPlaysRecorder)
	}
	if s.otsMovegen != nil {
		s.otsMovegen.SetGenPass(true)
		s.otsMovegen.SetPlayRecorder(movegen.AllMinimalPlaysRecorder)
	}

	return nil
}

func (s *Solver) clearStuckTables() {
	for i := 0; i < tilemapping.MaxAlphabetSize+1; i++ {
		s.stmPlayed[i] = false
		s.otsPlayed[i] = false
	}
}

// Given the plays and the given rack, return a list of tiles that were
// never played.
func (s *Solver) computeStuck(plays []move.PlayMaker, rack *tilemapping.Rack,
	stuckBitArray []bool) []tilemapping.MachineLetter {

	stuck := []tilemapping.MachineLetter{}
	for _, play := range plays {
		for _, t := range play.Tiles() {
			if t == 0 {
				continue
			}
			idx := t.IntrinsicTileIdx()
			stuckBitArray[idx] = true

		}
	}
	for _, ml := range rack.TilesOn() {
		if !stuckBitArray[ml] {
			// this tile was never played.
			stuck = append(stuck, ml)
		}

	}
	return stuck
}

func leaveAdjustment(myLeave, oppLeave tilemapping.MachineWord,
	myStuck, otherStuck []tilemapping.MachineLetter, ld *tilemapping.LetterDistribution) float32 {
	if len(myStuck) == 0 && len(otherStuck) == 0 {
		// Neither player is stuck so the adjustment is sum(stm)
		// minus 2 * sum(ots). This prioritizes moves as if the side to move
		// can play out in two.
		// XXX: this formula doesn't make sense to me. Change to
		// + instead of - for now.
		// log.Debug().Msgf("Calculating adjustment; myLeave, oppLeave (%v %v)",
		// 	myLeave.UserVisible(tilemapping.EnglishAlphabet()),
		// 	oppLeave.UserVisible(tilemapping.EnglishAlphabet()))
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
		stuckValue := tilemapping.MachineWord(myStuck).Score(ld)
		oppAdjustment = float32(b * stuckValue)
		// Opp can also one-tile me:
		oppAdjustment += c * float32(oppLeave.Score(ld))
		// But I can also one-tile:
		oppAdjustment -= d * float32(myLeave.Score(ld)-stuckValue)
	}
	if len(otherStuck) > 0 {
		// Same as above in reverse. In practice a lot of this will end up
		// nearly canceling out.
		stuckValue := tilemapping.MachineWord(otherStuck).Score(ld)
		myAdjustment = float32(b * stuckValue)
		myAdjustment += c * float32(myLeave.Score(ld))
		myAdjustment -= d * float32(oppLeave.Score(ld)-stuckValue)
	}
	return myAdjustment - oppAdjustment
}

// func (s *Solver) addPass(plays []*move.Move, ponturn int) []*move.Move {
// 	if len(plays) > 0 && plays[0].Action() != move.MoveTypePass {
// 		// movegen doesn't generate a pass move if unneeded (actually, I'm not
// 		// totally sure why). So generate it here, as sometimes a pass is beneficial
// 		// in the endgame.
// 		plays = append([]*move.Move{move.NewPassMove(s.game.RackFor(ponturn).TilesOn(), s.game.Alphabet())}, plays...)
// 	}
// 	return plays
// }

func (s *Solver) generateSTMPlays(parentNode *GameNode, depth int) []*GameNode {
	// STM means side-to-move
	stmRack := s.game.RackFor(s.game.PlayerOnTurn())
	pnot := (s.game.PlayerOnTurn() + 1) % s.game.NumPlayers()
	otherRack := s.game.RackFor(pnot)
	numTilesOnRack := stmRack.NumTiles()
	ld := s.game.Bag().LetterDistribution()

	sideToMovePlays := s.stmMovegen.GenAll(stmRack, false)
	nodes := make([]*GameNode, len(sideToMovePlays))
	for idx := range nodes {
		n := &GameNode{
			MinimalMove: sideToMovePlays[idx].(*move.MinimalMove).Copy(),
			parent:      parentNode,
		}
		// a simple evaluation algorithm.
		if n.TilesPlayed() == int(numTilesOnRack) {
			// don't set knownEnd quite yet. We can let the
			// calculateNValue function do that.
			n.heuristicValue.initialEstimate = float32(n.Score() + 2*otherRack.ScoreOn(ld))
		} else if depth > 2 {
			n.heuristicValue.initialEstimate = float32(n.Score() + 3*n.TilesPlayed())
		} else {
			n.heuristicValue.initialEstimate = float32(n.Score())
		}
		nodes[idx] = n
	}
	if len(nodes) == 1 && nodes[0].Type() == move.MoveTypePass {
		nodes[0].onlyPassPossible = true
	}
	return nodes

	// if stmRack.NumTiles() > 1 && (s.requestedPlies > 1 || parentMove == nil || len(parentMove.Tiles()) == 0) {
	// If opponent just scored and depth is 1, "6-pass" scoring is not available.
	// Skip adding pass if player has an out play ("6-pass" scoring never outperforms an out play).
	// This is more about "don't search a dubious pass subtree" than about memory allocation.
	// if !containsOutPlay(sideToMovePlays, int(numTilesOnRack)) {
	// sideToMovePlays = s.addPass(sideToMovePlays, s.game.PlayerOnTurn())
	// 	}
	// }
	// log.Debug().Msgf("stm plays %v", sideToMovePlays)
	// if !s.complexEvaluation {
	// Static evaluation must be fast and resource-efficient
	// for _, n := range nodes {

	// }
	// return nodes
	// }
	// return nil
	/*

		// log.Debug().Msgf("stm %v (%v), ots %v (%v)",
		// 	s.game.PlayerOnTurn(), stmRack.String(), pnot, otherRack.String())
		s.otsMovegen.SetSortingParameter(movegen.SortByScore)
		defer s.otsMovegen.SetSortingParameter(movegen.SortByNone)
		s.otsMovegen.GenAll(otherRack, false)

		toConsider := len(s.otsMovegen.Plays())
		if TwoPlyOppSearchLimit < toConsider {
			toConsider = TwoPlyOppSearchLimit
		}
		otherSidePlays := s.otsMovegen.Plays()[:toConsider]

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
				var oLeave tilemapping.MachineWord
				blockedAll := true
				for _, o := range otherSidePlays {
					if s.blocks(play, o, board, true) {
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
	*/
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
	nodes := []*GameNode{}
	child := endNode
	for {
		m := new(move.Move)
		m.SetAlphabet(s.game.Alphabet())
		child.MinimalMove.CopyToMove(m)
		seq = append([]*move.Move{m}, seq...)
		nodes = append([]*GameNode{child}, nodes...)
		child = child.parent
		if child == nil || child.MinimalMove == nil {
			break
		}
	}
	s.lastPrincipalVariationNodes = nodes
	return seq
}

// Solve solves the endgame given the current state of s.game, for the
// current player whose turn it is in that state.
func (s *Solver) Solve(ctx context.Context, plies int) (float32, []*move.Move, error) {
	if s.game.Bag().TilesRemaining() > 0 {
		return 0, nil, errors.New("bag is not empty; cannot use endgame solver")
	}
	log.Debug().Int("plies", plies).
		Bool("iterative-deepening", s.iterativeDeepeningOn).
		Bool("complex-evaluation", s.complexEvaluation).
		Msg("alphabeta-solve-config")
	s.requestedPlies = plies
	s.topLevel = nil
	s.ttable = make(map[uint64]*TNode)
	tstart := time.Now()
	s.skippedPlays = 0
	s.zobrist.Initialize(s.game.Board().Dim())
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
		if s.iterativeDeepeningOn {
			log.Debug().Msgf("Using iterative deepening with %v max plies", plies)

			// Generate first layer of moves.
			plays := s.generateSTMPlays(s.rootNode, 0)
			s.topLevel = append(s.topLevel, plays...)

			for p := 1; p <= plies; p++ {
				log.Debug().Msgf("%v %d Spread at beginning of endgame: %v (%d)", s.maximizingPlayer, s.initialTurnNum, s.initialSpread, s.game.ScorelessTurns())
				s.currentIDDepth = p
				bestNode, err := s.nalphabeta(ctx, s.rootNode, initialHashKey, p, float32(-Infinity), float32(Infinity), true)
				if err != nil && err != ErrEndEarly {
					log.Info().AnErr("alphabeta-err", err).Msg("iterative-deepening-on")
					break
				} else {
					bestNodeSoFar = bestNode
					bestV = bestNode.heuristicValue.value
					bestSeq = s.findBestSequence(bestNode)
					s.lastPrincipalVariation = bestSeq

					log.Info().Msgf("-- Spread swing estimate found after %d plies: %f", p, bestV)
					for idx, move := range bestSeq {
						log.Info().Msgf(" %d) %v", idx+1, move.ShortDescription())
					}
					log.Debug().Msgf(" with %d killer plays", len(s.killerCache))
					if err == ErrEndEarly {
						log.Info().Msg("found-win-ending-early")
						break
					}
				}
				// already sorting inside
				// sort.Slice(s.topLevel, func(i, j int) bool {
				// 	return s.topLevel[j].heuristicValue.less(&s.topLevel[i].heuristicValue)
				// })
			}
		} else {
			s.currentIDDepth = 0
			s.lastPrincipalVariation = nil
			bestNode, err := s.nalphabeta(ctx, s.rootNode, initialHashKey, plies, float32(-Infinity), float32(Infinity), true)
			if err != nil && err != ErrEndEarly {
				log.Info().AnErr("alphabeta-err", err).Msg("iterative-deepening-off")
			} else {
				bestNodeSoFar = bestNode
				bestV = bestNode.heuristicValue.value
				bestSeq = s.findBestSequence(bestNode)
				s.lastPrincipalVariation = bestSeq

				fmt.Printf("-- Spread swing estimate found after %d plies: %f", plies, bestV)
				for idx, move := range bestSeq {
					fmt.Printf(" %d) %v", idx+1, move.ShortDescription())
				}
				fmt.Printf(" with %d killer plays", len(s.killerCache))
				if err == ErrEndEarly {
					log.Info().Msg("found-win-ended-early")
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
	log.Info().Msgf("Number of cached killer plays: %d", len(s.killerCache))
	log.Debug().Msgf("Best sequence: (len=%v) %v", len(bestSeq), bestSeq)

	log.Info().
		Float64("time-elapsed-sec", time.Since(tstart).Seconds()).
		Int("passprune-plays", s.skippedPlays).
		Msg("solve-returning")
	return bestV, bestSeq, err
}

// nalphabeta - negamax version of alphabeta minimax pruning
func (s *Solver) nalphabeta(ctx context.Context, node *GameNode, nodeKey uint64,
	depth int, α float32, β float32, maximizingPlayer bool) (*GameNode, error) {

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	αOrig := α
	var ttEntry *TNode
	var ok bool
	if s.transpositionTableOptim {
		ttEntry, ok = s.ttable[nodeKey]
		if ok && int(ttEntry.depth) >= depth {
			if ttEntry.flag == tValid {
				return ttEntry.gameNode, nil
			}
			if ttEntry.flag == tLBound {
				α = max(α, ttEntry.gameNode.heuristicValue.value)
			} else if ttEntry.flag == tUBound {
				β = min(β, ttEntry.gameNode.heuristicValue.value)
			}
			if α >= β {
				return ttEntry.gameNode, nil
			}
		}
	}

	if depth == 0 || s.game.Playing() != pb.PlayState_PLAYING {
		// s.game.Playing() happens if the game is over; i.e. if the
		// parent node  is terminal.
		node.calculateNValue(s)
		if !maximizingPlayer {
			node.heuristicValue.negate()
		}
		return node, nil
	}
	var children []*GameNode

	atTopLevel := depth > 0 && s.currentIDDepth == depth

	if atTopLevel {
		children = s.topLevel
	} else {
		children = s.generateSTMPlays(node, depth)
	}

	s.sortPlaysForConsideration(children, depth, node, nodeKey)

	value := float32(-Infinity)

	var winningNode *GameNode
	// The winningNode will be set to one of these children. The parent
	// of the winning node will always be `node`; this is set in generateSTMPlays
	for _, childNode := range children {

		err := s.game.PlayMove(childNode.MinimalMove, false, 0)
		if err != nil {
			return nil, err
		}

		childKey := s.zobrist.AddMove(nodeKey, childNode.MinimalMove, maximizingPlayer, s.game.ScorelessTurns(), s.game.LastScorelessTurns())
		wn, err := s.nalphabeta(ctx, childNode, childKey, depth-1, -β, -α, !maximizingPlayer)
		if err != nil {
			s.game.UnplayLastMove()
			return wn, err
		}

		s.game.UnplayLastMove()

		// for negamax, take the max of value and the negative wn value.
		if -wn.heuristicValue.value > value {
			value = -wn.heuristicValue.value
			winningNode = wn
			if s.killerPlayOptim {
				s.killerCache[nodeKey] = childNode
			}
		}
		α = max(α, value)
		if α >= β {
			break // beta cut-off
		}

	}
	// wn.Negative() makes a copy. figure out how to allocate less.
	wnn := winningNode.Negative()
	if s.transpositionTableOptim {
		if ttEntry == nil {
			ttEntry = &TNode{}
		}
		if value <= αOrig {
			ttEntry.flag = tUBound
		} else if value >= β {
			ttEntry.flag = tLBound
		} else {
			ttEntry.flag = tValid
		}
		ttEntry.depth = int8(depth)
		ttEntry.gameNode = wnn
		s.ttable[nodeKey] = ttEntry
	}
	//  The negamax node's return value is a heuristic score from the point
	// of view of the node's current player.

	return wnn, nil
}

func (s *Solver) sortPlaysForConsideration(plays []*GameNode, depth int,
	node *GameNode, nodeKey uint64) {
	killerPlay := s.killerCache[nodeKey]

	oppPassed := node.MinimalMove != nil && node.Type() == move.MoveTypePass
	kpFound := false
	for _, play := range plays {
		play.priority = 0
		if s.earlyPassOptim && oppPassed && play.Type() == move.MoveTypePass {
			play.priority = 2
		}
		if killerPlay != nil && s.killerPlayOptim && play.Equals(killerPlay.MinimalMove) {
			play.priority = 1
			kpFound = true
		}
	}
	if s.killerPlayOptim && !kpFound && killerPlay != nil {
		log.Warn().Msgf("zobrist-collision; did not find %v (parent %v, nodekey %v, depth %d)", killerPlay, node, nodeKey, depth)
	}

	// Sort by priority then heuristic value.

	sort.Slice(plays, func(i, j int) bool {
		if plays[j].priority == plays[i].priority {
			return plays[j].heuristicValue.initialEstimate < plays[i].heuristicValue.initialEstimate
		}
		return plays[j].priority < plays[i].priority
	})
}

/*
func (s *Solver) canSkipIfOppStuck(play *move.Move, parent *GameNode, depth int) (bool, error) {
	if !s.stuckTileOrderOptim {
		return false, nil
	}
	// If the opponent is stuck with a tile, it is possible to trim the tree
	// early and avoid analyzing so many moves.
	// We are about to analyze "play".
	if play.Action() == move.MoveTypePass {
		// never skip a pass.
		log.Trace().Msg("cond0-fail")
		return false, nil
	}
	if !parent.onlyPassPossible {
		log.Trace().Msg("cond1-fail")
		return false, nil
	}
	if parent.parent.move == nil {
		log.Trace().Msg("cond2-fail")
		return false, nil
	}
	// If we are here, the opponent's last move was a forced pass, which
	// means they are stuck with one or more tiles.

	// Check: are they stuck if we made "play" instead of our last move?
	// Are they stuck if we make both "play" and our last move?
	// Do "play" and our last move intersect?
	// Does "play" score less than our last move? -- analyze if so
	// skip if "play" scores more
	// If they are the same score, sort alphabetically and only handle
	// the first one alphabetically.
	ourLastMove := parent.parent.move
	forcedPass := parent.move
	// checkpoint-1
	s.game.UnplayLastMove() // unplay opponent's pass
	s.game.UnplayLastMove() // unplay our previous move.
	err := s.game.PlayMove(play, false, 0)
	if err != nil {
		log.Err(err).Msg("skip-1")
		// if this fails there's a serious issue.
		return false, err
	}
	// Generate other side's plays and see if they're stuck.
	hasPlay := s.stmMovegen.AtLeastOneTileMove(s.game.RackFor(s.game.PlayerOnTurn()))

	// Set the play stack to what it used to be (at checkpoint-1),
	// before comparing.
	s.game.UnplayLastMove()
	err = s.game.PlayMove(ourLastMove, false, 0)
	if err != nil {
		log.Err(err).Msg("skip-2")
		return false, err
	}
	err = s.game.PlayMove(forcedPass, false, 0)
	if err != nil {
		log.Err(err).Msg("skip-3")
		return false, err
	}
	// Check if the opp is stuck.
	if hasPlay {
		// opp is not stuck after this play.
		log.Trace().Msg("cond3-fail")
		return false, nil
	}
	// OK, let's try playing this play on our turn and see if the opp
	// is stuck.
	err = s.game.PlayMove(play, false, 0)
	if err != nil {
		log.Err(err).Msg("skip-4")
		return false, err
	}
	hasPlay = s.stmMovegen.AtLeastOneTileMove(s.game.RackFor(s.game.PlayerOnTurn()))

	// rewind the stack again
	s.game.UnplayLastMove()
	// Check if the opp is stuck.
	if hasPlay {
		// opp is not stuck after this play.
		log.Trace().Msg("cond4-fail")
		return false, nil
	}
	// checkpt-2
	// If we are here, opp is stuck in these three situations:
	// - after `ourLastMove` (we already knew this because they had a forced pass)
	// - after `ourLastMove` and `play`
	// - after `play` _instead of_ `ourLastMove`
	// Check if the two plays are "independent". If they block each other
	// they are not independent and we can't skip examining the tree.
	if s.blocks(play, ourLastMove, s.game.Board(), true) {
		log.Trace().Msg("cond5-fail")
		return false, nil
	}
	if play.Score() > ourLastMove.Score() {
		log.Trace().Msg("cond6-fail")
		return false, nil
	} else if play.Score() == ourLastMove.Score() {
		if play.ShortDescription() > ourLastMove.ShortDescription() {
			log.Trace().Msg("cond7-fail")
			return false, nil
		}
	}
	// OK fine. Let's skip examining this branch of the tree.
	log.Trace().Str("play", play.ShortDescription()).Msg("skipping")
	s.skippedPlays++
	return true, nil
}
*/

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

func (s *Solver) SetStuckTileOrderOptim(i bool) {
	s.stuckTileOrderOptim = i
}

func (s *Solver) SetKillerPlayOptim(i bool) {
	s.killerPlayOptim = i
}

func (s *Solver) SetFirstWinOptim(i bool) {
	s.firstWinOptim = i
}

func (s *Solver) SetTranspositionTableOptim(i bool) {
	s.transpositionTableOptim = i
}

func (s *Solver) LastPrincipalVariationNodes() []*GameNode {
	return s.lastPrincipalVariationNodes
}
