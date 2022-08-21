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

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
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
	stmMovegen       movegen.MoveGenerator
	otsMovegen       movegen.MoveGenerator
	game             *game.Game
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
	moveCache        map[int][]*minimalMove
	mmCount          int

	placeholderMove *move.Move

	config *config.Config
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
	s.stmMovegen = m1
	s.otsMovegen = m2
	s.game = game
	s.totalNodes = 0
	s.iterativeDeepeningOn = true

	s.stmPlayed = make([]bool, alphabet.MaxAlphabetSize+1)
	s.otsPlayed = make([]bool, alphabet.MaxAlphabetSize+1)
	s.stmBlockingRects = make([]rect, 20)
	s.otsBlockingRects = make([]rect, 25)
	s.placeholderMove = new(move.Move)
	if game != nil {
		s.placeholderMove.SetAlphabet(game.Alphabet())
	}
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

func (s *Solver) generateSTMPlays(parent *GameNode) []*move.Move {
	// STM means side-to-move
	stmRack := s.game.RackFor(s.game.PlayerOnTurn())
	pnot := (s.game.PlayerOnTurn() + 1) % s.game.NumPlayers()
	otherRack := s.game.RackFor(pnot)
	numTilesOnRack := stmRack.NumTiles()
	board := s.game.Board()
	ld := s.game.Bag().LetterDistribution()

	s.stmMovegen.GenAll(stmRack, false)
	sideToMovePlays := s.addPass(s.stmMovegen.Plays(), s.game.PlayerOnTurn())
	// log.Debug().Msgf("stm plays %v", sideToMovePlays)
	if !s.complexEvaluation {
		// A simple evaluation function is a very dumb, but fast, function
		// of score and tiles played. /shrug
		for _, m := range s.stmMovegen.Plays() {
			m.SetValuation(float32(m.Score() + 3*m.TilesPlayed()))
		}
		sort.Slice(sideToMovePlays, func(i, j int) bool {
			return sideToMovePlays[i].Valuation() > sideToMovePlays[j].Valuation()
		})
		return sideToMovePlays
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
	// stmCopy := make([]*move.Move, len(sideToMovePlays))
	// for idx := range stmCopy {
	// 	stmCopy[idx] = new(move.Move)
	// 	stmCopy[idx].CopyFrom(sideToMovePlays[idx])
	// }
	return sideToMovePlays
}

func movesMatch(t alphabet.MachineWord, lv alphabet.MachineWord, mm *minimalMove) bool {
	return string(mm.tiles) == string(t) && string(mm.leave) == string(lv)
}

func (s *Solver) playToMinimalMove(p *move.Move) *minimalMove {
	// Create a minimal move, or return one from the cache.
	r, c, v := p.CoordsAndVertical()
	ptiles := p.Tiles()
	pleave := p.Leave()
	hashKey := c + (r << mmRowShift) + (p.Score() << mmScoreShift)
	if v {
		hashKey += (1 << mmVerticalShift)
	}
	extraHash := hashKey
	if len(ptiles) > 0 {
		extraHash = hashKey + (int(ptiles[0]) << 25)
	}
	mvs := s.moveCache[extraHash]
	if len(mvs) == 0 {
		mm := &minimalMove{
			tiles:   ptiles,
			leave:   pleave,
			hashKey: uint32(hashKey),
		}
		s.mmCount += 1
		s.moveCache[extraHash] = []*minimalMove{mm}
		return mm
	}

	for _, existing := range mvs {
		// we already know the coordinates and score match, just check
		// the actual leave and tiles.
		if movesMatch(ptiles, pleave, existing) {
			return existing
		}
	}
	// if we are here nothing matched.
	mm := &minimalMove{
		tiles:   ptiles,
		leave:   pleave,
		hashKey: uint32(hashKey),
	}
	s.mmCount += 1
	s.moveCache[extraHash] = append(s.moveCache[extraHash], mm)
	return mm
}

func (s *Solver) childGenerator(node *GameNode, maximizingPlayer bool) func() *GameNode {

	// log.Debug().Msgf("Trying to generate children for node %v", node)
	var plays []*move.Move
	if node.children == nil {
		plays = s.generateSTMPlays(node)
		// Append a minimal node for every generated play.
		node.children = make([]*GameNode, len(plays))
		for idx, p := range plays {
			s.totalNodes++
			mm := s.playToMinimalMove(p)
			node.children[idx] = &GameNode{
				move:      mm,
				parent:    node,
				valuation: p.Valuation(),
			}
		}
	} else {
		// We should only hit this during iterative deepening.
		sort.Slice(node.children, func(i, j int) bool {
			// If the plays exist already, sort them by value so more
			// promising nodes are visited first. This would happen
			// during iterative deepening.
			// Note: When we are the minimizing player,
			// the heuristic value is negated in the `calculateValue` function
			// in gamenode.go. The heuristic value is always relative to the
			// maximizing player. This is why we flip the less function.
			if !maximizingPlayer {
				i, j = j, i
			}
			// If a node hasn't been evaluated yet, push it to the end of the list.
			// This function should sort from biggest to smallest.
			if node.children[j].heuristicValue == nil {
				return false
			}
			if node.children[i].heuristicValue == nil {
				return true
			}
			return node.children[j].heuristicValue.less(node.children[i].heuristicValue)

		})

	}

	gen := func() func() *GameNode {
		idx := -1
		return func() *GameNode {
			idx++
			if len(plays) == 0 {

				// No plays were generated. This happens during iterative
				// deepening, when we re-use previously generated nodes.
				if idx == len(node.children) {
					// log.Debug().Msgf("no more children of %v to return", node)
					return nil
				}
				return node.children[idx]
			}
			if idx == len(plays) {
				return nil
			}

			// Otherwise, new plays were generated and saved in
			// node.children already.

			return node.children[idx]
		}
	}
	return gen()
}

func (s *Solver) findBestSequence(endNode *GameNode) []*move.Move {
	// findBestSequence assumes we have already run alphabeta / iterative deepening
	seq := []*move.Move{}

	child := endNode
	for {

		m := &move.Move{}
		child.move.CopyToMove(m)
		m.SetAlphabet(s.game.Alphabet())
		// log.Debug().Msgf("Children of %v:", child.parent)
		seq = append([]*move.Move{m}, seq...)
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
		Bool("pruning-disabled", s.disablePruning).
		Int("maxnodes", s.config.AlphaBetaNodeLimit).
		Int("maxtimesecs", s.config.AlphaBetaTimeLimit).
		Msg("alphabeta-solve-config")

	tstart := time.Now()
	s.mmCount = 0
	s.moveCache = make(map[int][]*minimalMove)
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
	var cancel2 context.CancelFunc
	if s.config.AlphaBetaTimeLimit > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(s.config.AlphaBetaTimeLimit)*time.Second)
		defer cancel()
	}
	if s.config.AlphaBetaNodeLimit > 0 {
		ctx, cancel2 = context.WithCancel(ctx)
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
	done := make(chan struct{})
	wg.Add(1)

	go func(ctx context.Context) {
		defer wg.Done()
		if s.iterativeDeepeningOn {
			log.Debug().Msgf("Using iterative deepening with %v max plies", plies)
			for p := 1; p <= plies; p++ {
				log.Debug().Msgf("scoreless turns: %v", s.game.ScorelessTurns())
				log.Debug().Msgf("Spread at beginning of endgame: %v", s.game.CurrentSpread())
				log.Debug().Msgf("Maximizing player is: %v", s.game.PlayerOnTurn())
				bestNode, err := s.alphabeta(ctx, s.rootNode, p, float32(-Infinity), float32(Infinity), true)
				if err != nil {
					log.Err(err).Msg("alphabeta-error")
					break
				} else {
					bestNodeSoFar = bestNode
					bestV = bestNode.heuristicValue.value
					bestSeq = s.findBestSequence(bestNode)

					fmt.Printf("-- Spread swing estimate found after %d plies: %f\n",
						p, bestV)
					fmt.Printf("--> Best seq so far is %v\n\n", bestSeq)
				}
			}
		} else {
			bestNode, err := s.alphabeta(ctx, s.rootNode, plies, float32(-Infinity), float32(Infinity), true)
			if err != nil {
				log.Err(err).Msg("alphabeta-error")
			} else {
				bestV = bestNode.heuristicValue.value
				bestSeq = s.findBestSequence(bestNode)
			}
		}
		if cancel2 != nil {
			log.Debug().Msg("sending <-done")
			done <- struct{}{}
		}
		log.Debug().Msg("exiting solver goroutine")
	}(ctx)

	if s.config.AlphaBetaNodeLimit > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			t := time.NewTicker(time.Second)

			for {
				select {
				case <-t.C:
					if s.totalNodes > s.config.AlphaBetaNodeLimit {
						log.Info().Msg("reached node limit")
						cancel2()
						t.Stop()
					}
					log.Debug().Int("nodes", s.totalNodes).Msg("node-count")

				case <-done:
					log.Debug().Msg("read from <-done")
					t.Stop()
					return
				}
			}

		}()
	}
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
	log.Debug().Msgf("Number of expanded nodes: %d", s.totalNodes)
	log.Debug().Msgf("Size of move cache map: %d", len(s.moveCache))
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

func (s *Solver) alphabeta(ctx context.Context, node *GameNode, depth int, α float32, β float32,
	maximizingPlayer bool) (*GameNode, error) {

	select {
	case <-ctx.Done():
		return nil, errors.New("context done")
	default:
	}

	// depthDbg := strings.Repeat(" ", depth)
	if depth == 0 || s.game.Playing() != pb.PlayState_PLAYING {
		// s.game.Playing() happens if the game is over; i.e. if the
		// current node is terminal.
		node.calculateValue(s)
		// log.Debug().Msgf("ending recursion, depth: %v, playing: %v, node: %v val: %v",
		// 	depth, s.game.Playing(), node.move, val)
		return node, nil
	}

	if maximizingPlayer {
		value := float32(-Infinity)
		var winningNode *GameNode
		iter := s.childGenerator(node, true)
		for child := iter(); child != nil; child = iter() {
			// Play the child
			// log.Debug().Msgf("%vGoing to play move %v", depthDbg, child.move)
			child.move.CopyToMove(s.placeholderMove)
			s.game.PlayMove(s.placeholderMove, false, 0)
			// log.Debug().Msgf("%vState is now %v", depthDbg,
			// s.game.String())
			wn, err := s.alphabeta(ctx, child, depth-1, α, β, false)
			if err != nil {
				s.game.UnplayLastMove()
				return nil, err
			}
			s.game.UnplayLastMove()
			// log.Debug().Msgf("%vAfter unplay, state is now %v", depthDbg, s.game.String())

			if wn.heuristicValue.value > value {
				value = wn.heuristicValue.value
				winningNode = wn
				// log.Debug().Msgf("%vFound a better move: %v (%v)", depthDbg, value, tm)
			}

			if !s.disablePruning {
				α = max(α, value)
				if α >= β {
					break // beta cut-off
				}
			}
		}
		node.heuristicValue = &nodeValue{
			value:          value,
			knownEnd:       winningNode.heuristicValue.knownEnd,
			sequenceLength: winningNode.heuristicValue.sequenceLength}
		return winningNode, nil
	}
	// Otherwise, not maximizing
	value := float32(Infinity)
	var winningNode *GameNode
	iter := s.childGenerator(node, false)
	for child := iter(); child != nil; child = iter() {
		// log.Debug().Msgf("%vGoing to play move %v", depthDbg, child.move)
		child.move.CopyToMove(s.placeholderMove)
		s.game.PlayMove(s.placeholderMove, false, 0)
		// log.Debug().Msgf("%vState is now %v", depthDbg,
		// s.game.String())
		wn, err := s.alphabeta(ctx, child, depth-1, α, β, true)
		if err != nil {
			s.game.UnplayLastMove()
			return nil, err
		}
		s.game.UnplayLastMove()
		// log.Debug().Msgf("%vAfter unplay, state is now %v", depthDbg, s.game.String())
		if wn.heuristicValue.value < value {
			value = wn.heuristicValue.value
			winningNode = wn
			// log.Debug().Msgf("%vFound a worse move: %v (%v)", depthDbg, value, tm)
		}

		if !s.disablePruning {
			β = min(β, value)
			if α >= β {
				break // alpha cut-off
			}
		}
	}
	node.heuristicValue = &nodeValue{
		value:          value,
		knownEnd:       winningNode.heuristicValue.knownEnd,
		sequenceLength: winningNode.heuristicValue.sequenceLength}
	return winningNode, nil
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
