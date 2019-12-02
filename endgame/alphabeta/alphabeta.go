// Package alphabeta implements an endgame solver using depth-limited
// minimax with alpha-beta pruning.
package alphabeta

import (
	"sort"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/mechanics"
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
	TwoPlyOppSearchLimit = 30
	// FutureAdjustment potentially weighs the value of future points
	// less than present points, to allow for the possibility of being
	// blocked. We just make it 1 because our 2-ply evaluation function
	// is reasonably accurate.
	FutureAdjustment = float32(1)
)

// Solver implements the minimax + alphabeta algorithm.
type Solver struct {
	movegen          movegen.MoveGenerator
	game             *mechanics.XWordGame
	totalNodes       int
	initialSpread    int
	maximizingPlayer int // This is the player who we call this function for.

	iterativeDeepeningOn bool
	disablePruning       bool
	rootNode             *gameNode
	// Some helpful variables to avoid big allocations
	// stm: side-to-move  ots: other side
	stmPlayed []bool
	otsPlayed []bool
	// Rectangle lists for side-to-move and other-side
	stmBlockingRects []rect
	otsBlockingRects []rect
	stmRectIndex     int
	otsRectIndex     int
}

// a game node has to have enough information to allow the game and turns
// to be reconstructed.
type gameNode struct {
	// the move corresponding to the node is the move that is being evaluated.
	move           *move.Move
	heuristicValue float32
	children       []*gameNode // children should be null until expanded.
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

func (g *gameNode) value(s *Solver, gameOver bool) float32 {
	g.calculateValue(s, gameOver)
	// log.Debug().Msgf("heuristic value of node %p is %v", g, g.heuristicValue)
	return g.heuristicValue
}

func (g *gameNode) calculateValue(s *Solver, gameOver bool) {
	// calculate the heuristic value of this node, and store it.
	// we start with a max node. At 1-ply (and all odd plies), maximizing
	// is always false.
	// log.Debug().Msgf("Need to calculate value for %v. Player on turn %v, maximizing %v", g.move, s.game.PlayerOnTurn(), maximizing)

	// Because calculateValue is called after PlayMove has been called,
	// the "playerOnTurn" is actually not the player who made the move
	// whose value we are calculating.
	opponent := s.game.PlayerOnTurn()
	playerWhoMadeMove := (opponent + 1) % (s.game.NumPlayers())

	// The initial spread is always from the maximizing point of view.
	initialSpread := s.initialSpread
	spreadNow := s.game.PointsFor(playerWhoMadeMove) - s.game.PointsFor(opponent)
	negateHeurVal := false
	if playerWhoMadeMove != s.maximizingPlayer {
		// spreadNow = -spreadNow
		initialSpread = -initialSpread
		negateHeurVal = true
	}

	// If the game is over, the value should just be the spread change.
	if gameOver {
		// Technically no one is on turn, but the player NOT on turn is
		// the one that just ended the game.
		// Note that because of the way we track state, it is the state
		// in the solver right now; that's why the game node doesn't matter
		// right here:
		g.heuristicValue = float32(spreadNow - initialSpread)
	} else {
		// The valuation is already an estimate of the overall gain or loss
		// in spread for this move (if taken to the end of the game).

		// `player` is NOT the one that just made a move.
		ptValue := g.move.Score()
		// don't double-count score; it's already in the valuation:
		moveVal := g.move.Valuation() - float32(ptValue)
		// What is the spread right now? The valuation should be relative
		// to that.
		// log.Debug().Msgf("calculating heur value for %v as %v + %v - %v",
		// 	g.move, spreadNow, moveVal, initialSpread)
		g.heuristicValue = float32(spreadNow) + moveVal - float32(initialSpread)
		// g.heuristicValue = s.game.EndgameSpreadEstimate(player, maximizing) - float32(initialSpread)
		// log.Debug().Msgf("Calculating heuristic value of %v as %v - %v",
		// 	g.move, s.game.EndgameSpreadEstimate(player), float32(initialSpread))
	}
	if negateHeurVal {
		// The maximizing player is always "us" - the player that we are
		// solving the endgame for. So if this not the maximizing node,
		// we want to negate the heuristic value, as it needs to be as
		// negative as possible relative to "us". I know, minimax is
		// hard to reason about, but I think this makes sense. At least
		// it seems to work.
		g.heuristicValue = -g.heuristicValue
		// log.Debug().Msg("Negating since not maximizing player")
	}
}

// Init initializes the solver
func (s *Solver) Init(movegen movegen.MoveGenerator, game *mechanics.XWordGame) {
	s.movegen = movegen
	s.game = game
	s.totalNodes = 0
	s.iterativeDeepeningOn = true

	s.stmPlayed = make([]bool, alphabet.MaxAlphabetSize+1)
	s.otsPlayed = make([]bool, alphabet.MaxAlphabetSize+1)
	s.stmBlockingRects = make([]rect, 20)
	s.otsBlockingRects = make([]rect, 25)
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
	myStuck, otherStuck []alphabet.MachineLetter, bag *alphabet.Bag) float32 {
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
			adjustment = -2 * float32(myLeave.Score(bag))
		} else {
			// Otherwise, the opponent did not go out. We pretend that
			// we are going to go out next turn (for face value, I suppose?),
			// and get twice our opp's rack.
			adjustment = float32(myLeave.Score(bag) + 2*oppLeave.Score(bag))
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
		stuckValue := alphabet.MachineWord(myStuck).Score(bag)
		oppAdjustment = float32(b * stuckValue)
		// Opp can also one-tile me:
		oppAdjustment += c * float32(oppLeave.Score(bag))
		// But I can also one-tile:
		oppAdjustment -= d * float32(myLeave.Score(bag)-stuckValue)
	}
	if len(otherStuck) > 0 {
		// Same as above in reverse. In practice a lot of this will end up
		// nearly canceling out.
		stuckValue := alphabet.MachineWord(otherStuck).Score(bag)
		myAdjustment = float32(b * stuckValue)
		myAdjustment += c * float32(myLeave.Score(bag))
		myAdjustment -= d * float32(oppLeave.Score(bag)-stuckValue)
	}
	return myAdjustment - oppAdjustment
}

func (s *Solver) addPass(plays []*move.Move, ponturn int) []*move.Move {
	if len(plays) > 0 && plays[0].Action() != move.MoveTypePass {
		// movegen doesn't generate a pass move if unneeded (actually, I'm not
		// totally sure why). So generate it here, as sometimes a pass is beneficial
		// in the endgame.
		plays = append(plays, move.NewPassMove(s.game.RackFor(ponturn).TilesOn()))
	}
	return plays
}

func (s *Solver) generateSTMPlays(maximizingPlayer bool) []*move.Move {
	// log.Debug().Msgf("Generating stm plays for maximizing %v", maximizingPlayer)
	stmRack := s.game.RackFor(s.game.PlayerOnTurn())
	pnot := (s.game.PlayerOnTurn() + 1) % s.game.NumPlayers()
	otherRack := s.game.RackFor(pnot)
	numTilesOnRack := stmRack.NumTiles()
	board := s.game.Board()
	bag := s.game.Bag()
	s.movegen.GenAll(stmRack)
	sideToMovePlays := s.addPass(s.movegen.Plays(), s.game.PlayerOnTurn())

	// log.Debug().Msgf("stm %v (%v), ots %v (%v)",
	// 	s.game.PlayerOnTurn(), stmRack.String(), pnot, otherRack.String())

	s.movegen.SetSortingParameter(movegen.SortByScore)
	defer s.movegen.SetSortingParameter(movegen.SortByNone)
	s.movegen.GenAll(otherRack)

	toConsider := len(s.movegen.Plays())
	if TwoPlyOppSearchLimit < toConsider {
		toConsider = TwoPlyOppSearchLimit
	}
	otherSidePlays := s.addPass(s.movegen.Plays()[:toConsider], pnot)

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
			play.SetValuation(float32(play.Score() + 2*otherRack.ScoreOn(bag)))
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
				// XXX: we also count them as stuck with all their tiles then.
				oLeave = otherRack.TilesOn()
				otherSideStuck = oLeave
			}
			adjust := leaveAdjustment(play.Leave(), oLeave, sideToMoveStuck, otherSideStuck,
				bag)

			play.SetValuation(float32(play.Score()-oScore) + FutureAdjustment*adjust)
			// log.Debug().Msgf("Setting evaluation of %v to (%v - %v + %v) = %v",
			// 	play, play.Score(), oScore, adjust, play.Valuation())
		}
	}
	// Finally sort by valuation.
	sort.Slice(sideToMovePlays, func(i, j int) bool {
		return sideToMovePlays[i].Valuation() > sideToMovePlays[j].Valuation()
	})
	return sideToMovePlays
}

func (s *Solver) findBestSequence(moveVal float32) []*move.Move {
	// findBestSequence assumes we have already run alphabeta / iterative deepening
	seq := []*move.Move{}
	parent := s.rootNode
	for {
		// parent.printChildren()
		for _, child := range parent.children {
			// log.Debug().Msgf("Found heuristic value for child: %v (%v)", child.move,
			// 	child.heuristicValue)
			if child.heuristicValue == moveVal {
				seq = append(seq, child.move)
				parent = child
				break
			}
		}
		if len(parent.children) == 0 || parent.children == nil {
			break
		}
	}
	return seq
}

// Solve solves the endgame given the current state of s.game, for the
// current player whose turn it is in that state.
func (s *Solver) Solve(plies int) (float32, *move.Move) {
	// Generate children moves.
	s.movegen.SetSortingParameter(movegen.SortByNone)
	defer s.movegen.SetSortingParameter(movegen.SortByEquity)
	log.Debug().Msgf("Attempting to solve endgame with %v plies...", plies)

	// technically the children are the actual board _states_ but
	// we don't keep track of those exactly
	n := &gameNode{}
	s.rootNode = n
	// the root node is basically the board state prior to making any moves.
	// the children of these nodes are the board states after every move.
	// however we treat the children as those actual moves themsselves.

	s.initialSpread = s.game.CurrentSpread()
	s.maximizingPlayer = s.game.PlayerOnTurn()
	log.Debug().Msgf("Spread at beginning of endgame: %v", s.initialSpread)
	log.Debug().Msgf("Maximizing player is: %v", s.maximizingPlayer)
	var bestV float32
	// XXX: We're going to need some sort of channel here to control
	// deepening and propagate results.
	if s.iterativeDeepeningOn {
		log.Debug().Msgf("Using iterative deepening with %v max plies", plies)
		for p := 1; p <= plies; p++ {
			bestV = s.alphabeta(n, p, float32(-Infinity), float32(Infinity), true)

			// Sort our plays by heuristic value for the next iteration, so that
			// more promising nodes are searched first.
			sort.Slice(s.rootNode.children, func(i, j int) bool {
				return s.rootNode.children[i].heuristicValue >
					s.rootNode.children[j].heuristicValue
			})
			log.Debug().Msgf("Spread swing estimate found after %v plies: %v", p, bestV)
		}
	} else {
		bestV = s.alphabeta(n, plies, float32(-Infinity), float32(Infinity), true)
	}
	log.Debug().Msgf("Best spread found: %v", bestV)
	// Go down tree and find best variation:
	bestSeq := s.findBestSequence(bestV)
	log.Debug().Msgf("Number of expanded nodes: %v", s.totalNodes)
	log.Debug().Msgf("Best sequence: %v", bestSeq)

	return bestV, bestSeq[0]
}

func (s *Solver) alphabeta(node *gameNode, depth int, α float32, β float32,
	maximizingPlayer bool) float32 {

	// depthDbg := strings.Repeat(" ", depth)

	if depth == 0 || !s.game.Playing() {
		// s.game.Playing() happens if the game is over; i.e. if the
		// current node is terminal.
		val := node.value(s, !s.game.Playing())
		// log.Debug().Msgf("ending recursion, depth: %v, playing: %v, node: %v val: %v",
		// 	depth, s.game.Playing(), node.move, val)
		return val
	}

	var plays []*move.Move
	if node.children == nil {
		plays = s.generateSTMPlays(maximizingPlayer)
	} else {
		sort.Slice(node.children, func(i, j int) bool {
			// If the plays exist already, sort them by value so more
			// promising nodes are visited first. This would happen
			// during iterative deepening.
			if maximizingPlayer {
				return node.children[i].heuristicValue > node.children[j].heuristicValue
			}
			return node.children[j].heuristicValue > node.children[i].heuristicValue
		})
	}
	childGenerator := func() func() (*gameNode, bool, bool) {
		idx := -1
		return func() (*gameNode, bool, bool) {
			idx++

			if len(plays) == 0 {
				// No plays were generated. This happens during iterative
				// deepening, when we re-use previously generated nodes.
				if idx == len(node.children) {
					return nil, false, false
				}
				return node.children[idx], true, false
			}
			if idx == len(plays) {
				return nil, false, false
			}
			s.totalNodes++
			return &gameNode{move: plays[idx]}, true, true
		}
	}

	if maximizingPlayer {
		value := float32(-Infinity)
		iter := childGenerator()
		for child, ok, newNode := iter(); ok; child, ok, newNode = iter() {
			// Play the child
			// log.Debug().Msgf("%vGoing to play move %v", depthDbg, child.move)
			s.game.PlayMove(child.move, true)
			// log.Debug().Msgf("%vState is now %v", depthDbg,
			// s.game.String())
			v := s.alphabeta(child, depth-1, α, β, false)
			s.game.UnplayLastMove()
			// log.Debug().Msgf("%vAfter unplay, state is now %v", depthDbg, s.game.String())
			if v > value {
				value = v
				// log.Debug().Msgf("%vFound a better move: %v (%v)", depthDbg, value, tm)
			}
			if !s.disablePruning {
				α = max(α, value)
				if α >= β {
					break // beta cut-off
				}
			}
			if newNode {
				node.children = append(node.children, child)
			}
		}
		node.heuristicValue = value
		return value
	}
	// Otherwise, not maximizing
	value := float32(Infinity)
	iter := childGenerator()
	for child, ok, newNode := iter(); ok; child, ok, newNode = iter() {
		// log.Debug().Msgf("%vGoing to play move %v", depthDbg, child.move)
		s.game.PlayMove(child.move, true)
		// log.Debug().Msgf("%vState is now %v", depthDbg,
		// s.game.String())
		v := s.alphabeta(child, depth-1, α, β, true)
		s.game.UnplayLastMove()
		// log.Debug().Msgf("%vAfter unplay, state is now %v", depthDbg, s.game.String())
		if v < value {
			value = v
			// log.Debug().Msgf("%vFound a worse move: %v (%v)", depthDbg, value, tm)
		}
		if !s.disablePruning {
			β = min(β, value)
			if α >= β {
				break // alpha cut-off
			}
		}
		if newNode {
			node.children = append(node.children, child)
		}
	}
	node.heuristicValue = value
	return value
}
