package negamax

import (
	"context"
	"time"

	"github.com/rs/zerolog/log"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// completeLineTimeout bounds the extra searching completeLine does for one
// variation. With the transposition table still warm from the main search
// these re-searches are usually instant.
const completeLineTimeout = 10 * time.Second

// completeLine extends a variation that stops before the game is over.
//
// A line can be cut short even though its value accounts for the whole game:
// when negamax hits an exact transposition-table entry at a non-PV node it
// returns the stored value without the moves that produced it, so the parent
// records its move followed by nothing. Under ABDADA and lazy SMP this happens
// constantly, since other threads fill the table. The value is right, but the
// line no longer adds up to it -- and the greedy playout that would have been
// shown, marked as estimated, is lost too.
//
// To recover, replay the line and, from where it stops, search again with the
// depth the main search had left at that point. That re-search is a PV node
// with a full window, so it always yields at least its best move; repeat until
// the line reaches the search horizon. Past the horizon, finish with the same
// greedy playout the leaf evaluation uses, marked as estimated -- exactly what
// an untruncated line would have shown.
//
// Must be called with s.game at the root, after the search has finished and
// before the transposition table is reset.
func (s *Solver) completeLine(ctx context.Context, pv *PVLine) {
	if pv.numMoves == 0 || pv.numSearched < pv.numMoves {
		// Empty, or it already ends in a greedy playout, which runs until the
		// game is over.
		return
	}
	g := s.game
	searchLength := pv.numMoves
	played := 0
	defer func() {
		for i := 0; i < played; i++ {
			g.UnplayLastMove()
		}
	}()
	play := func(i int) bool {
		if g.StateStackRemaining() <= 0 {
			return false
		}
		if _, err := g.PlaySmallMove(&pv.tinyMoves[i]); err != nil {
			log.Warn().Err(err).Int("idx", i).Msg("complete-line-replay-failed")
			return false
		}
		played++
		return true
	}
	for i := 0; i < pv.numMoves; i++ {
		if g.Playing() != pb.PlayState_PLAYING || !play(i) {
			return
		}
	}

	horizon := s.lastCompletedPly
	if horizon == 0 {
		horizon = s.requestedPlies
	}
	// Keep going even if the solve itself was cancelled or timed out; a line
	// that stops mid-game is the thing being fixed here.
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), completeLineTimeout)
	defer cancel()
	// A sentinel no node depth can equal, so that negamax treats every node of
	// the re-search as an interior one: it generates and orders moves itself
	// instead of reaching for the root's precomputed move list.
	savedIDDepth := s.currentIDDepths[0]
	s.currentIDDepths[0] = MaxVariantLength + 1
	defer func() { s.currentIDDepths[0] = savedIDDepth }()

	for g.Playing() == pb.PlayState_PLAYING && pv.numMoves < MaxVariantLength &&
		pv.numSearched == pv.numMoves && (pv.numMoves < horizon || s.useHeuristics) {

		depth := horizon - pv.numMoves
		if depth <= 0 {
			// Past the horizon. A depth-0 node is the greedy playout.
			depth = 0
		}
		sub := PVLine{g: g}
		var key uint64
		if s.transpositionTableOptim {
			key = s.positionKey()
		}
		if _, err := s.negamax(ctx, key, depth, -HugeNumber, HugeNumber, &sub, 0, true); err != nil {
			log.Warn().Err(err).Int("at-move", pv.numMoves).Msg("complete-line-search-failed")
			break
		}
		if sub.numMoves == 0 {
			break
		}
		start := pv.numMoves
		n := min(sub.numMoves, MaxVariantLength-start)
		copy(pv.tinyMoves[start:start+n], sub.tinyMoves[:n])
		pv.numMoves += n
		pv.numSearched += min(sub.numSearched, n)
		for i := start; i < pv.numMoves; i++ {
			if g.Playing() != pb.PlayState_PLAYING || !play(i) {
				break
			}
		}
	}

	if g.Playing() == pb.PlayState_PLAYING || pv.numSearched < pv.numMoves {
		return
	}
	// Searched all the way to the end of the game: the line should now add up
	// to exactly its value.
	if got := int16(g.SpreadFor(s.solvingPlayer) - s.initialSpread); got != pv.score {
		s.lineMismatches++
		line := ""
		for i := 0; i < pv.numMoves; i++ {
			line += pv.tinyMoves[i].ShortDescription() + "; "
		}
		log.Warn().Int16("value", pv.score).Int16("line-adds-to", got).
			Int("moves-from-search", searchLength).Int("horizon", horizon).
			Str("line", line).
			Msg("completed-endgame-line-disagrees-with-value")
	}
}

// positionKey hashes s.game's current position from scratch, the same way the
// search hashes the root and then updates it move by move.
func (s *Solver) positionKey() uint64 {
	g := s.game
	return s.ttable.Zobrist().Hash(
		g.Board().GetSquares(),
		g.RackFor(s.solvingPlayer),
		g.RackFor(1-s.solvingPlayer),
		g.PlayerOnTurn() != s.solvingPlayer,
		g.ScorelessTurns(),
	)
}
