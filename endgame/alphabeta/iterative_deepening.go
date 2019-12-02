package alphabeta

import (
	"sort"

	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/rs/zerolog/log"
)

func (s *Solver) deepen(maxPlies int) (float32, *move.Move) {
	// Call alphabeta with iterative deepening. Moves should be ranked
	// in valuation order from earlier iterations.
	s.movegen.SetSortingParameter(movegen.SortByNone)
	defer s.movegen.SetSortingParameter(movegen.SortByEquity)
	log.Debug().Msgf("Using iterative deepening with %v max plies", maxPlies)
	n := &gameNode{}
	s.rootNode = n

	s.initialSpread = s.game.CurrentSpread()
	s.maximizingPlayer = s.game.PlayerOnTurn()
	log.Debug().Msgf("Spread at beginning of endgame: %v", s.initialSpread)
	log.Debug().Msgf("Maximizing player is: %v", s.maximizingPlayer)
	var bestV float32
	for p := 1; p <= maxPlies; p++ {
		bestV = s.alphabeta(n, p, float32(-Infinity), float32(Infinity), true)

		// Sort our plays by heuristic value for the next iteration, so that
		// more promising nodes are searched first.
		sort.Slice(s.rootNode.children, func(i, j int) bool {
			return s.rootNode.children[i].heuristicValue >
				s.rootNode.children[j].heuristicValue
		})
		log.Debug().Msgf("Spread swing estimate found after %v plies: %v", p, bestV)
	}
	bestSeq := s.findBestSequence(bestV)
	log.Debug().Msgf("Number of expanded nodes: %v", s.totalNodes)
	log.Debug().Msgf("Best sequence: %v", bestSeq)
	return bestV, bestSeq[0]
}
