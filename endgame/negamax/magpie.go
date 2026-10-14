package negamax

import (
	"time"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/tinymove"
)

// maxSmallPlays is the most plays one endgame position can generate that a
// generation keeps.
const maxSmallPlays = 200000

// enterMagpie readies the solver's game for a solve: in C-board mode, with
// MAGPIE generating from a word graph pruned to the words playable in this
// endgame (when that optimization is on). The returned function undoes it.
func (s *Solver) enterMagpie() (func(), error) {
	if s.prunedKWGOptim {
		if err := s.game.SetMagpieBoard(true); err != nil {
			return nil, err
		}
		pos, err := s.game.MagpiePosition()
		if err != nil {
			return nil, err
		}
		t0 := time.Now()
		s.prunedKWG = pos.PrunedKWG()
		log.Debug().Dur("build-ms", time.Since(t0)).Msg("endgame-using-pruned-kwg")
	}
	if err := s.magpieGame(s.game); err != nil {
		return nil, err
	}
	return func() {
		if pos, err := s.game.MagpiePosition(); err == nil {
			// Back to full cross-sets before the Go board is refreshed.
			pos.SetOverrideKWG(nil)
		}
		s.game.SetMagpieBoard(false)
		s.prunedKWG.Close()
		s.prunedKWG = nil
	}, nil
}

// magpieGame puts g, the solver's game or a thread's copy of it, in C-board
// mode generating from the pruned word graph.
func (s *Solver) magpieGame(g *game.Game) error {
	if err := g.SetMagpieBoard(true); err != nil {
		return err
	}
	pos, err := g.MagpiePosition()
	if err != nil {
		return err
	}
	pos.SetOverrideKWG(s.prunedKWG)
	return nil
}

// smallPlays generates every play, a pass included, for player (-1 for the
// player on turn) in g, on thread's buffers: the result is reused by the
// thread's next call.
func (s *Solver) smallPlays(g *game.Game, thread, player int) []tinymove.SmallMove {
	if thread >= len(s.smallBufs) {
		// Only outside a solve (heuristics called on their own, in tests);
		// a solve makes the buffers up front in ensureArenas.
		s.ensureArenas()
	}
	pos, err := g.MagpiePosition()
	if err != nil {
		panic(err)
	}
	gen := pos.GenerateSmallFor(player, true, s.smallBufs[thread])
	out := s.smallOut[thread][:0]
	for i := range gen {
		if gen[i].TM == 0 {
			out = append(out, tinymove.PassMove())
			continue
		}
		out = append(out, tinymove.TilePlayMove(tinymove.TinyMove(gen[i].TM),
			gen[i].Score, gen[i].TilesPlayed, gen[i].PlayLength))
	}
	s.smallOut[thread] = out
	return out
}
