package alphabeta

import "github.com/rs/zerolog/log"

func (g *GameNode) printChildren() {
	log.Debug().Msgf("Printing children of gamenode for move (%v)", g.move)
	for _, c := range g.children {
		log.Debug().Msgf("Child %v (hval=%v)", c.move, c.heuristicValue)
		for _, d := range c.children {
			log.Debug().Msgf("  Child %v (hval=%v)", d.move, d.heuristicValue)
		}
	}
}
