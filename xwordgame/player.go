package xwordgame

import "github.com/domino14/macondo/movegen"

// A Player plays crossword game.
type Player struct {
	rack        *movegen.Rack
	rackLetters string // user-visible for ease in logging
	name        string
	points      int
}
