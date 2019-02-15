package xwordgame

import "github.com/domino14/macondo/alphabet"

// A Player plays crossword game.
type Player struct {
	rack        *alphabet.Rack
	rackLetters string // user-visible for ease in logging
	name        string
	points      int
}
