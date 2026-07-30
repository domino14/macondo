package turnplayer

import (
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// wordSmogPoolSize is the smallest number of plays a WordSmog generation asks
// for. Callers that want a single move (the bots) still need a handful of
// candidates to filter over.
const wordSmogPoolSize = 20

// GenerateWordSmogTopPlays generates the best numPlays moves best-first when
// gen is in WordSmog mode, and reports whether it did.
//
// In WordSmog every permutation of a legal multiset is a distinct legal play,
// so exhaustive generation is not a usable mode: an ordinary opening rack with
// two blanks yields around eight million moves. Shadow bounds each anchor and
// the top-N recorder keeps only what was asked for, which is all any turn
// player ever wanted from the list anyway.
//
// The returned moves are copies: the generator reuses its top-N move objects
// on the next call, and callers (the shell's play list, for one) hold on to
// what they get back.
func GenerateWordSmogTopPlays(gen *movegen.GordonGenerator, rack *tilemapping.Rack,
	addExchange bool, numPlays int) ([]*move.Move, bool) {

	if !gen.IsWordSmog() {
		return nil, false
	}
	pool := max(numPlays, wordSmogPoolSize)
	gen.SetRecordNTopPlays(pool)
	generated := gen.GenAll(rack, addExchange)

	plays := make([]*move.Move, len(generated))
	for i, m := range generated {
		plays[i] = &move.Move{}
		plays[i].CopyFrom(m)
	}
	return plays, true
}
