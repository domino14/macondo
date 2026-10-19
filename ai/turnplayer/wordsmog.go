package turnplayer

import (
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

const (
	// wordSmogMinPool is the smallest number of plays a WordSmog generation
	// asks for. Callers that want a single move (the bots) still need a
	// handful of candidates to filter over.
	wordSmogMinPool = 20
	// wordSmogMaxPool caps it. The top-N recorder deep-copies its way through
	// the whole array on every insertion, which is fine at 20 and ruinous at
	// scale: the two-blank opening rack takes 2.5s with a pool of 20 and over
	// five minutes with a pool of 20000. Callers asking for "everything"
	// (evalSingleMove passes 100000) get the best 100 instead of a hang.
	wordSmogMaxPool = 100
)

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
	pool := min(max(numPlays, wordSmogMinPool), wordSmogMaxPool)
	if numPlays > pool {
		log.Debug().Int("asked", numPlays).Int("pool", pool).
			Msg("WordSmog cannot enumerate; returning the best plays only")
	}
	gen.SetRecordNTopPlays(pool)
	generated := gen.GenAll(rack, addExchange)

	plays := make([]*move.Move, len(generated))
	for i, m := range generated {
		plays[i] = &move.Move{}
		plays[i].CopyFrom(m)
	}
	return plays, true
}
