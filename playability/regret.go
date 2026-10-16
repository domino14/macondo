// Package playability measures how much knowing a word is worth.
//
// The playability of a set of words S is the equity HastyBot gives up, per
// game, when S is removed from its lexicon and nothing else changes. At one
// position that loss is
//
//	regret(S) = E(best move) - E(best move that forms no word in S)
//
// A move is legal under the lexicon minus S exactly when it is legal under the
// full lexicon and none of the words it forms (the main word and every cross
// word) is in S. So filtering one move list, generated with the full lexicon,
// gives the same answer a pruned word graph would -- hooks included, since the
// hooked word is a cross word of the move that makes it.
//
// Regret is zero unless the best move forms a word in S, so each turn only
// needs work for the few words the best move forms. A word's own value uses
// S = {w}; an alphagram's value uses S = every anagram of it.
package playability

import (
	"sort"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

// Key is a word as raw (unblanked) machine letters, usable as a map key.
type Key string

// AlphaKey returns the alphagram of a word key.
func AlphaKey(k Key) Key {
	b := []byte(k)
	sort.Slice(b, func(i, j int) bool { return b[i] < b[j] })
	return Key(b)
}

// Visible converts a key back to letters.
func Visible(k Key, tm *tilemapping.TileMapping) string {
	return k.MW().UserVisible(tm)
}

// MW converts a key back to a machine word.
func (k Key) MW() tilemapping.MachineWord {
	mw := make(tilemapping.MachineWord, len(k))
	for i := 0; i < len(k); i++ {
		mw[i] = tilemapping.MachineLetter(k[i])
	}
	return mw
}

// KeyOf converts a machine word (unblanking it) to a key.
func KeyOf(mw tilemapping.MachineWord) Key {
	b := make([]byte, len(mw))
	for i, ml := range mw {
		b[i] = byte(ml.Unblank())
	}
	return Key(b)
}

// WordRegret is one word the best move formed, and what not knowing it costs.
type WordRegret struct {
	Word   Key
	Main   bool    // the move's main word, rather than only a cross word
	Regret float64 // regret for S = {Word}
}

// AlphaRegret is one alphagram the best move formed a word of.
type AlphaRegret struct {
	Alpha  Key
	Regret float64 // regret for S = all anagrams of Alpha
}

// Evaluator computes regrets for a sorted move list. It caches the words each
// move forms for the duration of one turn; reuse it across turns.
type Evaluator struct {
	formed [][]Key
	done   []bool
}

func (ev *Evaluator) reset(n int) {
	if cap(ev.done) < n {
		ev.formed = make([][]Key, n)
		ev.done = make([]bool, n)
	}
	ev.formed = ev.formed[:n]
	ev.done = ev.done[:n]
	for i := range ev.done {
		ev.done[i] = false
		ev.formed[i] = ev.formed[i][:0]
	}
}

func (ev *Evaluator) wordsOf(b *board.GameBoard, plays []*move.Move, i int) []Key {
	if ev.done[i] {
		return ev.formed[i]
	}
	ev.done[i] = true
	m := plays[i]
	if m.Action() != move.MoveTypePlay {
		return ev.formed[i]
	}
	ws, err := b.FormedWords(m)
	if err != nil {
		return ev.formed[i]
	}
	for _, w := range ws {
		ev.formed[i] = append(ev.formed[i], KeyOf(w))
	}
	return ev.formed[i]
}

// Regrets returns the per-word and per-alphagram regrets for the best move,
// plays[0]. plays must be sorted by equity, best first, and b must be the board
// the moves would be played onto (before the best move lands). If no move in the
// list avoids a word, the alternative is a pass.
func (ev *Evaluator) Regrets(b *board.GameBoard, plays []*move.Move) ([]WordRegret, []AlphaRegret) {
	if len(plays) == 0 || plays[0].Action() != move.MoveTypePlay {
		return nil, nil
	}
	ev.reset(len(plays))
	best := plays[0].Equity()
	bestWords := ev.wordsOf(b, plays, 0)

	var wrs []WordRegret
	var ars []AlphaRegret
	for wi, w := range bestWords {
		dup := false
		for _, prev := range bestWords[:wi] {
			if prev == w {
				dup = true
			}
		}
		if dup {
			continue
		}
		alt, _ := ev.firstAvoiding(b, plays, func(f Key) bool { return f == w })
		wrs = append(wrs, WordRegret{Word: w, Main: wi == 0, Regret: best - alt})
	}
	for wi, w := range bestWords {
		a := AlphaKey(w)
		dup := false
		for _, prev := range bestWords[:wi] {
			if len(prev) == len(w) && AlphaKey(prev) == a {
				dup = true
			}
		}
		if dup {
			continue
		}
		alt, _ := ev.firstAvoiding(b, plays, func(f Key) bool {
			return len(f) == len(a) && AlphaKey(f) == a
		})
		ars = append(ars, AlphaRegret{Alpha: a, Regret: best - alt})
	}
	return wrs, ars
}

// firstAvoiding returns the equity of the best move none of whose formed words
// is in the excluded set, and false (with a pass's equity) if there is none in
// the list.
func (ev *Evaluator) firstAvoiding(b *board.GameBoard, plays []*move.Move, excluded func(Key) bool) (float64, bool) {
	for i := 1; i < len(plays); i++ {
		ok := true
		for _, f := range ev.wordsOf(b, plays, i) {
			if excluded(f) {
				ok = false
				break
			}
		}
		if ok {
			return plays[i].Equity(), true
		}
	}
	return turnplayer.StaticPassPenalty, false
}

// RegretFor returns the regret of excluding the given set of words (as keys) at
// this position: the best move's equity minus the best equity among moves that
// form none of them. Used by the synthetic estimator, where the target word is
// fixed in advance and may not be in the best move at all (regret 0).
// found is false when no move in the list avoids the set; with a truncated
// (top-N) list the caller should then retry with the full list.
// It reuses the cache, so call Reset when the move list changes.
func (ev *Evaluator) RegretFor(b *board.GameBoard, plays []*move.Move, excluded func(Key) bool) (regret float64, found bool) {
	if len(plays) == 0 || plays[0].Action() != move.MoveTypePlay {
		return 0, true
	}
	hit := false
	for _, f := range ev.wordsOf(b, plays, 0) {
		if excluded(f) {
			hit = true
			break
		}
	}
	if !hit {
		return 0, true
	}
	alt, found := ev.firstAvoiding(b, plays, excluded)
	return plays[0].Equity() - alt, found
}

// Reset clears the per-turn cache for a new move list of n moves.
func (ev *Evaluator) Reset(n int) { ev.reset(n) }
