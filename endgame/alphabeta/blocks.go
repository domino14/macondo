package alphabeta

import (
	"fmt"

	"github.com/domino14/macondo/move"
	"github.com/rs/zerolog/log"
)

type rect struct {
	tlx, tly, brx, bry int
}

func (r *rect) set(tlx, tly, brx, bry int) {
	r.tlx, r.tly, r.brx, r.bry = tlx, tly, brx, bry
}
func (r rect) String() string {
	return fmt.Sprintf("<rect (%v, %v) - (%v, %v)>", r.tlx, r.tly, r.brx, r.bry)
}

// Return true if rectangles intersect. Assumes a coordinate system with x going
// up as you move right, and y going up as you move down.
// Assume the "points" here are the centers of the actual tiles.
func rectanglesIntersect(r1, r2 rect) bool {
	return !(r1.tlx > r2.brx || r2.tlx > r1.brx || r1.tly > r2.bry || r2.tly > r1.bry)
}

func (s *Solver) addRectangle(tlx, tly, brx, bry int, stm bool) {
	if stm {
		s.stmBlockingRects[s.stmRectIndex].set(tlx, tly, brx, bry)
		s.stmRectIndex++
	} else {
		s.otsBlockingRects[s.otsRectIndex].set(tlx, tly, brx, bry)
		s.otsRectIndex++
	}
}

func (s *Solver) setBlockingRectangles(play *move.Move, stm bool) {
	tlx := -Infinity
	tly := -Infinity
	brx := -Infinity
	bry := -Infinity

	r, c, vert := play.CoordsAndVertical()
	playLength := len(play.Tiles())
	// Note: for the other side, we only care about rectangles that are
	// on the tile positions themselves. For the side to move, we make
	// bigger rectangles. This is assuming we are calling this algorithm
	// with (side to move play) blocks (other side play).
	// Figure out the orthogonal rectangles first.
	for idx, t := range play.Tiles() {
		addRect := false
		if vert {
			if t.IsPlayedTile() && tly == -Infinity {
				tly = r + idx // start right on the first tile
				if stm {
					tlx = c - 1 // need to account for adjacent squares
				} else {
					tlx = c
				}
			}
			// If we are at the end of a contiguous group (or of the whole
			// word)
			if tly != -Infinity && (!t.IsPlayedTile() || idx == playLength-1) {
				if idx == playLength-1 && t.IsPlayedTile() {
					// end right at the actual play
					bry = r + idx
				} else {
					bry = r + idx - 1 // end right at the contiguous group
				}
				if stm {
					brx = c + 1
				} else {
					brx = c
				}
				addRect = true
			}
		} else {
			// this code sucks :/
			if t.IsPlayedTile() && tlx == -Infinity {
				tlx = c + idx
				if stm {
					tly = r - 1
				} else {
					tly = r
				}
			}
			// If we are at the end of a contiguous group (or of the whole
			// word)
			if tlx != -Infinity && (!t.IsPlayedTile() || idx == playLength-1) {
				if idx == playLength-1 && t.IsPlayedTile() {
					brx = c + idx
				} else {
					brx = c + idx - 1
				}
				if stm {
					bry = r + 1
				} else {
					bry = r
				}
				addRect = true
			}
		}
		if addRect {
			s.addRectangle(tlx, tly, brx, bry, stm)
			tlx = -Infinity
			tly = -Infinity
		}
	}
	// Then make one-square rectangles on word edges, in the direction
	// of the word, if we're on the side to move.
	// We don't need to cover the actual played tiles because the
	// orthogonal rectangles above do this.

	if !stm {
		return
	}

	if vert {
		s.addRectangle(c, r-1, c, r-1, stm)
		s.addRectangle(c, r+playLength, c, r+playLength, stm)
	} else {
		s.addRectangle(c-1, r, c-1, r, stm)
		s.addRectangle(c+playLength, r, c+playLength, r, stm)
	}
}

// Returns whether `play` blocks `other`
func (s *Solver) blocks(play *move.Move, other *move.Move) bool {
	// `play` blocks `other` if any square in `play` occupies any square
	// covered by `other`, or adjacent to it, or hooks onto a word formed
	// by `other`
	s.stmRectIndex = 0
	s.otsRectIndex = 0
	s.setBlockingRectangles(play, true)
	if play.HasDupe() {
		s.setBlockingRectangles(play.Dupe(), true)
	}
	log.Debug().Msgf("Blocking rects for stm play %v: %v", play, s.stmBlockingRects[0:s.stmRectIndex])
	s.setBlockingRectangles(other, false)
	if other.HasDupe() {
		s.setBlockingRectangles(other.Dupe(), false)
	}
	log.Debug().Msgf("Blocking rects for ots play %v: %v", other, s.otsBlockingRects[0:s.otsRectIndex])

	for i := 0; i < s.stmRectIndex; i++ {
		for j := 0; j < s.otsRectIndex; j++ {
			if rectanglesIntersect(s.stmBlockingRects[i], s.otsBlockingRects[j]) {
				log.Debug().Msgf("Rectangles intersect: %v, %v", s.stmBlockingRects[i], s.otsBlockingRects[j])
				return true
			}
		}
	}
	return false

}
