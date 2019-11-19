package alphabeta

import (
	"fmt"

	"github.com/domino14/macondo/move"
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
	// Figure out the orthogonal rectangles first.
	for idx, t := range play.Tiles() {
		addRect := false
		if vert {
			if t.IsPlayedTile() && tly == -Infinity {
				tly = r + idx // start right on the first tile
				tlx = c - 1   // need to account for adjacent squares
			}
			// If we are at the end of a contiguous group (or of the whole
			// word)
			if tly != -Infinity && (!t.IsPlayedTile() || idx == playLength-1) {
				bry = r + idx - 1 // end right at the contiguous group
				brx = c + 1
				addRect = true
			}
		} else {
			if t.IsPlayedTile() && tlx == -Infinity {
				tlx = c + idx
				tly = r - 1
			}
			// If we are at the end of a contiguous group (or of the whole
			// word)
			if tlx != -Infinity && (!t.IsPlayedTile() || idx == playLength-1) {
				brx = c + idx - 1
				bry = r + 1
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
	// of the word.
	if vert {
		s.addRectangle(c, r-1, c, r-1, stm)
		s.addRectangle(c, r+playLength, c, r+playLength, stm)
	} else {
		s.addRectangle(c-1, r, c-1, r, stm)
		s.addRectangle(c+playLength, r, c+playLength, r, stm)
	}
	// XXX 	handle single-tile plays that make two plays at once.
}

// Returns whether `play` blocks `other`
func (s *Solver) blocks(play *move.Move, other *move.Move) bool {
	// `play` blocks `other` if any square in `play` occupies any square
	// covered by `other`, or adjacent to it, or hooks onto a word formed
	// by `other`

	// rules of thumb:
	// - look at `other`'s  tiles played
	// - blocks if any tiles in `play` are ABOVE, BELOW, LEFT, RIGHT, or ON
	//      any PLAYED tiles by other
	// - blocks if any tiles in `play` are directly LEFT or RIGHT of the entire
	//     word formed by other (if the word is horizontal) or
	//     UP/DOWN if the word is vertical. (both if it's a 1 tile play that makes
	//     a horizontal and a vertical word)
	// So:
	// For every contiguous group of tiles that is PLACED in `other`,
	// create a rectangle that extrudes orthogonally to the direction of the
	// word, by one unit up and down. If this rectangle intersects the
	// rectangle formed by the outermost PLACED tiles of `play`, then `play`
	// blocks `other`.
	// Then, create a rectangle around the entire word formed by `other` and
	// extend it by one unit in the direction of the word both ways.
	// If the rectangle formed by the outermost PLACED tiles of `play`
	// intersects this rectangle, then `play` blocks `other`.
	s.stmRectIndex = 0
	s.otsRectIndex = 0
	s.setBlockingRectangles(play, true)
	if play.HasDupe() {
		s.setBlockingRectangles(play.Dupe(), true)
	}
	fmt.Println("Blocking rects for stm play", play, s.stmBlockingRects[0:s.stmRectIndex])
	s.setBlockingRectangles(other, false)
	if other.HasDupe() {
		s.setBlockingRectangles(other.Dupe(), false)
	}
	fmt.Println("Blocking rects for ots play", other, s.otsBlockingRects[0:s.otsRectIndex])

	for i := 0; i < s.stmRectIndex; i++ {
		for j := 0; j < s.otsRectIndex; j++ {
			if rectanglesIntersect(s.stmBlockingRects[i], s.otsBlockingRects[j]) {
				fmt.Println("Rectangles intersect", s.stmBlockingRects[i], s.otsBlockingRects[j])
				return true
			}
		}
	}
	return false

}

// func (s *Solver) blocksSimple(play *move.Move, other *move.Move) bool {
// 	// Use simple quadratic algorithm. This should be fast enough (at least
// 	// faster than making rectangles, I think, or comparable).
// 	r, c, vert := play.CoordsAndVertical()
// 	or, oc, overt := other.CoordsAndVertical()
// 	var x, y, ox, oy int
// 	for idx, t := range play.Tiles() {
// 		for oidx, ot := range other.Tiles() {
// 			if !t.IsPlayedTile() || !ot.IsPlayedTile() {
// 				// no way this til
// 			}
// 			if t.IsPlayedTile() && ot.IsPlayedTile() {
// 				if vert {
// 					x, y = c, r+idx
// 				} else {
// 					x, y = c+idx, r
// 				}
// 				if overt {
// 					ox, oy = oc, or+oidx
// 				} else {
// 					ox, oy = oc+idx, or
// 				}

// 			}
// 		}
// 	}
// }
