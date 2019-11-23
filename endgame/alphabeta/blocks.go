package alphabeta

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"

	"github.com/domino14/macondo/board"
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

func rectTransform(col, row int, startIdx, endIdx int, vertical, wide bool) (
	tlx, tly, brx, bry int) {
	// Given a top-left col/row, an index to start from, a direction, width,
	// and a length, return the relevant rectangle.
	wAdd := 0
	if wide {
		wAdd = 1
	}
	if vertical {
		tlx, brx = col-wAdd, col+wAdd
		tly, bry = row+startIdx, row+endIdx
	} else {
		tly, bry = row-wAdd, row+wAdd
		tlx, brx = col+startIdx, col+endIdx
	}
	return
}

func (s *Solver) setOrthogonalBlockingRects(play *move.Move, stm, wide bool) {
	r, c, vert := play.CoordsAndVertical()
	playLength := len(play.Tiles())

	startIdx := -Infinity
	endIdx := -Infinity
	for idx, t := range play.Tiles() {
		addRect := false

		if t.IsPlayedTile() && startIdx == -Infinity {
			startIdx = idx
		}
		// At the end of a contiguous group of tiles:
		if startIdx != -Infinity && (!t.IsPlayedTile() || idx == playLength-1) {
			if idx == playLength-1 && t.IsPlayedTile() {
				// end right at the actual play
				endIdx = idx
			} else {
				endIdx = idx - 1
			}
			addRect = true
		}
		if addRect {
			tlx, tly, brx, bry := rectTransform(c, r, startIdx, endIdx, vert, wide)
			s.addRectangle(tlx, tly, brx, bry, stm)
			startIdx, endIdx = -Infinity, -Infinity
		}
	}
}

func (s *Solver) setSTMBlockingRectangles(play *move.Move) {
	s.setOrthogonalBlockingRects(play, true, false)
}

func (s *Solver) setOTSBlockingRectangles(play *move.Move) {
	// set wide orthogonal rectangles here
	s.setOrthogonalBlockingRects(play, false, true)
	// Then make one-square rectangles on word edges, in the direction
	// of the word.
	// We don't need to cover the actual played tiles because the
	// orthogonal rectangles above do this.
	r, c, vert := play.CoordsAndVertical()
	playLength := len(play.Tiles())

	if vert {
		s.addRectangle(c, r-1, c, r-1, false)
		s.addRectangle(c, r+playLength, c, r+playLength, false)
	} else {
		s.addRectangle(c-1, r, c-1, r, false)
		s.addRectangle(c+playLength, r, c+playLength, r, false)
	}

	// In order to account for hooks, we need to actually explore the
	// stmPlay and add blocking rectangles for it here, ONLY if the
	// two plays have different directions.
	// sr, sc, svert := stmPlay.CoordsAndVertical()
	// if svert == vert {
	// 	return
	// }
	// splayLength := len(stmPlay.Tiles())
	// if svert {
	// 	s.addRectangle(sc, sr-1, sc, sr-1, true)
	// 	s.addRectangle(sc, sr+splayLength, sc, sr+splayLength, true)
	// } else {
	// 	s.addRectangle(sc-1, sr, sc-1, sr, true)
	// 	s.addRectangle(sc+splayLength, sr, sc+splayLength, sr, true)
	// }
}

func (s *Solver) setHookBlockingRectangles(play *move.Move, board *board.GameBoard) {

	// determine all words formed by `other`
	// for every cross word:
	//    - draw rectangles at each edge of the cross word
	r, c, vert := play.CoordsAndVertical()
	for idx, t := range play.Tiles() {
		if !t.IsPlayedTile() {
			continue
		}
		// Search up and down (or sideways)
		var x int

		if !vert {
			for x = r - 1; x >= 0; x-- {
				if board.GetLetter(x, idx+c) == alphabet.EmptySquareMarker {
					break
				}
			}
			s.addRectangle(idx+c, x, idx+c, x, false)
			for x = r + 1; x < board.Dim(); x++ {
				if board.GetLetter(x, idx+c) == alphabet.EmptySquareMarker {
					break
				}
			}
			s.addRectangle(idx+c, x, idx+c, x, false)
		} else {
			for x = c - 1; x >= 0; x-- {
				if board.GetLetter(idx+r, c) == alphabet.EmptySquareMarker {
					break
				}
			}
			s.addRectangle(x, idx+r, x, idx+r, false)
			for x = c + 1; x < board.Dim(); x++ {
				if board.GetLetter(idx+r, c) == alphabet.EmptySquareMarker {
					break
				}
			}
			s.addRectangle(x, idx+r, x, idx+r, false)
		}

	}
}

// Returns whether `play` blocks `other`
func (s *Solver) blocks(play *move.Move, other *move.Move, board *board.GameBoard) bool {
	// `play` blocks `other` if any square in `play` occupies any square
	// covered by `other`, or adjacent to it, or hooks onto a word formed
	// by `other`

	s.stmRectIndex = 0
	s.otsRectIndex = 0

	// How to model this:
	// For the `side to move` to block an `other side` play, we can draw
	// minimal rectangles around every PLAYED tile for the `side to move`,
	// and  draw bigger rectangles for the `other side` and figure out
	// if there are intersections.

	s.setSTMBlockingRectangles(play)
	s.setOTSBlockingRectangles(other)
	if other.HasDupe() {
		s.setOTSBlockingRectangles(other.Dupe())
	}

	// log.Debug().Msgf("Blocking rects for stm play %v: %v", play, s.stmBlockingRects[0:s.stmRectIndex])
	// log.Debug().Msgf("Blocking rects for ots play %v: %v", other, s.otsBlockingRects[0:s.otsRectIndex])

	for i := 0; i < s.stmRectIndex; i++ {
		for j := 0; j < s.otsRectIndex; j++ {
			if rectanglesIntersect(s.stmBlockingRects[i], s.otsBlockingRects[j]) {
				// log.Debug().Msgf("Rectangles intersect: %v, %v", s.stmBlockingRects[i], s.otsBlockingRects[j])
				return true
			}
		}
	}
	// If we haven't found a block yet, look for hooks.
	s.otsRectIndex = 0
	s.setHookBlockingRectangles(other, board)
	// log.Debug().Msgf("Blocking rects for ots play (after hook search) %v: %v", other, s.otsBlockingRects[0:s.otsRectIndex])

	for i := 0; i < s.stmRectIndex; i++ {
		for j := 0; j < s.otsRectIndex; j++ {
			if rectanglesIntersect(s.stmBlockingRects[i], s.otsBlockingRects[j]) {
				// log.Debug().Msgf("Rectangles intersect: %v, %v", s.stmBlockingRects[i], s.otsBlockingRects[j])
				return true
			}
		}
	}

	return false

}
