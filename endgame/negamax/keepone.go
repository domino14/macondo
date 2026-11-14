package negamax

import (
	"fmt"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/tinymove"
)

// The keep-one bound.
//
// A move that leaves one tile y on the rack is worth, to the player making
// it, at most its score, plus the most y can score played alone afterwards,
// plus the going-out bonus (twice the opponent's rack). The opponent can
// always pass in reply, which leaves the player exactly that: y dropped on
// its best square to go out, or less if y has no square. Whatever else the
// opponent does is at least as good for them as passing.
//
// So where a node has alpha to beat, a move whose bound does not beat it
// needs no search. After the move y drops either where it could before, for
// the same score -- the best of those is found by one scan of the board, once
// per kept tile type -- or on one of the few squares the move changes, which
// are bounded from the tiles the move puts next to them.
//
// The bound is a fact about the game, not about a search of some depth, so
// it holds wherever the opponent's reply is searched (depth 2 and up) and
// does not make a node's value depend on its depth. The idea is MAGPIE's
// (keep_one_bound in endgame.c).

// keepOneBound is what a node needs to bound its moves that keep one tile.
type keepOneBound struct {
	b  *board.GameBoard
	ld *tilemapping.LetterDistribution
	// rackSum is the sum of the rack's machine letters (a blank is 0), from
	// which the tile a move keeps is what its own tiles leave.
	rackSum  int
	rackSize int
	// base is the node's spread for the player on turn plus the going-out
	// bonus.
	base int
	// bestDrop is the best score of each kept tile type played alone on the
	// node's board, or -1 with no legal square; scanned says which are
	// known.
	scanned  uint64
	bestDrop [tilemapping.MaxAlphabetSize + 1]int32
	// lists, when not nil, are the thread's move lists, whose list for the
	// player on turn here has every play of one tile: the best of each
	// type comes from it without a scan.
	lists  *pathLists
	onTurn int
}

func (k *keepOneBound) init(g *game.Game, lists *pathLists, onTurn int, ourSpread int) {
	k.lists = lists
	k.onTurn = onTurn
	k.b = g.Board()
	k.ld = g.Bag().LetterDistribution()
	rack := g.RackFor(onTurn)
	k.rackSize = int(rack.NumTiles())
	k.rackSum = 0
	for ml, n := range rack.LetArr {
		k.rackSum += ml * int(n)
	}
	k.base = ourSpread + 2*g.RackFor(1-onTurn).ScoreOn(k.ld)
	k.scanned = 0
}

// bound returns an upper bound on the value, to the player on turn, of the
// move m, which must place all but one of the rack's tiles. It stops looking
// for a tighter bound once the bound is above alpha, where it decides
// nothing.
func (k *keepOneBound) bound(m *tinymove.SmallMove, alpha int) int {
	b, ld := k.b, k.ld
	tm := uint64(m.TinyMove())
	n := m.TilesPlayed()
	keptSum := k.rackSum
	var placed [game.RackTileLimit]tilemapping.MachineLetter
	for i := 0; i < n && i < len(placed); i++ {
		ml := tilemapping.MachineLetter((tm >> (20 + 6*uint(i))) & 63)
		if tm&(1<<(12+uint(i))) != 0 {
			// A blank: 0 in the rack's sum, and no points on the board.
			ml = 0
		} else {
			keptSum -= int(ml)
		}
		placed[i] = ml
	}
	kept := tilemapping.MachineLetter(keptSum)
	if k.scanned&(1<<uint(kept)) == 0 {
		k.bestDrop[kept] = -1
		if k.lists != nil {
			t := k.lists.get(k.onTurn).soloPlays(b)
			if t.found&(1<<uint(kept)) != 0 {
				k.bestDrop[kept] = int32(t.score[kept])
			}
			if checkMoveLists {
				score, _, _, found := scanLoneTile(b, kept, ld)
				if !found {
					score = -1
				}
				if int32(score) != k.bestDrop[kept] {
					panic(fmt.Sprintf("keep-one bound: best play of tile %d alone is %d from the list, %d from the board",
						kept, k.bestDrop[kept], score))
				}
			}
		} else if score, _, _, found := scanLoneTile(b, kept, ld); found {
			k.bestDrop[kept] = int32(score)
		}
		k.scanned |= 1 << uint(kept)
	}
	moveBound := k.base + m.Score()
	nextDrop := max(int(k.bestDrop[kept]), 0)
	// The squares the move touches can only raise the bound.
	if moveBound+nextDrop > alpha {
		return moveBound + nextDrop
	}

	keptFace := loneTileFace(kept, ld)
	row, col, vertical := m.CoordsAndVertical()
	length := m.PlayLength()
	dr, dc := 0, 1
	mainDir, crossDir := board.HorizontalDirection, board.VerticalDirection
	if vertical {
		dr, dc = 1, 0
		mainDir, crossDir = board.VerticalDirection, board.HorizontalDirection
	}
	// The main word's tiles at face value: the move's and the ones it plays
	// through.
	wordScore := 0
	var placedFace [board.MaxBoardDim]int
	var isPlaced [board.MaxBoardDim]bool
	tidx := 0
	for i := 0; i < length; i++ {
		r, c := row+dr*i, col+dc*i
		if onBoard := b.GetLetter(r, c); onBoard != 0 {
			wordScore += ld.Score(onBoard)
			continue
		}
		face := 0
		if tidx < n && placed[tidx] != 0 {
			face = ld.Score(placed[tidx])
		}
		tidx++
		placedFace[i] = face
		isPlaced[i] = true
		wordScore += face
	}
	// The empty squares at the two ends of the main word.
	nextDrop = max(nextDrop,
		k.dropBeside(row-dr, col-dc, -dr, -dc, wordScore, crossDir, kept, keptFace),
		k.dropBeside(row+dr*length, col+dc*length, dr, dc, wordScore, crossDir, kept, keptFace))
	// Each placed tile joins the tiles on either side of it across the main
	// word; the empty squares at the two ends of that run are touched too.
	dim := b.Dim()
	for i := 0; i < length; i++ {
		if !isPlaced[i] {
			continue
		}
		r, c := row+dr*i, col+dc*i
		run := placedFace[i] + b.GetCrossScore(r, c, crossDir)
		for sign := -1; sign <= 1; sign += 2 {
			sr, sc := sign*dc, sign*dr
			er, ec := r+sr, c+sc
			for er >= 0 && er < dim && ec >= 0 && ec < dim && b.HasLetter(er, ec) {
				er, ec = er+sr, ec+sc
			}
			nextDrop = max(nextDrop,
				k.dropBeside(er, ec, sr, sc, run, mainDir, kept, keptFace))
		}
	}
	return moveBound + nextDrop
}

// dropBeside bounds what the kept tile scores played alone on the empty
// square (r, c) once the move is made. The square is at the end of a run of
// tiles, worth runScore at face value, that the move makes or extends; (sr,
// sc) steps from the run through the square and beyond. otherDir is the
// direction across the run, in which the move changes nothing for the
// square, so its cross-set and cross-score there still hold; whether the
// word along the run would be valid is not checked. Returns 0 when the
// square is off the board or the tile cannot go there.
func (k *keepOneBound) dropBeside(r, c, sr, sc int, runScore int,
	otherDir board.BoardDirection, kept tilemapping.MachineLetter, keptFace int) int {

	b := k.b
	dim := b.Dim()
	if r < 0 || r >= dim || c < 0 || c >= dim {
		return 0
	}
	idx := b.GetSqIdx(r, c)
	otherSet := uint64(b.GetCrossSetIdx(idx, otherDir))
	if kept == 0 {
		if otherSet>>1 == 0 {
			return 0
		}
	} else if otherSet&(1<<uint(kept)) == 0 {
		return 0
	}
	// Tiles on the far side of the square join the same word.
	for fr, fc := r+sr, c+sc; fr >= 0 && fr < dim && fc >= 0 && fc < dim && b.HasLetter(fr, fc); fr, fc = fr+sr, fc+sc {
		runScore += k.ld.Score(b.GetLetter(fr, fc))
	}
	tile := keptFace * b.GetLetterMultiplier(idx)
	wm := b.GetWordMultiplier(idx)
	score := (runScore + tile) * wm
	// A word across the run as well, if the square has a tile on either
	// side that way.
	or, oc := sc, sr
	if or < 0 || oc < 0 {
		or, oc = -or, -oc
	}
	hasOther := (r-or >= 0 && c-oc >= 0 && b.HasLetter(r-or, c-oc)) ||
		(r+or < dim && c+oc < dim && b.HasLetter(r+or, c+oc))
	if hasOther {
		score += (b.GetCrossScoreIdx(idx, otherDir) + tile) * wm
	}
	return score
}
