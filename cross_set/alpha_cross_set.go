package cross_set

import (
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/alphadawg"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tinymove"
)

// ----------------------------------------------------------------------
// WordSmogCrossSetGenerator generates cross sets via an alpha dawg.
//
// In WordSmog any anagram of a word is legal, so a square's cross set only
// depends on the multiset of tiles in the perpendicular run(s) around it --
// not on their order, and not on which side of the gap they sit. The left and
// right runs therefore merge into a single tally, and the answer is the same
// for every gap position inside a block.

type WordSmogCrossSetGenerator struct {
	Dist      *tilemapping.LetterDistribution
	AlphaDawg *kwg.KWG
}

func (g *WordSmogCrossSetGenerator) Generate(b *Board, row int, col int, dir board.BoardDirection) {
	GenAlphaCrossSet(b, row, col, dir, g.AlphaDawg, g.Dist)
}

func (g *WordSmogCrossSetGenerator) GenerateAll(b *Board) {
	generateAll(g, b)
}

func (g *WordSmogCrossSetGenerator) UpdateForMove(b *Board, m *move.Move) {
	updateForMove(g, b, m)
}

func (g *WordSmogCrossSetGenerator) UpdateForSmallMove(b *Board, m *tinymove.SmallMove, moveTiles *[board.MaxBoardDim]tilemapping.MachineLetter) {
	updateForSmallMove(g, b, m, moveTiles)
}

// GenAllAlphaCrossSets is a convenience wrapper mirroring GenAllCrossSets.
func GenAllAlphaCrossSets(b *Board, kad *kwg.KWG, ld *tilemapping.LetterDistribution) {
	gen := WordSmogCrossSetGenerator{Dist: ld, AlphaDawg: kad}
	gen.GenerateAll(b)
}

// GenAlphaCrossSet generates the WordSmog cross-set and cross-score for a
// single square. It mirrors GenCrossSet's structure; see magpie's
// game_gen_alpha_cross_set.
func GenAlphaCrossSet(b *Board, row int, col int, dir board.BoardDirection,
	kad *kwg.KWG, ld *tilemapping.LetterDistribution) {

	if row < 0 || row >= b.Dim() || col < 0 || col >= b.Dim() {
		return
	}
	tDir := throughDir(dir)

	// If the square has a letter in it, its cross set and cross score should
	// both be 0. Extension sets for filled squares are set while generating
	// the adjacent empty squares.
	if b.HasLetter(row, col) {
		b.ClearCrossSet(row, col, dir)
		b.SetCrossScore(row, col, 0, dir)
		return
	}
	// No adjacent tile in either direction: every letter is allowed.
	if b.LeftAndRightEmpty(row, col) {
		b.SetCrossSet(row, col, board.TrivialCrossSet, dir)
		b.SetCrossScore(row, col, 0, dir)
		b.SetLeftExtSet(row, col, tDir, board.TrivialCrossSet)
		b.SetRightExtSet(row, col, tDir, board.TrivialCrossSet)
		return
	}

	rightCol := b.WordEdge(row, col+1, Right)
	leftCol := b.WordEdge(row, col-1, Left)
	nonemptyToLeft := leftCol < col
	nonemptyToRight := rightCol > col

	// WordSmog has no notion of a prefix or suffix, so the extension sets that
	// shadow play consults carry no information here. They must still be
	// written as trivial rather than left at zero: shadowPlayForAnchor bails
	// out entirely when an anchor's extension sets are both empty. Write them
	// wherever GenCrossSet would.
	b.SetLeftExtSet(row, col, tDir, board.TrivialCrossSet)
	b.SetRightExtSet(row, col, tDir, board.TrivialCrossSet)

	var tally alphadawg.Tally
	score := 0

	if nonemptyToLeft {
		for c := col - 1; c >= leftCol; c-- {
			tally[b.GetLetter(row, c).Unblank()]++
		}
		score += b.TraverseBackwardsForScore(row, col-1, ld)
		b.SetLeftExtSet(row, col-1, tDir, board.TrivialCrossSet)
		b.SetRightExtSet(row, col-1, tDir, board.TrivialCrossSet)
		if leftCol > 0 {
			b.SetLeftExtSet(row, leftCol-1, tDir, board.TrivialCrossSet)
		}
	}

	if nonemptyToRight {
		for c := col + 1; c <= rightCol; c++ {
			tally[b.GetLetter(row, c).Unblank()]++
		}
		score += b.TraverseBackwardsForScore(row, rightCol, ld)
		b.SetLeftExtSet(row, rightCol, tDir, board.TrivialCrossSet)
		b.SetRightExtSet(row, rightCol, tDir, board.TrivialCrossSet)
	}

	b.SetCrossScore(row, col, score, dir)
	b.SetCrossSet(row, col,
		CrossSet(alphadawg.ComputeCrossSet(kad, &tally, int(ld.TileMapping().NumLetters()))),
		dir)
}
