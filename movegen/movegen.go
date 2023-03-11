// Package movegen contains all the move-generating functions. It makes
// heavy use of the GADDAG.
// Implementation notes:
// - Is the specification in the paper a bit buggy? Basically, if I assume
// an anchor is the leftmost tile of a word, the way the algorithm works,
// it will create words blindly. For example, if I have a word FIRE on the
// board, and I have the letter E on my rack, and I specify F as the anchor,
// it will create the word EF! (Ignoring the fact that IRE is on the board)
// You can see this by just stepping through the algorithm.
// It seems that anchors can only be on the rightmost tile of a word
package movegen

import (
	"sort"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
)

type SortBy int

const (
	SortByScore SortBy = iota
	SortByNone
)

// MoveGenerator is a generic interface for generating moves.
type MoveGenerator interface {
	GenAll(rack *tilemapping.Rack, addExchange bool) []*move.Move
	SetSortingParameter(s SortBy)
	Plays() []*move.Move
	SetPlayRecorder(pf PlayRecorderFunc)
	SetEquityCalculators([]equity.EquityCalculator)
}

// GordonGenerator is the main move generation struct. It implements
// Steven A. Gordon's algorithm from his paper "A faster Scrabble Move Generation
// Algorithm"
type GordonGenerator struct {
	// curRow is the current row for which we are generating moves. Note
	// that we are always thinking in terms of rows, and columns are the
	// current anchor column. In order to generate vertical moves, we just
	// transpose the `board`.
	curRowIdx     int
	curAnchorCol  int
	lastAnchorCol int

	vertical bool // Are we generating moves vertically or not?

	tilesPlayed        int
	plays              []*move.Move
	numPossibleLetters int
	sortingParameter   SortBy

	// These are pointers to the actual structures in `game`. They are
	// duplicated here to speed up the algorithm, since we access them
	// so frequently (yes it makes a difference)
	gaddag gaddag.WordGraph
	board  *board.GameBoard
	// Used for scoring:
	letterDistribution *tilemapping.LetterDistribution

	// Used for play-finding without allocation
	strip         []tilemapping.MachineLetter
	exchangestrip []tilemapping.MachineLetter
	leavestrip    []tilemapping.MachineLetter

	playRecorder      PlayRecorderFunc
	equityCalculators []equity.EquityCalculator

	// used for play recorder:
	winner      *move.Move
	placeholder *move.Move
	game        *game.Game
}

// NewGordonGenerator returns a Gordon move generator.
func NewGordonGenerator(gd gaddag.WordGraph, board *board.GameBoard,
	ld *tilemapping.LetterDistribution) *GordonGenerator {

	gen := &GordonGenerator{
		gaddag:             gd,
		board:              board,
		numPossibleLetters: int(ld.TileMapping().NumLetters()),
		sortingParameter:   SortByScore,
		letterDistribution: ld,
		strip:              make([]tilemapping.MachineLetter, board.Dim()),
		exchangestrip:      make([]tilemapping.MachineLetter, 7), // max rack size. can make a parameter later.
		leavestrip:         make([]tilemapping.MachineLetter, 7),
		playRecorder:       AllPlaysRecorder,
		winner:             new(move.Move),
		placeholder:        new(move.Move),
	}
	return gen
}

// SetSortingParameter tells the play sorter to sort by score, equity, or
// perhaps other things. This is useful for the endgame solver, which does
// not care about equity.
func (gen *GordonGenerator) SetSortingParameter(s SortBy) {
	gen.sortingParameter = s
}

func (gen *GordonGenerator) SetPlayRecorder(pr PlayRecorderFunc) {
	gen.playRecorder = pr
}

func (gen *GordonGenerator) SetEquityCalculators(calcs []equity.EquityCalculator) {
	gen.equityCalculators = calcs
}

func (gen *GordonGenerator) SetGame(g *game.Game) {
	gen.game = g
}

// GenAll generates all moves on the board. It assumes anchors have already
// been updated, as well as cross-sets / cross-scores.
func (gen *GordonGenerator) GenAll(rack *tilemapping.Rack, addExchange bool) []*move.Move {
	gen.winner.SetEmpty()

	gen.plays = gen.plays[:0]

	orientations := [2]board.BoardDirection{
		board.HorizontalDirection, board.VerticalDirection}

	// Once for each orientation
	for idx, dir := range orientations {
		gen.vertical = idx%2 != 0
		gen.genByOrientation(rack, dir)
		gen.board.Transpose()
	}

	// Only add a pass move if nothing else is possible. Note: in endgames,
	// we will have to add a pass move another way (if it's a strategic pass).
	// Probably in the endgame package.
	if len(gen.plays) == 0 {
		tilesOnRack := rack.TilesOn()
		passMove := move.NewPassMove(tilesOnRack, rack.Alphabet())
		gen.plays = append(gen.plays, passMove)
	} else if len(gen.plays) > 1 {
		switch gen.sortingParameter {
		case SortByScore:
			sort.Slice(gen.plays, func(i, j int) bool {
				return gen.plays[i].Score() > gen.plays[j].Score()
			})
		case SortByNone:
			// Do not sort the plays. It is assumed that we will sort plays
			// elsewhere (for example, a dedicated endgame engine)
			break
		}
	}
	if addExchange {
		gen.generateExchangeMoves(rack, 0, 0)
	}
	return gen.plays
}

func (gen *GordonGenerator) genByOrientation(rack *tilemapping.Rack, dir board.BoardDirection) {
	dim := gen.board.Dim()

	for row := 0; row < dim; row++ {
		gen.curRowIdx = row
		// A bit of a hack. Set this to a large number at the beginning of
		// every loop
		gen.lastAnchorCol = 100
		for col := 0; col < dim; col++ {
			if gen.board.IsAnchor(row, col, dir) {
				gen.curAnchorCol = col
				gen.recursiveGen(col, rack, gen.gaddag.GetRootNodeIndex(), col, col, !gen.vertical)
				gen.lastAnchorCol = col
			}
		}
	}
}

// recursiveGen is an implementation of the Gordon Gen function.
func (gen *GordonGenerator) recursiveGen(col int, rack *tilemapping.Rack,
	nodeIdx uint32, leftstrip, rightstrip int, uniquePlay bool) {

	var csDirection board.BoardDirection
	// If a letter L is already on this square, then goOn...
	// curSquare := gen.board.GetSquare(gen.curRowIdx, col)
	curLetter := gen.board.GetLetter(gen.curRowIdx, col)

	if gen.vertical {
		csDirection = board.HorizontalDirection
	} else {
		csDirection = board.VerticalDirection
	}
	crossSet := gen.board.GetCrossSet(gen.curRowIdx, col, csDirection)

	if curLetter != 0 {
		// nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, curLetter.Unblank())
		raw := curLetter.Unblank()
		var nnIdx uint32
		var accepts bool
		for i := nodeIdx; ; i++ {
			if gen.gaddag.Tile(i) == uint8(raw) {
				nnIdx = gen.gaddag.ArcIndex(i)
				accepts = gen.gaddag.Accepts(i)
				break
			}
			if gen.gaddag.IsEnd(i) {
				break
			}
		}
		// is curLetter in the letter set of the nodeIdx?
		gen.goOn(col, curLetter, rack, nnIdx, accepts, leftstrip, rightstrip, uniquePlay)
	} else if !rack.Empty() {
		for i := nodeIdx; ; i++ {
			ml := tilemapping.MachineLetter(gen.gaddag.Tile(i))
			if ml != 0 && (rack.LetArr[ml] != 0 || rack.LetArr[0] != 0) && crossSet.Allowed(ml) {
				nnIdx := gen.gaddag.ArcIndex(i)
				accepts := gen.gaddag.Accepts(i)
				if rack.LetArr[ml] > 0 {
					rack.Take(ml)
					gen.tilesPlayed++
					gen.goOn(col, ml, rack, nnIdx, accepts, leftstrip, rightstrip, uniquePlay)
					gen.tilesPlayed--
					rack.Add(ml)
				}
				// check blank
				if rack.LetArr[0] > 0 {
					rack.Take(0)
					gen.tilesPlayed++
					gen.goOn(col, ml.Blank(), rack, nnIdx, accepts, leftstrip, rightstrip, uniquePlay)
					gen.tilesPlayed--
					rack.Add(0)
				}
			}
			if gen.gaddag.IsEnd(i) {
				break
			}
		}
	}
}

// goOn is an implementation of the Gordon GoOn function.
func (gen *GordonGenerator) goOn(curCol int, L tilemapping.MachineLetter,
	rack *tilemapping.Rack, newNodeIdx uint32, accepts bool,
	leftstrip, rightstrip int, uniquePlay bool) {

	if curCol <= gen.curAnchorCol {
		if gen.board.HasLetter(gen.curRowIdx, curCol) {
			gen.strip[curCol] = 0
		} else {
			gen.strip[curCol] = L
			if gen.vertical && gen.board.GetCrossSet(gen.curRowIdx, curCol, board.HorizontalDirection) == board.TrivialCrossSet {
				// If the horizontal direction is the trivial cross-set, this means
				// that there are no letters perpendicular to where we just placed
				// this letter. So any play we generate here should be unique.
				// We use this to avoid generating duplicates of single-tile plays.
				uniquePlay = true
			}
		}
		leftstrip = curCol

		// if L on OldArc and no letter directly left, then record play.
		noLetterDirectlyLeft := curCol == 0 ||
			!gen.board.HasLetter(gen.curRowIdx, curCol-1)

		// Check to see if there is a letter directly to the left.
		if accepts && noLetterDirectlyLeft && gen.tilesPlayed > 0 {
			// Only record the play if it is unique:
			// if 1 tile has been played, there should be no letters in the across
			// direction (otherwise the cross-set is not trivial)
			if uniquePlay || gen.tilesPlayed > 1 {
				gen.playRecorder(gen, rack, leftstrip, rightstrip, move.MoveTypePlay)
			}
		}
		if newNodeIdx == 0 {
			return
		}
		// Keep generating prefixes if there is room to the left, and don't
		// revisit an anchor we just saw.
		// This seems to work because we always shift direction afterwards, so we're
		// only looking at the first of a consecutive set of anchors going backwards,
		// and then always looking forward from then on.
		if curCol > 0 && curCol-1 != gen.lastAnchorCol {
			gen.recursiveGen(curCol-1, rack, newNodeIdx, leftstrip, rightstrip, uniquePlay)
		}
		// Then shift direction.
		// Get the index of the SeparationToken
		separationNodeIdx := gen.gaddag.NextNodeIdx(newNodeIdx, 0)
		// Check for no letter directly left AND room to the right (of the anchor
		// square)
		if separationNodeIdx != 0 && noLetterDirectlyLeft && gen.curAnchorCol < gen.board.Dim()-1 {
			gen.recursiveGen(gen.curAnchorCol+1, rack, separationNodeIdx, leftstrip, rightstrip, uniquePlay)
		}

	} else {
		if gen.board.HasLetter(gen.curRowIdx, curCol) {
			gen.strip[curCol] = 0
		} else {
			gen.strip[curCol] = L
			if gen.vertical && gen.board.GetCrossSet(gen.curRowIdx, curCol, board.HorizontalDirection) == board.TrivialCrossSet {
				// see explanation above.
				uniquePlay = true
			}
		}
		rightstrip = curCol

		noLetterDirectlyRight := curCol == gen.board.Dim()-1 ||
			!gen.board.HasLetter(gen.curRowIdx, curCol+1)
		if accepts && noLetterDirectlyRight && gen.tilesPlayed > 0 {
			if uniquePlay || gen.tilesPlayed > 1 {
				gen.playRecorder(gen, rack, leftstrip, rightstrip, move.MoveTypePlay)
			}
		}
		if newNodeIdx != 0 && curCol < gen.board.Dim()-1 {
			// There is room to the right
			gen.recursiveGen(curCol+1, rack, newNodeIdx, leftstrip, rightstrip, uniquePlay)
		}
	}
}

func (gen *GordonGenerator) crossDirection() board.BoardDirection {
	if gen.vertical {
		return board.HorizontalDirection
	}
	return board.VerticalDirection
}

func (gen *GordonGenerator) scoreMove(word tilemapping.MachineWord, row, col, tilesPlayed int) int {

	return gen.board.ScoreWord(word, row, col, tilesPlayed, gen.crossDirection(), gen.letterDistribution)
}

// Plays returns the generator's generated plays.
func (gen *GordonGenerator) Plays() []*move.Move {
	return gen.plays
}

// zero-allocation generation of exchange moves without duplicates:
func (gen *GordonGenerator) generateExchangeMoves(rack *tilemapping.Rack, ml tilemapping.MachineLetter, stripidx int) {

	// magic function written by @andy-k
	for int(ml) < len(rack.LetArr) && rack.LetArr[ml] == 0 {
		ml++
	}
	if int(ml) == len(rack.LetArr) {
		gen.playRecorder(gen, rack, 0, stripidx, move.MoveTypeExchange)
	} else {
		gen.generateExchangeMoves(rack, ml+1, stripidx)
		numthis := rack.LetArr[ml]
		for i := 0; i < numthis; i++ {
			gen.exchangestrip[stripidx] = ml
			stripidx += 1
			rack.Take(ml)
			gen.generateExchangeMoves(rack, ml+1, stripidx)
		}
		for i := 0; i < numthis; i++ {
			rack.Add(ml)
		}
	}
}
