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

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/strategy"
)

type SortBy int

const (
	SortByScore SortBy = iota
	SortByNone
)

// MoveGenerator is a generic interface for generating moves.
type MoveGenerator interface {
	GenAll(rack *alphabet.Rack, addExchange bool) []*move.Move
	SetSortingParameter(s SortBy)
	Plays() []*move.Move
	SetPlayRecorder(pf PlayRecorderFunc)
	SetStrategizer(st strategy.Strategizer)
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
	gaddag *gaddag.SimpleGaddag
	board  *board.GameBoard

	// These are data structures needed by the move generator, which are
	// closely tied to the game board:
	anchors *Anchors
	csetGen *cross_set.GaddagCrossSetGenerator
	// They are in this state:
	state *State

	// Used for scoring:
	letterDistribution *alphabet.LetterDistribution

	// Used for play-finding without allocation
	strip         []alphabet.MachineLetter
	exchangestrip []alphabet.MachineLetter
	leavestrip    []alphabet.MachineLetter

	playRecorder PlayRecorderFunc
	strategizer  strategy.Strategizer

	// used for play recorder:
	winner      *move.Move
	placeholder *move.Move
	game        *game.Game
}

// NewGordonGenerator returns a Gordon move generator.
func NewGordonGenerator(gd *gaddag.SimpleGaddag, board *board.GameBoard,
	ld *alphabet.LetterDistribution) *GordonGenerator {

	state := &State{
		anchors: MakeAnchors(board),
		csetGen: cross_set.MakeGaddagCrossSetGenerator(board, gd, ld),
	}

	gen := &GordonGenerator{
		gaddag:             gd,
		board:              board,
		csetGen:            state.csetGen,
		anchors:            state.anchors,
		numPossibleLetters: int(gd.GetAlphabet().NumLetters()),
		sortingParameter:   SortByScore,
		state:              state,
		letterDistribution: ld,
		strip:              make([]alphabet.MachineLetter, board.Dim()),
		exchangestrip:      make([]alphabet.MachineLetter, 7), // max rack size. can make a parameter later.
		leavestrip:         make([]alphabet.MachineLetter, 7),
		playRecorder:       AllPlaysRecorder,
		winner:             new(move.Move),
		placeholder:        new(move.Move),
	}
	return gen
}

func (gen *GordonGenerator) State() *State {
	return gen.state
}

func (gen *GordonGenerator) ResetCrossesAndAnchors() {
	gen.csetGen.GenerateAll(gen.board)
	gen.anchors.UpdateAllAnchors()
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

func (gen *GordonGenerator) SetStrategizer(st strategy.Strategizer) {
	gen.strategizer = st
}

func (gen *GordonGenerator) SetGame(g *game.Game) {
	gen.game = g
}

// GenAll generates all moves on the board. It assumes anchors have already
// been updated, as well as cross-sets / cross-scores.
func (gen *GordonGenerator) GenAll(rack *alphabet.Rack, addExchange bool) []*move.Move {
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

func (gen *GordonGenerator) genByOrientation(rack *alphabet.Rack, dir board.BoardDirection) {
	dim := gen.board.Dim()

	for row := 0; row < dim; row++ {
		gen.curRowIdx = row
		// A bit of a hack. Set this to a large number at the beginning of
		// every loop
		gen.lastAnchorCol = 100
		for col := 0; col < dim; col++ {
			if gen.anchors.IsAnchor(row, col, dir) {
				gen.curAnchorCol = col
				gen.recursiveGen(col, rack, gen.gaddag.GetRootNodeIndex(), col, col, !gen.vertical)
				gen.lastAnchorCol = col
			}
		}
	}
}

// recursiveGen is an implementation of the Gordon Gen function.
func (gen *GordonGenerator) recursiveGen(col int, rack *alphabet.Rack,
	nodeIdx uint32, leftstrip, rightstrip int, uniquePlay bool) {

	// If a letter L is already on this square, then goOn...
	curLetter := gen.board.GetLetter(gen.curRowIdx, col)
	if curLetter != alphabet.EmptySquareMarker {
		nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, curLetter.Unblank())
		gen.goOn(col, curLetter, rack, nnIdx, nodeIdx, leftstrip, rightstrip, uniquePlay)

	} else if !rack.Empty() {
		csDirection := gen.crossDirection()
		crossSet := gen.csetGen.CS.Get(gen.curRowIdx, col, csDirection)
		for ml := alphabet.MachineLetter(0); ml < alphabet.MachineLetter(gen.numPossibleLetters); ml++ {
			if rack.LetArr[ml] == 0 {
				continue
			}
			if crossSet.Allowed(ml) {
				nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, ml)
				rack.Take(ml)
				gen.tilesPlayed++
				gen.goOn(col, ml, rack, nnIdx, nodeIdx, leftstrip, rightstrip, uniquePlay)
				rack.Add(ml)
				gen.tilesPlayed--
			}

		}

		if rack.LetArr[alphabet.BlankMachineLetter] > 0 {
			// It's a blank. Loop only through letters in the cross-set.
			for i := 0; i < gen.numPossibleLetters; i++ {
				if crossSet.Allowed(alphabet.MachineLetter(i)) {
					nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, alphabet.MachineLetter(i))
					rack.Take(alphabet.BlankMachineLetter)
					gen.tilesPlayed++
					gen.goOn(col, alphabet.MachineLetter(i).Blank(), rack, nnIdx, nodeIdx, leftstrip, rightstrip, uniquePlay)
					rack.Add(alphabet.BlankMachineLetter)
					gen.tilesPlayed--
				}
			}
		}

	}
}

// goOn is an implementation of the Gordon GoOn function.
func (gen *GordonGenerator) goOn(curCol int, L alphabet.MachineLetter,
	rack *alphabet.Rack, newNodeIdx uint32, oldNodeIdx uint32,
	leftstrip, rightstrip int, uniquePlay bool) {

	if curCol <= gen.curAnchorCol {
		if gen.board.HasLetter(gen.curRowIdx, curCol) {
			gen.strip[curCol] = alphabet.PlayedThroughMarker
		} else {
			gen.strip[curCol] = L
			if gen.vertical && gen.csetGen.CS.Get(gen.curRowIdx, curCol, board.HorizontalDirection) == cross_set.TrivialCrossSet {
				// If the horizontal direction is the trivial cross-set, this means
				// that there are no letters perpendicular to where we just placed
				// this letter. So any play we generate here should be unique.
				// We use this to avoid generating duplicates of single-tile plays.
				uniquePlay = true
			}
		}
		leftstrip = curCol

		// if L on OldArc and no letter directly left, then record play.
		noLetterDirectlyLeft := curCol == 0 || !gen.board.HasLetter(gen.curRowIdx, curCol-1)

		// Check to see if there is a letter directly to the left.
		if gen.gaddag.InLetterSet(L, oldNodeIdx) && noLetterDirectlyLeft && gen.tilesPlayed > 0 {
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
		separationNodeIdx := gen.gaddag.NextNodeIdx(newNodeIdx, alphabet.SeparationMachineLetter)
		// Check for no letter directly left AND room to the right (of the anchor
		// square)
		if separationNodeIdx != 0 && noLetterDirectlyLeft && gen.curAnchorCol < gen.board.Dim()-1 {
			gen.recursiveGen(gen.curAnchorCol+1, rack, separationNodeIdx, leftstrip, rightstrip, uniquePlay)
		}

	} else {
		if gen.board.HasLetter(gen.curRowIdx, curCol) {
			gen.strip[curCol] = alphabet.PlayedThroughMarker
		} else {
			gen.strip[curCol] = L
			if gen.vertical && gen.csetGen.CS.Get(gen.curRowIdx, curCol, board.HorizontalDirection) == cross_set.TrivialCrossSet {
				// see explanation above.
				uniquePlay = true
			}
		}
		rightstrip = curCol

		noLetterDirectlyRight := curCol == gen.board.Dim()-1 ||
			!gen.board.HasLetter(gen.curRowIdx, curCol+1)

		if gen.gaddag.InLetterSet(L, oldNodeIdx) && noLetterDirectlyRight && gen.tilesPlayed > 0 {
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
	} else {
		return board.VerticalDirection
	}
}

func (gen *GordonGenerator) scoreMove(word alphabet.MachineWord, row, col, tilesPlayed int) int {
	return gen.board.ScoreWord(word, row, col, tilesPlayed, gen.crossDirection(), gen.letterDistribution)
}

// Plays returns the generator's generated plays.
func (gen *GordonGenerator) Plays() []*move.Move {
	return gen.plays
}

// zero-allocation generation of exchange moves without duplicates:
func (gen *GordonGenerator) generateExchangeMoves(rack *alphabet.Rack, ml alphabet.MachineLetter, stripidx int) {

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
