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
	"math"
	"sort"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tinymove"
)

type SortBy int

const (
	SortByScore SortBy = iota
	SortByNone
)

// MaxAnchorCount is the maximum number of anchors we can track.
// For a 15x15 board this is overkill, but supports 21x21 super boards.
const MaxAnchorCount = 21 * 21

// Anchor stores shadow generation result for one anchor square.
// Used for prioritizing move generation by maximum possible equity.
type Anchor struct {
	HighestPossibleEquity float64
	HighestPossibleScore  int16
	Row                   uint8
	Col                   uint8
	LastAnchorCol         int8 // -1 means no previous anchor
	Vertical              bool
}

// heapifyAnchors converts gen.anchors[:gen.anchorCount] into a max-heap
// ordered by HighestPossibleEquity. This is an in-place operation with no allocations.
func (gen *GordonGenerator) heapifyAnchors() {
	n := gen.anchorCount
	// Build heap (heapify)
	for i := n/2 - 1; i >= 0; i-- {
		gen.siftDown(i, n)
	}
}

// popAnchor removes and returns the anchor with highest equity.
// Returns a zero Anchor if the heap is empty.
func (gen *GordonGenerator) popAnchor() Anchor {
	if gen.anchorCount == 0 {
		return Anchor{}
	}
	// Get the max (root)
	a := gen.anchors[0]
	gen.anchorCount--
	if gen.anchorCount > 0 {
		// Move last element to root and sift down
		gen.anchors[0] = gen.anchors[gen.anchorCount]
		gen.siftDown(0, gen.anchorCount)
	}
	return a
}

// siftDown maintains the max-heap property by sifting element at index i down.
func (gen *GordonGenerator) siftDown(i, n int) {
	for {
		largest := i
		left := 2*i + 1
		right := 2*i + 2

		if left < n && gen.anchors[left].HighestPossibleEquity > gen.anchors[largest].HighestPossibleEquity {
			largest = left
		}
		if right < n && gen.anchors[right].HighestPossibleEquity > gen.anchors[largest].HighestPossibleEquity {
			largest = right
		}
		if largest == i {
			break
		}
		gen.anchors[i], gen.anchors[largest] = gen.anchors[largest], gen.anchors[i]
		i = largest
	}
}

// MoveGenerator is a generic interface for generating moves.
type MoveGenerator interface {
	GenAll(rack *tilemapping.Rack, addExchange bool) []*move.Move
	SetMaxCanExchange(int)
	SetSortingParameter(s SortBy)
	Plays() []*move.Move
	SmallPlays() []tinymove.SmallMove
	SetPlayRecorder(pf PlayRecorderFunc)
	SetEquityCalculators([]equity.EquityCalculator)
	AtLeastOneTileMove(rack *tilemapping.Rack) bool
	SetMaxTileUsage(int)
	SetGenPass(bool)
}

// type moveHeap []*move.Move

// func (m moveHeap) Len() int { return len(m) }

// func (m moveHeap) Less(i, j int) bool { return m[i].Equity() < m[j].Equity() }

// func (m moveHeap) Swap(i, j int) { m[i], m[j] = m[j], m[i] }

// func (m *moveHeap) Push(x interface{}) {
// 	*m = append(*m, x.(*move.Move))
// }

// func (m *moveHeap) Pop() interface{} {
// 	old := *m
// 	n := len(old)
// 	x := old[n-1]
// 	*m = old[0 : n-1]
// 	return x
// }

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

	tilesPlayed      int
	plays            []*move.Move
	smallPlays       []tinymove.SmallMove
	sortingParameter SortBy

	// These are pointers to the actual structures in `game`. They are
	// duplicated here to speed up the algorithm, since we access them
	// so frequently (yes it makes a difference)
	gaddag   *kwg.KWG
	board    *board.GameBoard
	boardDim int
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

	genPass        bool
	quitEarly      bool
	maxTileUsage   int
	maxCanExchange int

	// For top N only:
	topNPlays       []*move.Move
	maxTopMovesSize int
	isTopPlayMode   bool // True when using TopPlayOnly or TopN recorder

	// Shadow generation state
	shadowEnabled bool
	anchors       [MaxAnchorCount]Anchor
	anchorCount   int

	// Current anchor extension sets (copied per-anchor)
	anchorLeftExtSet  uint64
	anchorRightExtSet uint64

	// Rack cross-set (bitmask of letters in rack)
	rackCrossSet uint64

	// Descending tile scores for shadow (max rack size = 7)
	descendingTileScores   [7]int
	fullRackDescTileScores [7]int // Copy for restoration
	descTileScoresCopy     [7]int // For shadowPlayRight restoration

	// Unrestricted multipliers tracking
	// Stores effective letter multiplier and cross-word multiplier info
	descendingEffLetterMuls [7]uint16
	descendingCrossWordMuls [7]uint16 // Packed: (crossWordMul << 8) | column
	descendingLetterMuls    [7]uint8  // Original letter multipliers for recalculation
	numUnrestrictedMuls     int
	lastWordMultiplier      int

	// Copies for restoration after shadowPlayRight
	descEffLetterMulsCopy   [7]uint16
	descCrossWordMulsCopy   [7]uint16
	descLetterMulsCopy      [7]uint8
	numUnrestMulsCopy       int

	// Shadow scoring accumulators (reset per anchor)
	shadowMainwordRestrictedScore int
	shadowPerpAdditionalScore     int
	shadowWordMultiplier          int

	// Shadow result tracking (reset per anchor)
	highestShadowEquity float64
	highestShadowScore  int
	shadowTilesPlayed   int

	// Position tracking for shadow
	currentLeftCol  int
	currentRightCol int
	firstPlayedCol  int

	// Rack copy for shadow restoration
	shadowRackCopy   tilemapping.Rack // Value type, not pointer
	rackCrossSetCopy uint64

	// Number of letters on rack (cached)
	numLettersOnRack int

	// Tile scores lookup (indexed by MachineLetter)
	// Pre-populated from LetterDistribution. Blanked letters have 0 score.
	tileScores [64]int

	// Best leave values for shadow equity calculation.
	// Index is leave size (0-7), value is max leave value for any leave of that size.
	// This gives us an upper bound on leave equity for shadow pruning.
	bestLeaves [8]float64

	// For computing bestLeaves during exchange generation
	computeBestLeavesInExchange bool
	leaveCalcForExchange        equity.Leaves
}

// NewGordonGenerator returns a Gordon move generator.
func NewGordonGenerator(gd gaddag.WordGraph, board *board.GameBoard,
	ld *tilemapping.LetterDistribution) *GordonGenerator {

	gen := &GordonGenerator{
		gaddag:             gd.(*kwg.KWG),
		board:              board,
		boardDim:           board.Dim(),
		sortingParameter:   SortByScore,
		letterDistribution: ld,
		strip:              make([]tilemapping.MachineLetter, board.Dim()),
		exchangestrip:      make([]tilemapping.MachineLetter, 7), // max rack size. can make a parameter later.
		leavestrip:         make([]tilemapping.MachineLetter, 7),
		playRecorder:       AllPlaysRecorder,
		winner:             new(move.Move),
		placeholder:        new(move.Move),
		maxTileUsage:       100, // basically unlimited
		maxCanExchange:     game.DefaultExchangeLimit,
	}

	// Pre-populate tile scores lookup (once at creation)
	for ml := 0; ml < int(ld.TileMapping().NumLetters()); ml++ {
		gen.tileScores[ml] = ld.Score(tilemapping.MachineLetter(ml))
		// Blanked versions (ml | BlankMask) will have score 0, already zero-initialized
	}

	return gen
}

func (gen *GordonGenerator) SetMaxCanExchange(m int) {
	gen.maxCanExchange = m
}

// SetSortingParameter tells the play sorter to sort by score, equity, or
// perhaps other things. This is useful for the endgame solver, which does
// not care about equity.
func (gen *GordonGenerator) SetSortingParameter(s SortBy) {
	gen.sortingParameter = s
}

func (gen *GordonGenerator) SetPlayRecorder(pr PlayRecorderFunc) {
	gen.playRecorder = pr
	gen.maxTopMovesSize = 0
	gen.isTopPlayMode = false
}

// SetTopPlayOnlyRecorder sets the recorder to TopPlayOnlyRecorder and marks
// that we're in top-play mode (eligible for shadow optimization).
func (gen *GordonGenerator) SetTopPlayOnlyRecorder() {
	gen.playRecorder = TopPlayOnlyRecorder
	gen.isTopPlayMode = true
}

func (gen *GordonGenerator) SetRecordNTopPlays(n int) {
	gen.playRecorder = TopNPlayRecorder
	gen.maxTopMovesSize = n
	gen.isTopPlayMode = true
	gen.topNPlays = make([]*move.Move, n)
	for i := range gen.topNPlays {
		gen.topNPlays[i] = new(move.Move)
		gen.topNPlays[i].SetEquity(math.Inf(-1))
		gen.topNPlays[i].SetEmpty()
	}
}

func (gen *GordonGenerator) SetEquityCalculators(calcs []equity.EquityCalculator) {
	gen.equityCalculators = calcs
}

// SetShadowEnabled enables or disables shadow-based move generation.
// When enabled and using TopPlayOnly or TopN recorder, anchors will be
// prioritized by maximum possible equity for early pruning.
func (gen *GordonGenerator) SetShadowEnabled(enabled bool) {
	gen.shadowEnabled = enabled
}

func (gen *GordonGenerator) ShadowEnabled() bool {
	return gen.shadowEnabled
}

func (gen *GordonGenerator) SetGame(g *game.Game) {
	gen.game = g
}

func (gen *GordonGenerator) SetGenPass(p bool) {
	gen.genPass = p
}

func (gen *GordonGenerator) SetMaxTileUsage(t int) {
	gen.maxTileUsage = t
}

// GenAll generates all moves on the board. It assumes anchors have already
// been updated, as well as cross-sets / cross-scores.
func (gen *GordonGenerator) GenAll(rack *tilemapping.Rack, addExchange bool) []*move.Move {

	gen.winner.SetEmpty()
	gen.quitEarly = false
	gen.plays = gen.plays[:0]
	for i := range gen.topNPlays {
		gen.topNPlays[i].SetEquity(math.Inf(-1))
		gen.topNPlays[i].SetEmpty()
	}

	// XXX: I shouldn't be doing this every time GenAll gets called. Only if actually
	// using the SmallMove recorder. This causes a needless allocation per loop.
	ptr := SmallPlaySlicePool.Get().(*[]tinymove.SmallMove)
	gen.smallPlays = *ptr
	gen.smallPlays = gen.smallPlays[0:0]

	// Use shadow-based generation when enabled and in top-play mode.
	// Shadow is beneficial when we only need the best move(s) since we can
	// prune anchors that can't produce better moves than what we've found.
	useShadow := gen.shadowEnabled && gen.isTopPlayMode

	// When shadow is enabled, we need bestLeaves computed BEFORE shadow generation.
	// If exchanges are also allowed, we compute bestLeaves during exchange generation
	// (like Magpie does) - one pass instead of two.
	exchangesGenerated := false
	if useShadow && addExchange {
		// Generate exchanges FIRST (before shadow), computing bestLeaves in the same pass
		gen.prepareForBestLeavesInExchange()
		gen.generateExchangeMoves(rack, 0, 0)
		gen.computeBestLeavesInExchange = false // Reset for safety
		exchangesGenerated = true
	}

	if useShadow {
		gen.genShadow(rack, exchangesGenerated)
		gen.recordScoringPlaysFromAnchors(rack)
	} else {
		gen.vertical = false
		gen.genByOrientation(rack, board.HorizontalDirection)
		gen.board.Transpose()
		gen.vertical = true
		gen.genByOrientation(rack, board.VerticalDirection)
		gen.board.Transpose()
	}

	// Only add a pass move if nothing else is possible. Note: in endgames,
	// we can have strategic passes. Check genPass variable as well.
	if (len(gen.plays) == 0 && len(gen.smallPlays) == 0) || gen.genPass {
		gen.playRecorder(gen, rack, 0, 0, move.MoveTypePass, 0)
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

	// Generate exchanges if not already done (non-shadow path)
	if addExchange && !exchangesGenerated {
		gen.generateExchangeMoves(rack, 0, 0)
	}
	*ptr = gen.smallPlays

	if gen.maxTopMovesSize != 0 {
		// We're in top-N mode. gen.plays is empty
		delfrom := -1
		for i := 0; i < gen.maxTopMovesSize; i++ {
			if gen.topNPlays[i].IsEmpty() {
				delfrom = i
				break
			}
		}
		if delfrom != -1 {
			return gen.topNPlays[:delfrom]
		}
		return gen.topNPlays

	}

	return gen.plays
}

// AtLeastOneTileMove generates moves. We don't care what they are; return true
// if there is at least one move that plays tiles, false otherwise.
func (gen *GordonGenerator) AtLeastOneTileMove(rack *tilemapping.Rack) bool {
	pr := gen.playRecorder
	gen.quitEarly = false
	defer gen.SetPlayRecorder(pr)

	gen.SetPlayRecorder(
		func(MoveGenerator, *tilemapping.Rack, int, int, move.MoveType, int) {
			gen.quitEarly = true
		},
	)

	gen.plays = gen.plays[:0]
	gen.vertical = false
	gen.genByOrientation(rack, board.HorizontalDirection)

	if gen.quitEarly {
		return true
	}
	gen.board.Transpose()
	gen.vertical = true
	gen.genByOrientation(rack, board.VerticalDirection)
	gen.board.Transpose()

	return gen.quitEarly
}

func (gen *GordonGenerator) genByOrientation(rack *tilemapping.Rack, dir board.BoardDirection) {

	for row := 0; row < gen.boardDim; row++ {
		gen.curRowIdx = row
		// A bit of a hack. Set this to a large number at the beginning of
		// every loop
		gen.lastAnchorCol = 100
		for col := 0; col < gen.boardDim; col++ {
			if gen.board.IsAnchor(row, col, dir) {
				gen.curAnchorCol = col
				gen.recursiveGen(col, rack, gen.gaddag.GetRootNodeIndex(), col, col, !gen.vertical, 0, 0, 1)
				gen.lastAnchorCol = col
			}
		}
	}
}

// recursiveGen is an implementation of the Gordon Gen function.
func (gen *GordonGenerator) recursiveGen(col int, rack *tilemapping.Rack,
	nodeIdx uint32, leftstrip, rightstrip int, uniquePlay bool,
	baseScore int, crossScores int, wordMultiplier int) {

	if gen.quitEarly {
		return
	}
	sqIdx := gen.board.GetSqIdx(gen.curRowIdx, col)

	var csDirection board.BoardDirection
	// If a letter L is already on this square, then goOn...
	// curSquare := gen.board.GetSquare(gen.curRowIdx, col)
	curLetter := gen.board.GetLetter(gen.curRowIdx, col)

	if gen.vertical {
		csDirection = board.HorizontalDirection
	} else {
		csDirection = board.VerticalDirection
	}
	crossSet := gen.board.GetCrossSetIdx(sqIdx, csDirection)
	gd := gen.gaddag
	if curLetter != 0 {
		// There is a letter in this square.
		// nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, curLetter.Unblank())
		raw := curLetter.Unblank()
		var nnIdx uint32
		var accepts bool
		for i := nodeIdx; ; i++ {
			if gd.Tile(i) == uint8(raw) {
				nnIdx = gd.ArcIndex(i)
				accepts = gd.Accepts(i)
				break
			}
			if gd.IsEnd(i) {
				break
			}
		}
		// is curLetter in the letter set of the nodeIdx?
		gen.goOn(col, curLetter, rack, nnIdx, accepts, leftstrip, rightstrip, uniquePlay,
			baseScore+gen.letterDistribution.Score(curLetter),
			crossScores,
			wordMultiplier,
		)
	} else if !rack.Empty() {
		lm := gen.board.GetLetterMultiplier(sqIdx)
		cs := gen.board.GetCrossScoreIdx(sqIdx, csDirection)
		wm := gen.board.GetWordMultiplier(sqIdx)
		emptyAdjacent := crossSet == board.TrivialCrossSet
		for i := nodeIdx; ; i++ {
			ml := tilemapping.MachineLetter(gd.Tile(i))
			if ml != 0 && crossSet.Allowed(ml) && (rack.LetArr[ml] != 0 || rack.LetArr[0] != 0) {
				arcIdx := gd.ArcIndex(i)
				accepts := gd.Accepts(i)
				if rack.LetArr[ml] > 0 {
					rack.Take(ml)
					gen.tilesPlayed++
					sml := gen.letterDistribution.Score(ml)
					addlCrossScore := 0
					if !emptyAdjacent {
						if wm > 1 {
							addlCrossScore = wm * (cs + sml)
						} else {
							addlCrossScore = cs + sml*lm
						}
					}
					gen.goOn(col, ml, rack, arcIdx, accepts, leftstrip, rightstrip, uniquePlay,
						baseScore+(sml*lm),
						crossScores+addlCrossScore,
						wordMultiplier*wm)

					gen.tilesPlayed--
					rack.Add(ml)
				}
				// check blank
				if rack.LetArr[0] > 0 {
					rack.Take(0)
					gen.tilesPlayed++
					// XXX: this won't work for non-zero-score blanks if
					// that's ever a thing.
					gen.goOn(col, ml.Blank(), rack, arcIdx, accepts, leftstrip, rightstrip, uniquePlay,
						baseScore,
						crossScores+cs*wm,
						wordMultiplier*wm)
					gen.tilesPlayed--
					rack.Add(0)
				}
			}
			if gd.IsEnd(i) {
				break
			}
		}
	}
}

// goOn is an implementation of the Gordon GoOn function.
func (gen *GordonGenerator) goOn(curCol int, L tilemapping.MachineLetter,
	rack *tilemapping.Rack, newNodeIdx uint32, accepts bool,
	leftstrip, rightstrip int, uniquePlay bool, baseScore, crossScores, wordMultiplier int) {
	var bingoBonus int
	if gen.tilesPlayed == game.RackTileLimit {
		bingoBonus = 50
	}
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
			if (uniquePlay || gen.tilesPlayed > 1) && gen.tilesPlayed <= gen.maxTileUsage {
				gen.playRecorder(gen, rack, leftstrip, rightstrip, move.MoveTypePlay,
					baseScore*wordMultiplier+crossScores+bingoBonus)
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
			gen.recursiveGen(curCol-1, rack, newNodeIdx, leftstrip, rightstrip, uniquePlay, baseScore, crossScores, wordMultiplier)
		}
		// Then shift direction.
		// Get the index of the SeparationToken
		separationNodeIdx := gen.gaddag.NextNodeIdx(newNodeIdx, 0)
		// Check for no letter directly left AND room to the right (of the anchor
		// square)
		if separationNodeIdx != 0 && noLetterDirectlyLeft && gen.curAnchorCol < gen.boardDim-1 {
			gen.recursiveGen(gen.curAnchorCol+1, rack, separationNodeIdx, leftstrip, rightstrip, uniquePlay, baseScore, crossScores, wordMultiplier)
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

		noLetterDirectlyRight := curCol == gen.boardDim-1 ||
			!gen.board.HasLetter(gen.curRowIdx, curCol+1)
		if accepts && noLetterDirectlyRight && gen.tilesPlayed > 0 {
			if (uniquePlay || gen.tilesPlayed > 1) && gen.tilesPlayed <= gen.maxTileUsage {
				gen.playRecorder(gen, rack, leftstrip, rightstrip, move.MoveTypePlay,
					baseScore*wordMultiplier+crossScores+bingoBonus)
			}
		}
		if newNodeIdx != 0 && curCol < gen.boardDim-1 {
			// There is room to the right
			gen.recursiveGen(curCol+1, rack, newNodeIdx, leftstrip, rightstrip, uniquePlay, baseScore, crossScores, wordMultiplier)
		}
	}

}

func (gen *GordonGenerator) crossDirection() board.BoardDirection {
	if gen.vertical {
		return board.HorizontalDirection
	}
	return board.VerticalDirection
}

// Debug methods for testing
func (gen *GordonGenerator) GenShadowDebug(rack *tilemapping.Rack) {
	gen.genShadow(rack, false)
}

func (gen *GordonGenerator) AnchorCountDebug() int {
	return gen.anchorCount
}

func (gen *GordonGenerator) GetAnchorDebug(i int) Anchor {
	return gen.anchors[i]
}

func (gen *GordonGenerator) scoreMove(word tilemapping.MachineWord, row, col, tilesPlayed int) int {

	return gen.board.ScoreWord(word, row, col, tilesPlayed, gen.crossDirection(), gen.letterDistribution)
}

// Plays returns the generator's generated plays.
func (gen *GordonGenerator) Plays() []*move.Move {
	if gen.maxTopMovesSize > 0 {
		delfrom := -1
		for i := 0; i < gen.maxTopMovesSize; i++ {
			if gen.topNPlays[i].IsEmpty() {
				delfrom = i
				break
			}
		}
		if delfrom != -1 {
			return gen.topNPlays[:delfrom]
		}
		return gen.topNPlays
	}
	return gen.plays
}

// SmallPlays returns the generator's generated SmallPlays
func (gen *GordonGenerator) SmallPlays() []tinymove.SmallMove {
	return gen.smallPlays
}

// generateExchangeMoves generates exchange moves without duplicates (zero-allocation).
// When computeBestLeavesInExchange is true, it also computes the best leave value
// for each leave size in the same pass (like Magpie does).
func (gen *GordonGenerator) generateExchangeMoves(rack *tilemapping.Rack, ml tilemapping.MachineLetter, stripidx int) {
	// magic function written by @andy-k
	for int(ml) < len(rack.LetArr) && rack.LetArr[ml] == 0 {
		ml++
	}
	if int(ml) == len(rack.LetArr) {
		// Record exchange move (the rack now contains the leave)
		gen.playRecorder(gen, rack, 0, stripidx, move.MoveTypeExchange, 0)

		// Also compute best leave value for this leave size if enabled
		if gen.computeBestLeavesInExchange && gen.leaveCalcForExchange != nil {
			leaveSize := int(rack.NumTiles())
			if leaveSize < len(gen.bestLeaves) {
				leaveLength := rack.NoAllocTilesOn(gen.leavestrip)
				leave := tilemapping.MachineWord(gen.leavestrip[:leaveLength])
				leaveValue := gen.leaveCalcForExchange.LeaveValue(leave)
				if leaveValue > gen.bestLeaves[leaveSize] {
					gen.bestLeaves[leaveSize] = leaveValue
				}
			}
		}
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

// prepareForBestLeavesInExchange sets up for computing bestLeaves during exchange generation.
// This must be called before generateExchangeMoves when shadow is enabled.
func (gen *GordonGenerator) prepareForBestLeavesInExchange() {
	// Initialize bestLeaves to very negative values
	for i := range gen.bestLeaves {
		gen.bestLeaves[i] = -math.MaxFloat64
	}

	// Find a leave calculator among the equity calculators
	gen.leaveCalcForExchange = nil
	for _, calc := range gen.equityCalculators {
		if lc, ok := calc.(equity.Leaves); ok {
			gen.leaveCalcForExchange = lc
			break
		}
	}

	// If no leave calculator, set bestLeaves to 0 (no leave adjustment)
	if gen.leaveCalcForExchange == nil {
		for i := range gen.bestLeaves {
			gen.bestLeaves[i] = 0
		}
		gen.computeBestLeavesInExchange = false
	} else {
		gen.computeBestLeavesInExchange = true
	}
}

// computeBestLeavesFromExchanges enumerates all possible leaves (like exchange generation)
// and tracks the best leave value for each leave size. This is used for shadow equity
// upper bounds when exchanges are NOT allowed (so we can't piggyback on exchange generation).
func (gen *GordonGenerator) computeBestLeavesFromExchanges(rack *tilemapping.Rack, leaveCalc equity.Leaves, ml tilemapping.MachineLetter) {
	// Skip empty slots
	for int(ml) < len(rack.LetArr) && rack.LetArr[ml] == 0 {
		ml++
	}

	if int(ml) == len(rack.LetArr) {
		// Compute leave value for current rack state (the leave)
		leaveSize := int(rack.NumTiles())
		if leaveSize < len(gen.bestLeaves) {
			leaveLength := rack.NoAllocTilesOn(gen.leavestrip)
			leave := tilemapping.MachineWord(gen.leavestrip[:leaveLength])
			leaveValue := leaveCalc.LeaveValue(leave)
			if leaveValue > gen.bestLeaves[leaveSize] {
				gen.bestLeaves[leaveSize] = leaveValue
			}
		}
	} else {
		// Try keeping all copies of this tile (don't remove any)
		gen.computeBestLeavesFromExchanges(rack, leaveCalc, ml+1)

		// Try removing 1, 2, ... copies of this tile
		numthis := rack.LetArr[ml]
		for i := 0; i < numthis; i++ {
			rack.Take(ml)
			gen.computeBestLeavesFromExchanges(rack, leaveCalc, ml+1)
		}

		// Restore all copies
		for i := 0; i < numthis; i++ {
			rack.Add(ml)
		}
	}
}

func (gen *GordonGenerator) GADDAG() *kwg.KWG {
	return gen.gaddag
}
