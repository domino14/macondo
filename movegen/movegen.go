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

	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/leavemap"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/wmp"
)

type SortBy int

const (
	SortByScore SortBy = iota
	SortByNone
)

// DefaultSmallPlayCapacity is the pre-allocated capacity for the smallPlays
// slice on each GordonGenerator. It matches magpie's
// DEFAULT_SMALL_MOVE_LIST_CAPACITY (config_defs.h:10). Two-blank racks can
// produce very large move counts, so we allocate generously to avoid any
// reallocation during move generation.
const DefaultSmallPlayCapacity = 250_000

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

	// Row cache for cache-friendly board access during move generation.
	cache rowCache

	// Leave map for O(1) leave value lookup during move generation.
	leavemap     leavemap.LeaveMap
	klv          *equity.KLV // cached KLV reference

	// WMP move generator state. Inactive by default; SetWMP enables
	// it. When active, shadow play uses WMP existence checks to
	// tighten highestShadowEquity. wmpRecordSubAnchors controls
	// whether shadow_play_for_anchor dumps per-(blocks, tiles)
	// sub-anchors into the heap (the MAGPIE behavior, useful only
	// with a wordmap_gen-style word generator) or sticks with the
	// single-anchor-per-square macondo path. Default false.
	wmpData             *wmp.WMP
	wmpMoveGen          wmp.MoveGen
	wmpRecordSubAnchors bool
	pegValues    []float64   // pre-endgame adjustment values
	tilesInBag   int         // cached bag state for equity calc
	oppRackScore int         // cached for endgame equity

	// Shadow play state
	shadow shadowState
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
		smallPlays:         make([]tinymove.SmallMove, 0, DefaultSmallPlayCapacity),
		playRecorder:       AllPlaysRecorder,
		winner:             new(move.Move),
		placeholder:        new(move.Move),
		maxTileUsage:       100, // basically unlimited
		maxCanExchange:     game.DefaultExchangeLimit,
		cache:              rowCache{loadedRow: -1},
		// shadow.useShadow starts false; set by SetPlayRecorderTopPlay/SetRecordNTopPlays
	}
	return gen
}

func (gen *GordonGenerator) SetMaxCanExchange(m int) {
	gen.maxCanExchange = m
}

// SetWMP enables WMP-aware move generation. Pass nil to disable.
// When set, shadow play uses the WMP existence checks to tighten the
// highestShadowEquity bound for each anchor square AND records
// per-(playthrough_blocks, tiles_to_play) sub-anchors that are
// processed by wordmapGen (instead of recursive_gen) during the
// best-first scoring pass. Mirrors MAGPIE's player_set_wmp + the
// wmp_move_gen_init call inside gen_load_position.
func (gen *GordonGenerator) SetWMP(w *wmp.WMP) {
	gen.wmpData = w
	// Sub-anchor recording is meaningful only when there's a fast
	// per-(blocks, tiles) word generator on the consumer side; we
	// have wordmapGen now, so enable it by default whenever WMP is
	// turned on. Pass nil to SetWMP to switch back to the non-WMP
	// path entirely.
	gen.wmpRecordSubAnchors = w != nil
}

// SetWMPRecordSubAnchors lets callers override the default
// sub-anchor recording behavior. SetWMP turns this on automatically;
// the override is mostly for benchmarks that want to compare the
// shadow-only-gating variant against the full WMP integration.
func (gen *GordonGenerator) SetWMPRecordSubAnchors(b bool) {
	gen.wmpRecordSubAnchors = b
}

// WMP returns the WMP currently in use, or nil if WMP is disabled.
func (gen *GordonGenerator) WMP() *wmp.WMP {
	return gen.wmpData
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
	// AllPlays recorders don't benefit from shadow; disable it to avoid
	// the overhead of computing upper bounds.
	gen.shadow.useShadow = false
}

// SetPlayRecorderTopPlay sets the play recorder to TopPlayOnlyRecorder
// with shadow enabled for best-first move finding.
func (gen *GordonGenerator) SetPlayRecorderTopPlay() {
	gen.playRecorder = TopPlayOnlyRecorder
	gen.maxTopMovesSize = 0
	gen.shadow.useShadow = true
}

func (gen *GordonGenerator) SetRecordNTopPlays(n int) {
	gen.playRecorder = TopNPlayRecorder
	gen.shadow.useShadow = true
	gen.maxTopMovesSize = n
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
// When shadow is enabled (via SetPlayRecorderTopPlay or SetRecordNTopPlays),
// GenAll uses the shadow algorithm for
// best-first move finding, which allows early termination when looking for
// only the top move(s). Shadow can be disabled for endgame where all moves
// must be found without the overhead of computing upper bounds.
func (gen *GordonGenerator) GenAll(rack *tilemapping.Rack, addExchange bool) []*move.Move {

	if gen.shadow.useShadow {
		return gen.GenAllWithShadow(rack, addExchange)
	}

	gen.winner.SetEmpty()
	gen.quitEarly = false
	gen.plays = gen.plays[:0]
	for i := range gen.topNPlays {
		gen.topNPlays[i].SetEquity(math.Inf(-1))
		gen.topNPlays[i].SetEmpty()
	}

	gen.smallPlays = gen.smallPlays[:0]

	gen.vertical = false
	gen.genByOrientation(rack, board.HorizontalDirection)
	gen.board.Transpose()
	gen.vertical = true
	gen.genByOrientation(rack, board.VerticalDirection)
	gen.board.Transpose()

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

	if addExchange {
		gen.generateExchangeMoves(rack, 0, 0)
	}

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
	var csDir board.BoardDirection
	if dir == board.HorizontalDirection {
		csDir = board.VerticalDirection
	} else {
		csDir = board.HorizontalDirection
	}

	for row := 0; row < gen.boardDim; row++ {
		gen.curRowIdx = row
		gen.cache.loadRow(gen.board, row, csDir, gen.boardDim)
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

	// Auto-load row cache if not loaded for current row/direction
	var csDir board.BoardDirection
	if gen.vertical {
		csDir = board.HorizontalDirection
	} else {
		csDir = board.VerticalDirection
	}
	if gen.cache.loadedRow != gen.curRowIdx || gen.cache.loadedDir != csDir {
		gen.cache.loadRow(gen.board, gen.curRowIdx, csDir, gen.boardDim)
	}

	sq := &gen.cache.squares[col]
	curLetter := sq.letter
	crossSet := sq.crossSet
	// Hoist the underlying KWG node slice out of the gaddag once,
	// so the inner loop can read each node value into a local
	// uint32 and decode the four bit fields (Tile, ArcIndex,
	// IsEnd, Accepts) directly on the local instead of going
	// through four separate kwg.KWG method calls per iteration —
	// each of which would do its own bounds-checked slice access
	// into the unexported k.nodes field. The accessor and the
	// bit-field constants come from word-golib (KWG.Nodes,
	// kwg.KWGNode*).
	nodes := gen.gaddag.Nodes()
	if curLetter != 0 {
		raw := uint32(curLetter.Unblank())
		var nnIdx uint32
		var accepts bool
		for i := nodeIdx; ; i++ {
			node := nodes[i]
			if node>>kwg.KWGNodeTileShift == raw {
				nnIdx = node & kwg.KWGNodeArcMask
				accepts = node&kwg.KWGNodeAcceptsBit != 0
				break
			}
			if node&kwg.KWGNodeIsEndBit != 0 {
				break
			}
		}
		gen.goOn(col, curLetter, rack, nnIdx, accepts, leftstrip, rightstrip, uniquePlay,
			baseScore+gen.letterDistribution.Score(curLetter),
			crossScores,
			wordMultiplier,
		)
	} else if !rack.Empty() {
		// Decode packed cachedSquare fields once into ints so the
		// arithmetic in the inner loop stays in native register width.
		lm := int(sq.letterMul)
		cs := int(sq.crossScore)
		wm := int(sq.wordMul)
		emptyAdjacent := crossSet == board.TrivialCrossSet
		// Hoist blank count: rack is always restored between iterations
		// so LetArr[0] is constant across the loop from this call's
		// perspective. The compiler can't see through the goOn call
		// boundary, so we hoist manually.
		nBlank := rack.LetArr[0]
		for i := nodeIdx; ; i++ {
			node := nodes[i]
			ml := tilemapping.MachineLetter(node >> kwg.KWGNodeTileShift)
			// Check cross-set before reading nMl: in restrictive
			// mid-game positions most nodes fail here, so deferring
			// the LetArr[ml] read avoids the load on the cold path.
			if ml != 0 && crossSet.Allowed(ml) {
				nMl := rack.LetArr[ml]
				if nMl == 0 && nBlank == 0 {
					if node&kwg.KWGNodeIsEndBit != 0 {
						break
					}
					continue
				}
				arcIdx := node & kwg.KWGNodeArcMask
				accepts := node&kwg.KWGNodeAcceptsBit != 0
				if nMl > 0 {
					rack.Take(ml)
					if gen.leavemap.Initialized {
						gen.leavemap.TakeLetter(ml, nMl-1)
					}
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
					if gen.leavemap.Initialized {
						gen.leavemap.AddLetter(ml, nMl-1)
					}
					rack.Add(ml)
				}
				// check blank
				if nBlank > 0 {
					rack.Take(0)
					if gen.leavemap.Initialized {
						gen.leavemap.TakeLetter(0, nBlank-1)
					}
					gen.tilesPlayed++
					// XXX: this won't work for non-zero-score blanks if
					// that's ever a thing.
					gen.goOn(col, ml.Blank(), rack, arcIdx, accepts, leftstrip, rightstrip, uniquePlay,
						baseScore,
						crossScores+cs*wm,
						wordMultiplier*wm)
					gen.tilesPlayed--
					if gen.leavemap.Initialized {
						gen.leavemap.AddLetter(0, nBlank-1)
					}
					rack.Add(0)
				}
			}
			if node&kwg.KWGNodeIsEndBit != 0 {
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
	// Hoist the per-column cache lookup to a local pointer so the
	// (up to three) field reads share a single bounds-checked
	// access. The Go compiler doesn't CSE these on its own across
	// the if/else branches.
	curSq := &gen.cache.squares[curCol]
	if curCol <= gen.curAnchorCol {
		if curSq.letter != 0 {
			gen.strip[curCol] = 0
		} else {
			gen.strip[curCol] = L
			if gen.vertical && curSq.crossSet == board.TrivialCrossSet {
				uniquePlay = true
			}
		}
		leftstrip = curCol

		noLetterDirectlyLeft := curCol == 0 ||
			gen.cache.squares[curCol-1].letter == 0

		if accepts && noLetterDirectlyLeft && gen.tilesPlayed > 0 {
			if (uniquePlay || gen.tilesPlayed > 1) && gen.tilesPlayed <= gen.maxTileUsage {
				gen.playRecorder(gen, rack, leftstrip, rightstrip, move.MoveTypePlay,
					baseScore*wordMultiplier+crossScores+bingoBonus)
			}
		}
		if newNodeIdx == 0 {
			return
		}
		if curCol > 0 && curCol-1 != gen.lastAnchorCol {
			gen.recursiveGen(curCol-1, rack, newNodeIdx, leftstrip, rightstrip, uniquePlay, baseScore, crossScores, wordMultiplier)
		}
		separationNodeIdx := gen.gaddag.NextNodeIdx(newNodeIdx, 0)
		if separationNodeIdx != 0 && noLetterDirectlyLeft && gen.curAnchorCol < gen.boardDim-1 {
			gen.recursiveGen(gen.curAnchorCol+1, rack, separationNodeIdx, leftstrip, rightstrip, uniquePlay, baseScore, crossScores, wordMultiplier)
		}

	} else {
		if curSq.letter != 0 {
			gen.strip[curCol] = 0
		} else {
			gen.strip[curCol] = L
			if gen.vertical && curSq.crossSet == board.TrivialCrossSet {
				uniquePlay = true
			}
		}
		rightstrip = curCol

		noLetterDirectlyRight := curCol == gen.boardDim-1 ||
			gen.cache.squares[curCol+1].letter == 0

		if accepts && noLetterDirectlyRight && gen.tilesPlayed > 0 {
			if (uniquePlay || gen.tilesPlayed > 1) && gen.tilesPlayed <= gen.maxTileUsage {
				gen.playRecorder(gen, rack, leftstrip, rightstrip, move.MoveTypePlay,
					baseScore*wordMultiplier+crossScores+bingoBonus)
			}
		}
		if newNodeIdx != 0 && curCol < gen.boardDim-1 {
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

// generateExchangeMoves generates exchange moves without duplicates.
// Maintains the leave map bitmask index as tiles are exchanged.
func (gen *GordonGenerator) generateExchangeMoves(rack *tilemapping.Rack, ml tilemapping.MachineLetter, stripidx int) {
	// magic function written by @andy-k
	for int(ml) < len(rack.LetArr) && rack.LetArr[ml] == 0 {
		ml++
	}
	if int(ml) == len(rack.LetArr) {
		gen.playRecorder(gen, rack, 0, stripidx, move.MoveTypeExchange, 0)
	} else {
		gen.generateExchangeMoves(rack, ml+1, stripidx)
		numthis := rack.LetArr[ml]
		for i := 0; i < numthis; i++ {
			if stripidx >= len(gen.exchangestrip) {
				log.Warn().
					Int("stripidx", stripidx).
					Int("exchangestrip_len", len(gen.exchangestrip)).
					Int("rack_tiles", int(rack.NumTiles())).
					Msg("rack exceeds maximum exchange limit, skipping invalid exchange moves")
				break
			}
			gen.exchangestrip[stripidx] = ml
			stripidx += 1
			rack.Take(ml)
			if gen.leavemap.Initialized {
				gen.leavemap.TakeLetter(ml, rack.LetArr[ml])
			}
			gen.generateExchangeMoves(rack, ml+1, stripidx)
		}
		for i := 0; i < numthis; i++ {
			if gen.leavemap.Initialized {
				gen.leavemap.AddLetter(ml, rack.LetArr[ml])
			}
			rack.Add(ml)
		}
	}
}

func (gen *GordonGenerator) GADDAG() *kwg.KWG {
	return gen.gaddag
}

// SetGADDAG replaces the underlying word graph used for move generation.
// Used by the endgame solver to swap in a pruned KWG for the duration of
// a solve and restore the original afterwards.
func (gen *GordonGenerator) SetGADDAG(g *kwg.KWG) {
	gen.gaddag = g
}
