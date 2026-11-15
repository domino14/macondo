package cross_set

import (
	"fmt"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/word-golib/tilemapping"
)

// A memory of cross-sets by the words beside the empty square.
//
// An empty square's cross-set and cross-score depend only on the tiles of
// the words on either side of it and on the lexicon. An endgame search plays
// and unplays the same few tiles over and over, so nearly every cross-set it
// asks the lexicon for is one it has worked out before. CachedGenerator
// remembers them.
//
// The key is the tiles themselves, not a hash of them: the word on the left
// read outwards from the square, a separator, and the word on the right, one
// byte a tile (with its blank flag, which the score depends on). A lane of a
// 15-square board has at most 14 tiles beside a square, so the key fits 16
// bytes exactly, and a hit is always for the very same words. Bigger boards
// are not cached.
//
// The idea is MAGPIE's (cross_set_cache.h).
//
// With extension sets switched off on the board
// (GameBoard.SetSkipExtensionSets), as the endgame solver has them, the
// cross-set and cross-score are all a square's update writes. With them on,
// as simulations have them, the update also writes the extension sets of
// the two words, which depend on the same key; those are remembered beside
// the entry, in a table made the first time it is needed.
//
// It is not safe for use by more than one goroutine: give each thread its
// own.

const (
	cacheMaxDim    = 15
	cacheSeparator = 0xFF
)

type cacheEntry struct {
	key      [2]uint64
	crossSet uint64
	score    int32
	// gen is the generation the entry was written in; see Reset.
	gen uint32
}

// cacheExt holds the extension sets an entry's update writes: the left
// word's left and right sets, and the right word's.
type cacheExt struct {
	// gen is the generation it was written in, with the entry of the same
	// index; 0 when that entry was written without extension sets.
	gen  uint32
	sets [4]uint64
}

// CachedGenerator is a GaddagCrossSetGenerator with a memory.
type CachedGenerator struct {
	Dist   *tilemapping.LetterDistribution
	Gaddag gaddag.WordGraph
	// Verify makes every hit recompute the cross-set from the lexicon and
	// panic if the cache disagrees.
	Verify bool

	entries []cacheEntry
	exts    []cacheExt
	shift   uint
	gen     uint32

	Hits, Misses uint64
}

// NewCachedGenerator makes a generator remembering 2^bits cross-sets.
func NewCachedGenerator(dist *tilemapping.LetterDistribution, gd gaddag.WordGraph,
	bits int) *CachedGenerator {
	return &CachedGenerator{
		Dist:    dist,
		Gaddag:  gd,
		entries: make([]cacheEntry, 1<<uint(bits)),
		shift:   uint(64 - bits),
		gen:     1,
	}
}

// Reset forgets everything and sets the lexicon to answer from. It must be
// called whenever the lexicon changes.
func (g *CachedGenerator) Reset(dist *tilemapping.LetterDistribution, gd gaddag.WordGraph) {
	g.Dist = dist
	g.Gaddag = gd
	g.Hits, g.Misses = 0, 0
	g.gen++
	if g.gen == 0 {
		// The counter wrapped: entries of generation 1 could be taken for
		// new ones, so clear them all.
		clear(g.entries)
		clear(g.exts)
		g.gen = 1
	}
}

func (g *CachedGenerator) Generate(b *Board, row int, col int, dir board.BoardDirection) {
	dim := b.Dim()
	skipExt := b.SkipExtensionSets()
	if dim > cacheMaxDim ||
		row < 0 || row >= dim || col < 0 || col >= dim {
		GenCrossSet(b, row, col, dir, g.Gaddag, g.Dist)
		return
	}
	// The row's squares, read straight from the board's: the square is at
	// pos and its neighbours along the row a step either way.
	squares := b.SquaresSlice()
	pos, step := b.GetSqIdx(row, col), b.GetSqIdx(0, 1)
	if squares[pos] != 0 ||
		((col == 0 || squares[pos-step] == 0) && (col == dim-1 || squares[pos+step] == 0)) {
		// A tile on the square, or none beside it.
		GenCrossSet(b, row, col, dir, g.Gaddag, g.Dist)
		return
	}
	// The key: left word outwards, separator, right word. Tiles are never
	// zero and never the separator, so the bytes decode one way only.
	var key [2]uint64
	n := uint(0)
	for c, i := col-1, pos-step; c >= 0; c, i = c-1, i-step {
		ml := squares[i]
		if ml == 0 {
			break
		}
		key[n>>3] |= uint64(ml) << (8 * (n & 7))
		n++
	}
	numLeft := int(n)
	key[n>>3] |= uint64(cacheSeparator) << (8 * (n & 7))
	n++
	for c, i := col+1, pos+step; c < dim; c, i = c+1, i+step {
		ml := squares[i]
		if ml == 0 {
			break
		}
		key[n>>3] |= uint64(ml) << (8 * (n & 7))
		n++
	}
	h := (key[0] ^ key[1]*0x9E3779B97F4A7C15) * 0xFF51AFD7ED558CCD
	h ^= h >> 32
	h *= 0xC4CEB9FE1A85EC53
	idx := h >> g.shift
	e := &g.entries[idx]
	// The words' edges: the left word's last tile is at col-1 and the right
	// word's first at col+1.
	numRight := int(n) - numLeft - 1
	leftCol, rightCol := col-numLeft, col+numRight
	tDir := throughDir(dir)
	var x *cacheExt
	if !skipExt {
		if g.exts == nil {
			g.exts = make([]cacheExt, len(g.entries))
		}
		x = &g.exts[idx]
	}
	if e.gen == g.gen && e.key == key && (skipExt || x.gen == g.gen) {
		g.Hits++
		if g.Verify {
			GenCrossSet(b, row, col, dir, g.Gaddag, g.Dist)
			if uint64(b.GetCrossSet(row, col, dir)) != e.crossSet ||
				b.GetCrossScore(row, col, dir) != int(e.score) {
				panic(fmt.Sprintf("cross-set cache disagrees at row %d col %d: "+
					"cached set %x score %d, computed set %x score %d",
					row, col, e.crossSet, e.score,
					uint64(b.GetCrossSet(row, col, dir)), b.GetCrossScore(row, col, dir)))
			}
			if !skipExt {
				if sets := readExtSets(b, row, col, numLeft, numRight, tDir); sets != x.sets {
					panic(fmt.Sprintf("cross-set cache disagrees at row %d col %d: "+
						"cached extension sets %x, computed %x", row, col, x.sets, sets))
				}
			}
			return
		}
		b.SetCrossSet(row, col, CrossSet(e.crossSet), dir)
		b.SetCrossScore(row, col, int(e.score), dir)
		if !skipExt {
			// The writes GenCrossSet makes, in its order.
			if numLeft > 0 {
				b.SetLeftExtSet(row, col-1, tDir, x.sets[0])
				b.SetRightExtSet(row, col-1, tDir, x.sets[1])
				if leftCol > 0 {
					b.SetLeftExtSet(row, leftCol-1, tDir, x.sets[0])
				}
			}
			if numRight > 0 {
				b.SetLeftExtSet(row, rightCol, tDir, x.sets[2])
				b.SetRightExtSet(row, rightCol, tDir, x.sets[3])
				b.SetLeftExtSet(row, col, tDir, x.sets[2])
			}
		}
		return
	}
	g.Misses++
	GenCrossSet(b, row, col, dir, g.Gaddag, g.Dist)
	e.key = key
	e.crossSet = uint64(b.GetCrossSet(row, col, dir))
	e.score = int32(b.GetCrossScore(row, col, dir))
	e.gen = g.gen
	if !skipExt {
		x.gen = g.gen
		x.sets = readExtSets(b, row, col, numLeft, numRight, tDir)
	} else if g.exts != nil {
		g.exts[idx].gen = 0
	}
}

// readExtSets reads back the extension sets GenCrossSet has just written
// for the empty square at (row, col) with numLeft tiles before it and
// numRight after: the left word's left and right sets, from its last tile,
// and the right word's, from its first.
func readExtSets(b *Board, row, col, numLeft, numRight int, tDir board.BoardDirection) (sets [4]uint64) {
	if numLeft > 0 {
		sets[0] = b.GetLeftExtSet(row, col-1, tDir)
		sets[1] = b.GetRightExtSet(row, col-1, tDir)
	}
	if numRight > 0 {
		sets[2] = b.GetLeftExtSet(row, col+numRight, tDir)
		sets[3] = b.GetRightExtSet(row, col+numRight, tDir)
	}
	return sets
}

func (g *CachedGenerator) GenerateAll(b *Board) {
	generateAll(g, b)
}

func (g *CachedGenerator) UpdateForMove(b *Board, m *move.Move) {
	updateForMove(g, b, m)
}

func (g *CachedGenerator) UpdateForSmallMove(b *Board, m *tinymove.SmallMove,
	moveTiles *[board.MaxBoardDim]tilemapping.MachineLetter) {
	updateForSmallMove(g, b, m, moveTiles)
}
