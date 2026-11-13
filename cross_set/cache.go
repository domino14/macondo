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
// A CachedGenerator caches only while the board has extension sets switched
// off (GameBoard.SetSkipExtensionSets), as the endgame solver does, since
// then the cross-set and cross-score are all a square's update writes;
// otherwise it behaves as a GaddagCrossSetGenerator. It is not safe for use
// by more than one goroutine: give each thread its own.

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

// CachedGenerator is a GaddagCrossSetGenerator with a memory.
type CachedGenerator struct {
	Dist   *tilemapping.LetterDistribution
	Gaddag gaddag.WordGraph
	// Verify makes every hit recompute the cross-set from the lexicon and
	// panic if the cache disagrees.
	Verify bool

	entries []cacheEntry
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
		g.gen = 1
	}
}

func (g *CachedGenerator) Generate(b *Board, row int, col int, dir board.BoardDirection) {
	dim := b.Dim()
	if !b.SkipExtensionSets() || dim > cacheMaxDim ||
		row < 0 || row >= dim || col < 0 || col >= dim ||
		b.HasLetter(row, col) || b.LeftAndRightEmpty(row, col) {
		GenCrossSet(b, row, col, dir, g.Gaddag, g.Dist)
		return
	}
	// The key: left word outwards, separator, right word. Tiles are never
	// zero and never the separator, so the bytes decode one way only.
	var key [2]uint64
	n := uint(0)
	for c := col - 1; c >= 0; c-- {
		ml := b.GetLetter(row, c)
		if ml == 0 {
			break
		}
		key[n>>3] |= uint64(ml) << (8 * (n & 7))
		n++
	}
	key[n>>3] |= uint64(cacheSeparator) << (8 * (n & 7))
	n++
	for c := col + 1; c < dim; c++ {
		ml := b.GetLetter(row, c)
		if ml == 0 {
			break
		}
		key[n>>3] |= uint64(ml) << (8 * (n & 7))
		n++
	}
	h := (key[0] ^ key[1]*0x9E3779B97F4A7C15) * 0xFF51AFD7ED558CCD
	h ^= h >> 32
	h *= 0xC4CEB9FE1A85EC53
	e := &g.entries[h>>g.shift]
	if e.gen == g.gen && e.key == key {
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
			return
		}
		b.SetCrossSet(row, col, CrossSet(e.crossSet), dir)
		b.SetCrossScore(row, col, int(e.score), dir)
		return
	}
	g.Misses++
	GenCrossSet(b, row, col, dir, g.Gaddag, g.Dist)
	e.key = key
	e.crossSet = uint64(b.GetCrossSet(row, col, dir))
	e.score = int32(b.GetCrossScore(row, col, dir))
	e.gen = g.gen
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
