package tinymove

// DefaultSmallMoveArenaSize is the initial capacity (in SmallMoves) for a
// SmallMoveArena. 65536 * 16 bytes = 1 MB, matching magpie's
// DEFAULT_INITIAL_SMALL_MOVE_ARENA_SIZE.
const DefaultSmallMoveArenaSize = 65536

// SmallMoveArena is a bump allocator for SmallMove slices.
// It uses LIFO (stack) discipline: Alloc advances the offset,
// Dealloc shrinks it. The backing slice grows as needed and is
// never freed — the GC handles it.
//
// Each thread in the endgame/pre-endgame solver gets its own arena,
// so no synchronization is needed.
type SmallMoveArena struct {
	buf    []SmallMove
	offset int
}

// NewSmallMoveArena returns an arena with the given initial capacity.
func NewSmallMoveArena(initialCap int) *SmallMoveArena {
	return &SmallMoveArena{
		buf: make([]SmallMove, initialCap),
	}
}

// Alloc returns a sub-slice of length n from the arena, bumping the offset.
// The returned slice shares the arena's backing array.
// If there is insufficient capacity the backing slice is doubled (or grown to
// fit), which is expected to be rare given the generous initial size.
func (a *SmallMoveArena) Alloc(n int) []SmallMove {
	needed := a.offset + n
	if needed > len(a.buf) {
		newCap := len(a.buf) * 2
		if newCap < needed {
			newCap = needed
		}
		newBuf := make([]SmallMove, newCap)
		copy(newBuf[:a.offset], a.buf[:a.offset])
		a.buf = newBuf
	}
	s := a.buf[a.offset : a.offset+n]
	a.offset += n
	return s
}

// Dealloc releases n elements from the top of the arena (LIFO).
func (a *SmallMoveArena) Dealloc(n int) {
	a.offset -= n
}

// Reset resets the arena to empty, logically freeing all allocations.
func (a *SmallMoveArena) Reset() {
	a.offset = 0
}
