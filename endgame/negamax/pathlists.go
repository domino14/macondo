package negamax

import (
	"fmt"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tinymove"
)

// Incremental move lists.
//
// Generating every play of a position from scratch is most of what a search
// node and a leaf playout cost, yet one move changes little: the plays of a
// row or column (a "lane") depend only on that lane's squares -- their tiles,
// cross-sets, cross-scores and anchors -- and on the rack. A move changes
// those for a handful of lanes and leaves the rest alone. So each thread keeps
// the lists of the positions along its current path, split by lane, and
// derives a position's list from the nearest ancestor's list for the same
// player: the lanes the moves in between touched are generated again, and the
// others are copied, dropping the plays whose tiles the player has since
// played. The result is the list GenAll would produce, in the same order,
// without the pass.
//
// The idea is MAGPIE's (path_move_lists.c); this version works on whole
// lanes.

const (
	// maxLanes is rows plus columns of the largest board.
	maxLanes = 2 * board.MaxBoardDim
	// tileCountWords holds 4 bits per tile type.
	tileCountWords = 3
	// maxListTileTypes is the largest alphabet (blank included) the packed
	// counts hold.
	maxListTileTypes = tileCountWords * 16
	// tileCountHighBits has the top bit of every 4-bit field set.
	tileCountHighBits = 0x8888888888888888
)

// tileCounts is a multiset of tiles, 4 bits per tile type (the blank is type
// 0). A rack or a play holds at most 7 of a type, so the top bit of each field
// is free for the borrow test in holds.
type tileCounts [tileCountWords]uint64

// rackTileCounts packs a rack with the top bit of every field set, ready for
// holds.
func rackTileCounts(rack *tilemapping.Rack) tileCounts {
	var c tileCounts
	for i := range c {
		c[i] = tileCountHighBits
	}
	for ml, n := range rack.LetArr {
		if n > 0 {
			c[ml>>4] += uint64(n) << (uint(ml&15) * 4)
		}
	}
	return c
}

// playTileCounts packs the tiles a play takes from the rack.
func playTileCounts(m *tinymove.SmallMove) tileCounts {
	var c tileCounts
	n := m.TilesPlayed()
	tm := uint64(m.TinyMove())
	for i := 0; i < n; i++ {
		ml := (tm >> (20 + 6*uint(i))) & 63
		if tm&(1<<(12+uint(i))) != 0 {
			ml = 0
		}
		c[ml>>4] += 1 << ((ml & 15) * 4)
	}
	return c
}

// holds reports whether the rack (from rackTileCounts) holds the tiles of
// used. Each field of the rack is 8 plus its count; subtracting a count no
// larger leaves the 8, and a larger one clears it without borrowing from the
// next field.
func (rack *tileCounts) holds(used *tileCounts) bool {
	for i := range rack {
		if (rack[i]-used[i])&tileCountHighBits != tileCountHighBits {
			return false
		}
	}
	return true
}

// moveList is one player's tile plays in one position, by lane: lane L's are
// moves[laneBegin[L]:laneBegin[L+1]]. Lane r < dim is row r's horizontal
// plays and lane dim+c is column c's vertical plays, which is the order
// GenAll emits them in.
type moveList struct {
	moves     []tinymove.SmallMove
	used      []tileCounts
	laneBegin [maxLanes + 1]int32
	valid     bool
}

// pathLevel is what the lists keep for one position on the path: both
// players' lists, if they have been asked for, and the move played from it.
type pathLevel struct {
	lists [2]moveList
	// hits has a bit for each lane whose plays the move played from this
	// position changes.
	hits uint64
	// mover is the player who made that move, and tilePlay whether it placed
	// tiles (and so shrank their rack).
	mover    int8
	tilePlay bool
	// changed lists the empty squares whose cross-sets, cross-scores or
	// neighbours that move changes (row<<8 | col): what a one-tile play on
	// a square depends on.
	changed  [maxChangedSquares]uint16
	nChanged uint8
	// lone caches each player's best one-tile play in this position, for a
	// player down to a single tile.
	lone [2]lonePlay
}

// maxChangedSquares bounds pathLevel.changed: the two ends of the word, and
// the two ends of the crossing run through each of up to seven tiles.
const maxChangedSquares = 16

// lonePlay is the best play of a player's last tile in one position: the
// highest-scoring square, the first of those in row order, as scanLoneTile
// finds it.
type lonePlay struct {
	valid bool
	found bool
	tile  tilemapping.MachineLetter
	row   int8
	col   int8
	score int32
}

// pathLists is one thread's incremental move lists. Levels are indexed by the
// game's backup-stack depth, so undoing a move needs no bookkeeping: the game
// tells the lists about each move as it is played (see onMove), which marks
// the lists of the level it leads to as stale.
type pathLists struct {
	g      *game.Game
	mg     movegen.MoveGenerator
	levels []pathLevel
	// base is the stack depth the lists were bound at. The moves played
	// before then were not seen, so no list is derived across it.
	base int
	dim  int
	ends [maxLanes]int32
	// check compares every derived list with a full generation.
	check bool
}

// bind attaches the lists to a game and its move generator and forgets
// everything they held.
func (p *pathLists) bind(g *game.Game, mg movegen.MoveGenerator, check bool) {
	p.release()
	p.g = g
	p.mg = mg
	p.dim = g.Board().Dim()
	p.base = g.StateStackDepth()
	p.check = check
	for i := range p.levels {
		p.levels[i].lists[0].valid = false
		p.levels[i].lists[1].valid = false
		p.levels[i].lone[0].valid = false
		p.levels[i].lone[1].valid = false
	}
	g.SetSmallMoveHook(p.onMove)
}

// release detaches the lists from their game.
func (p *pathLists) release() {
	if p.g != nil {
		p.g.SetSmallMoveHook(nil)
		p.g = nil
		p.mg = nil
	}
}

func (p *pathLists) grow(n int) {
	for len(p.levels) < n {
		p.levels = append(p.levels, pathLevel{})
	}
}

// onMove is the game's small-move hook: it runs just before m is played.
func (p *pathLists) onMove(m *tinymove.SmallMove) {
	n := p.g.StateStackDepth()
	p.grow(n + 2)
	lv := &p.levels[n]
	lv.mover = int8(p.g.PlayerOnTurn())
	lv.tilePlay = !m.IsPass()
	// A player holding one tile has its best play kept at every level of
	// the path, so that each position's is derived from its parent's.
	for player := 0; player < 2; player++ {
		if rack := p.g.RackFor(player); rack.NumTiles() == 1 {
			p.lonePlayAt(n, player, loneTile(rack))
		}
	}
	lv.hits = p.moveHits(m, lv)
	next := &p.levels[n+1]
	next.lists[0].valid = false
	next.lists[1].valid = false
	next.lone[0].valid = false
	next.lone[1].valid = false
}

// bestLoneTilePlay is bestSingleTilePlay for the player's one tile, worked
// out from the nearest position up the path that has it: only the squares
// the moves since then changed can have become a better play.
func (p *pathLists) bestLoneTilePlay(player int) (tinymove.SmallMove, bool) {
	n := p.g.StateStackDepth()
	p.grow(n + 1)
	tile := loneTile(p.g.RackFor(player))
	c := p.lonePlayAt(n, player, tile)
	if !c.found {
		return tinymove.SmallMove{}, false
	}
	return loneTileMove(p.g.Board(), tile, p.g.Bag().LetterDistribution(),
		int(c.row), int(c.col), int(c.score)), true
}

// lonePlayAt returns level n's cached play of the player's one tile, filling
// it in if needed. The game must be at level n.
func (p *pathLists) lonePlayAt(n, player int, tile tilemapping.MachineLetter) *lonePlay {
	c := &p.levels[n].lone[player]
	if c.valid && c.tile == tile {
		return c
	}
	b := p.g.Board()
	ld := p.g.Bag().LetterDistribution()
	c.valid = true
	c.tile = tile
	// The nearest ancestor that has the play. The player has not moved since
	// then (it would hold a different rack), so the tile is the same.
	src := -1
	for j := n - 1; j >= p.base; j-- {
		lv := &p.levels[j]
		if lv.lone[player].valid && lv.lone[player].tile == tile {
			src = j
			break
		}
		if lv.tilePlay && int(lv.mover) == player {
			break
		}
	}
	if src >= 0 {
		old := &p.levels[src].lone[player]
		derived := true
		if old.found {
			// The old best stands unless a tile now covers its square or it
			// is one of the changed squares.
			if b.HasLetter(int(old.row), int(old.col)) {
				derived = false
			}
			key := uint16(old.row)<<8 | uint16(old.col)
			for j := src; j < n && derived; j++ {
				lv := &p.levels[j]
				for i := 0; i < int(lv.nChanged); i++ {
					if lv.changed[i] == key {
						derived = false
						break
					}
				}
			}
		}
		if derived {
			c.found, c.row, c.col, c.score = old.found, old.row, old.col, old.score
			tileFace := loneTileFace(tile, ld)
			for j := src; j < n; j++ {
				lv := &p.levels[j]
				for i := 0; i < int(lv.nChanged); i++ {
					r, col := int(lv.changed[i]>>8), int(lv.changed[i]&0xff)
					if b.HasLetter(r, col) {
						continue
					}
					score, ok := loneTileAt(b, tile, tileFace, r, col)
					if !ok {
						continue
					}
					// Higher score, or the same score earlier in row order.
					if !c.found || int32(score) > c.score || (int32(score) == c.score &&
						(r < int(c.row) || (r == int(c.row) && col < int(c.col)))) {
						c.found = true
						c.row, c.col, c.score = int8(r), int8(col), int32(score)
					}
				}
			}
			if p.check {
				score, row, col, found := scanLoneTile(b, tile, ld)
				if found != c.found || (found && (int32(score) != c.score ||
					int8(row) != c.row || int8(col) != c.col)) {
					panic(fmt.Sprintf("derived one-tile play (%v %d,%d for %d) differs from the scan (%v %d,%d for %d)",
						c.found, c.row, c.col, c.score, found, row, col, score))
				}
			}
			return c
		}
	}
	score, row, col, found := scanLoneTile(b, tile, ld)
	c.found = found
	c.row, c.col, c.score = int8(row), int8(col), int32(score)
	return c
}

// moveHits returns the lanes whose plays the move changes, from the board
// before the move is played. A lane's plays change when one of its squares
// gains a tile, or has its cross-set, cross-score or anchor changed, which
// happens only to the empty squares at the two ends of a run of tiles the
// move extends or creates:
//
//   - the move's own lane;
//   - for every tile placed, the crossing lane through its square, and the
//     lanes (parallel to the move's) of the empty squares at the two ends of
//     the crossing run of tiles through it;
//   - the crossing lanes of the empty squares at the two ends of the word.
func (p *pathLists) moveHits(m *tinymove.SmallMove, lv *pathLevel) uint64 {
	lv.nChanged = 0
	if m.IsPass() {
		return 0
	}
	b := p.g.Board()
	dim := p.dim
	row, col, vertical := m.CoordsAndVertical()
	length := m.PlayLength()
	// The move runs along `along` in lane `fixed`; work in those coordinates
	// and map back to squares with at().
	fixed, start := row, col
	ownBase, crossBase := 0, dim
	if vertical {
		fixed, start = col, row
		ownBase, crossBase = dim, 0
	}
	hasLetter := func(lane, pos int) bool {
		if vertical {
			return b.HasLetter(pos, lane)
		}
		return b.HasLetter(lane, pos)
	}
	// noteChanged records the square at (lane, pos) for lv.changed.
	noteChanged := func(lane, pos int) {
		if vertical {
			lane, pos = pos, lane
		}
		lv.changed[lv.nChanged] = uint16(lane)<<8 | uint16(pos)
		lv.nChanged++
	}
	hits := uint64(1) << uint(ownBase+fixed)
	for i := 0; i < length; i++ {
		pos := start + i
		if hasLetter(fixed, pos) {
			// A tile already on the board that the play goes through.
			continue
		}
		hits |= 1 << uint(crossBase+pos)
		lo := fixed - 1
		for lo >= 0 && hasLetter(lo, pos) {
			lo--
		}
		if lo >= 0 {
			hits |= 1 << uint(ownBase+lo)
			noteChanged(lo, pos)
		}
		hi := fixed + 1
		for hi < dim && hasLetter(hi, pos) {
			hi++
		}
		if hi < dim {
			hits |= 1 << uint(ownBase+hi)
			noteChanged(hi, pos)
		}
	}
	if start > 0 {
		hits |= 1 << uint(crossBase+start-1)
		noteChanged(fixed, start-1)
	}
	if start+length < dim {
		hits |= 1 << uint(crossBase+start+length)
		noteChanged(fixed, start+length)
	}
	return hits
}

// get returns the player's tile plays in the game's current position, without
// the pass. The list belongs to the path lists and stays as it is until a
// move is played into this position's level again; callers must not change
// it.
func (p *pathLists) get(player int) *moveList {
	n := p.g.StateStackDepth()
	p.grow(n + 1)
	l := &p.levels[n].lists[player]
	if l.valid {
		return l
	}
	// The nearest ancestor that has this player's list, and what the moves
	// since then changed.
	var src *moveList
	var mask uint64
	rackChanged := false
	for j := n - 1; j >= p.base; j-- {
		lv := &p.levels[j]
		mask |= lv.hits
		if lv.tilePlay && int(lv.mover) == player {
			rackChanged = true
		}
		if lv.lists[player].valid {
			src = &lv.lists[player]
			break
		}
	}
	numLanes := 2 * p.dim
	if src == nil {
		mask = uint64(1)<<uint(numLanes) - 1
	}
	rack := p.g.RackFor(player)
	var generated []tinymove.SmallMove
	if mask != 0 {
		generated = p.mg.GenLanes(rack, mask, &p.ends)
	}
	rackCounts := rackTileCounts(rack)
	l.moves = l.moves[:0]
	l.used = l.used[:0]
	genPos := 0
	for lane := 0; lane < numLanes; lane++ {
		l.laneBegin[lane] = int32(len(l.moves))
		if mask&(1<<uint(lane)) != 0 {
			end := int(p.ends[lane])
			for i := genPos; i < end; i++ {
				l.moves = append(l.moves, generated[i])
				l.used = append(l.used, playTileCounts(&generated[i]))
			}
			genPos = end
			continue
		}
		from, to := src.laneBegin[lane], src.laneBegin[lane+1]
		if !rackChanged {
			l.moves = append(l.moves, src.moves[from:to]...)
			l.used = append(l.used, src.used[from:to]...)
			continue
		}
		for i := from; i < to; i++ {
			if rackCounts.holds(&src.used[i]) {
				l.moves = append(l.moves, src.moves[i])
				l.used = append(l.used, src.used[i])
			}
		}
	}
	l.laneBegin[numLanes] = int32(len(l.moves))
	l.valid = true
	if p.check {
		p.checkAgainstFullGeneration(l, player)
	}
	return l
}

// checkAgainstFullGeneration panics unless the list is exactly what GenAll
// produces for the player. For tests.
func (p *pathLists) checkAgainstFullGeneration(l *moveList, player int) {
	p.mg.GenAll(p.g.RackFor(player), false)
	full := p.mg.SmallPlays()
	if n := len(full); n > 0 && full[n-1].IsPass() {
		full = full[:n-1]
	}
	if len(full) != len(l.moves) {
		panic(fmt.Sprintf("incremental move list has %d plays, full generation %d (depth %d, player %d)",
			len(l.moves), len(full), p.g.StateStackDepth(), player))
	}
	for i := range full {
		if full[i] != l.moves[i] {
			panic(fmt.Sprintf("incremental move list differs at %d: %s vs %s (depth %d, player %d)",
				i, l.moves[i].ShortDescription(), full[i].ShortDescription(),
				p.g.StateStackDepth(), player))
		}
	}
}
