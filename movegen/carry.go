// carry.go: a best-play generator's memory of what it found for a rack on
// earlier boards. After MAGPIE's best_move_carry.
//
// In a simulation the same rack is generated for again and again on boards
// that differ in a few lanes: every candidate play is answered from the same
// opponent rack, and the racks drawn after often match too. Which plays a
// lane (a row, or a column) holds for a rack, and what each is worth, is
// decided by the lane alone: its tiles, its anchors, and the cross-sets and
// cross-scores of its squares, with the number of tiles in the bag for the
// leave values. So when a generation has searched a lane and its best play
// overall was M, a later generation for the same rack and bag that finds the
// lane unchanged knows that no play of the lane is better than M. If its
// own best play so far is at least M, the lane is not looked at: no bound
// is worked out for it and nothing is generated from it.
//
// The best plays themselves are remembered too, a few for each rack. One
// that is still a play of the new board (see carryRevalidate) is scored
// there, and the generation starts from the best of them: anchors that
// cannot reach it are dropped as they are found, and more lanes are skipped.
//
// Lanes are matched on their whole contents, not on a hash of them, and "at
// least M" is the recorder's own order (equity, then Move.TiebreaksBetter),
// so the play found is the one a generation without the memory finds.
package movegen

import (
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/leavemap"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/wmp"
)

const (
	// carryRacks is the number of racks remembered, each in the slot its
	// hash picks; a new rack replaces the one there.
	carryRacks = 64
	// carryMovesPerRack is the number of best plays kept for a rack, the
	// oldest replaced first.
	carryMovesPerRack = 8
	// carryBoundsPerLane is the number of different contents a lane is
	// remembered with, the oldest replaced first.
	carryBoundsPerLane = 4
	// carryMaxAlphabet bounds the tile types of a rack that can be
	// remembered.
	carryMaxAlphabet = 64
)

// carriedLane says: with these contents, the lane holds no play better than
// the rack's remembered move moveIdx.
type carriedLane struct {
	hash    uint64 // of the contents; never 0
	epoch   uint32 // the rack entry's epoch when written
	stamp   uint32 // the move slot's stamp when written
	anchors uint32
	moveIdx uint8
}

type carriedRack struct {
	// epoch is 0 for an unused entry. Lane bounds written under another
	// epoch are dead.
	epoch      uint32
	tilesInBag int
	numTypes   int
	counts     [carryMaxAlphabet]uint8
	moves      [carryMovesPerRack]move.Move
	// through[i] is the tiles of the lane of moves[i] under it, by position
	// along the move, when it was last a generation's best play. The move's
	// own tiles mark squares played through only as such, and on another
	// board they may hold other tiles.
	through [carryMovesPerRack][maxBoardDim]tilemapping.MachineLetter
	// stamps[i] changes whenever moves[i] is replaced.
	stamps    [carryMovesPerRack]uint32
	numMoves  int
	nextMove  int
	nextBound [2][maxBoardDim]uint8
	// What a generation works out for the rack before it looks at the
	// board (see setUpRack), once a generation has: the leave map, the best
	// leaves by size and the word map generator's subracks.
	hasSetup   bool
	leavemap   leavemap.LeaveMap
	bestLeaves [game.RackTileLimit + 1]float64
	wmpState   wmp.RackState
	// lanes[(dir*dim+row)*carryBoundsPerLane+i] is the i'th bound of a lane,
	// and its contents are the dim squares of squares from that index times
	// dim.
	lanes   []carriedLane
	squares []cachedSquare
}

func (e *carriedRack) bound(dim, dir, row, i int) (*carriedLane, []cachedSquare) {
	idx := (dir*dim+row)*carryBoundsPerLane + i
	return &e.lanes[idx], e.squares[idx*dim : idx*dim+dim]
}

type bestMoveCarry struct {
	racks     [carryRacks]carriedRack
	nextEpoch uint32
	// dim is the board dimension the entries are laid out for.
	dim int

	// For the generation in progress: the rack's entry (nil when the
	// generation does not use the memory), and by lane its hash (0 for a
	// lane without anchors), its anchors, the remembered bound with the
	// same contents (-1 for none) and whether it is skipped.
	cur     *carriedRack
	scratch move.Move
	hashes  [2][maxBoardDim]uint64
	anchors [2][maxBoardDim]uint32
	found   [2][maxBoardDim]int8
	skip    [2][maxBoardDim]bool

	// Counts, for tests and measurements.
	generations, rackHits, lanesSeen, lanesSkipped, seeded uint64
}

// SetBestMoveCarry switches the generator's memory of best plays by rack
// (see the top of carry.go). It is used only by best-play generations with
// tiles in the bag, and changes how fast they run, not what they find.
// Switching it on, or changing what the generator generates with (game,
// lexicon, word map, equity calculators), forgets everything.
func (gen *GordonGenerator) SetBestMoveCarry(on bool) {
	if !on {
		gen.carry = nil
		return
	}
	// The entries are laid out for the board at the first generation.
	gen.carry = &bestMoveCarry{}
}

// BestMoveCarryStats returns how many generations used the memory, how many
// of them found their rack in it, how many lanes with anchors those saw,
// how many of these were skipped, and how many generations started from a
// remembered play.
func (gen *GordonGenerator) BestMoveCarryStats() (generations, rackHits, lanesSeen, lanesSkipped, seeded uint64) {
	c := gen.carry
	if c == nil {
		return
	}
	return c.generations, c.rackHits, c.lanesSeen, c.lanesSkipped, c.seeded
}

// forgetCarry empties the memory, if there is one.
func (gen *GordonGenerator) forgetCarry() {
	c := gen.carry
	if c == nil {
		return
	}
	for i := range c.racks {
		c.racks[i].epoch = 0
	}
	c.cur = nil
}

// carryBegin decides whether this generation uses the memory, and if so
// finds or starts the rack's entry.
func (gen *GordonGenerator) carryBegin(rack *tilemapping.Rack) {
	c := gen.carry
	if c == nil {
		return
	}
	c.cur = nil
	// Only where a play is priced as its score plus the leave map's value
	// (see TopPlayOnlyRecorder), which with tiles in the bag depends on
	// nothing but the rack, the leave and the bag's size.
	// This runs before the leave map is filled; these are the conditions
	// under which it will be.
	if !gen.bestPlayOnly || gen.klv == nil || gen.game == nil ||
		rack.NumTiles() == 0 || rack.NumTiles() > leavemap.MaxRackSize ||
		gen.board.IsEmpty() || gen.tilesInBag <= 0 || gen.isWordSmog ||
		len(rack.LetArr) > carryMaxAlphabet {
		return
	}
	// A bag with at least as many tiles as there are pre-endgame values
	// gives every play the leave value alone, however many it has.
	bagKey := gen.tilesInBag
	if bagKey > len(gen.pegValues) {
		bagKey = len(gen.pegValues)
	}
	if c.dim != gen.boardDim {
		gen.forgetCarry()
		c.dim = gen.boardDim
		for i := range c.racks {
			c.racks[i].lanes = make([]carriedLane, 2*c.dim*carryBoundsPerLane)
			c.racks[i].squares = make([]cachedSquare, 2*c.dim*carryBoundsPerLane*c.dim)
		}
		// Room for every remembered play's tiles and leave, so that
		// remembering one never allocates.
		buf := make([]tilemapping.MachineLetter, (carryRacks*carryMovesPerRack+1)*2*c.dim)
		for i := range c.racks {
			for j := range c.racks[i].moves {
				c.racks[i].moves[j].Reserve(buf, c.dim)
				buf = buf[2*c.dim:]
			}
		}
		c.scratch.Reserve(buf, c.dim)
	}
	hash := uint64(bagKey) + 0x9E3779B97F4A7C15
	for _, n := range rack.LetArr {
		if n > 255 {
			return
		}
		hash = (hash ^ uint64(n)) * 0xFF51AFD7ED558CCD
		hash ^= hash >> 29
	}
	entry := &c.racks[hash>>32&(carryRacks-1)]
	c.generations++
	same := entry.epoch != 0 && entry.tilesInBag == bagKey &&
		entry.numTypes == len(rack.LetArr)
	if same {
		for ml, n := range rack.LetArr {
			if entry.counts[ml] != uint8(n) {
				same = false
				break
			}
		}
	}
	if same {
		c.rackHits++
	} else {
		c.nextEpoch++
		if c.nextEpoch == 0 {
			// Wrapped: old bounds could pass for new ones.
			for i := range c.racks {
				c.racks[i].epoch = 0
			}
			c.nextEpoch = 1
		}
		entry.epoch = c.nextEpoch
		entry.tilesInBag = bagKey
		entry.numTypes = len(rack.LetArr)
		for ml, n := range rack.LetArr {
			entry.counts[ml] = uint8(n)
		}
		entry.numMoves = 0
		entry.nextMove = 0
		entry.nextBound = [2][maxBoardDim]uint8{}
		entry.hasSetup = false
	}
	c.cur = entry
}

// carrySaveSetup remembers what setUpRack has just worked out for the rack.
func (gen *GordonGenerator) carrySaveSetup() {
	c := gen.carry
	if !gen.leavemap.Initialized {
		// Not a generation the memory is for after all.
		c.cur = nil
		return
	}
	entry := c.cur
	entry.leavemap = gen.leavemap
	entry.bestLeaves = gen.shadow.bestLeaves
	if gen.wmpMoveGen.IsActive() {
		gen.wmpMoveGen.SaveRackState(&entry.wmpState)
	}
	entry.hasSetup = true
}

// carryLoadSetup is setUpRack for a remembered rack. The leave values are
// the same: they depend on the rack and, through the pre-endgame values, on
// the bag's size, which the entry is for. So are the subracks' word map
// entries: the word map has not changed since (setWMP forgets everything).
func (gen *GordonGenerator) carryLoadSetup(rack *tilemapping.Rack) {
	entry := gen.carry.cur
	gen.leavemap = entry.leavemap
	gen.shadow.bestLeaves = entry.bestLeaves
	gen.wmpMoveGen.Init(gen.letterDistribution, rack, gen.wmpData)
	if gen.wmpMoveGen.IsActive() {
		gen.wmpMoveGen.LoadRackState(&entry.wmpState)
	}
}

// carryHashLane hashes a lane's contents: its anchors, and each square's
// tile, cross-set and cross-score. The multipliers belong to the lane, and
// the extension sets follow from its tiles. Never 0.
func carryHashLane(squares *[maxBoardDim]cachedSquare, dim int, anchors uint32) uint64 {
	h1 := uint64(0x9E3779B97F4A7C15) ^ uint64(anchors)
	h2 := uint64(0xC2B2AE3D27D4EB4F)
	for col := 0; col < dim; col++ {
		sq := &squares[col]
		h1 = (h1 ^ uint64(sq.crossSet)) * 0xFF51AFD7ED558CCD
		h1 ^= h1 >> 29
		h2 = (h2 ^ (uint64(uint32(sq.crossScore)) | uint64(sq.letter)<<32)) * 0xFF51AFD7ED558CCD
		h2 ^= h2 >> 29
	}
	h := (h1 ^ h2>>32 ^ h2<<32) * 0xC4CEB9FE1A85EC53
	h ^= h >> 32
	return h | 1
}

// carrySameLane reports whether the remembered lane has exactly these
// contents.
func carrySameLane(b *carriedLane, kept []cachedSquare, squares *[maxBoardDim]cachedSquare, anchors uint32) bool {
	if b.anchors != anchors {
		return false
	}
	for col := range kept {
		p, q := &kept[col], &squares[col]
		if p.crossSet != q.crossSet || p.letter != q.letter || p.crossScore != q.crossScore {
			return false
		}
	}
	return true
}

// carryNoteLane is called once for each lane of the generation, with the
// lane loaded into gen.shadow.lanes if it has anchors. It finds the lane's
// remembered bound.
func (gen *GordonGenerator) carryNoteLane(dir, row int, anchors uint32) {
	c := gen.carry
	c.anchors[dir][row] = anchors
	c.found[dir][row] = -1
	c.skip[dir][row] = false
	if anchors == 0 {
		c.hashes[dir][row] = 0
		return
	}
	squares := &gen.shadow.lanes[dir][row]
	hash := carryHashLane(squares, gen.boardDim, anchors)
	c.hashes[dir][row] = hash
	c.lanesSeen++
	entry := c.cur
	for i := 0; i < carryBoundsPerLane; i++ {
		b, kept := entry.bound(gen.boardDim, dir, row, i)
		if b.hash == hash && b.epoch == entry.epoch &&
			entry.stamps[b.moveIdx] == b.stamp &&
			carrySameLane(b, kept, squares, anchors) {
			c.found[dir][row] = int8(i)
			return
		}
	}
}

// carryNoWorse reports whether a is at least b in the recorder's order.
func carryNoWorse(a, b *move.Move) bool {
	if a.Equity() != b.Equity() {
		return a.Equity() > b.Equity()
	}
	return !b.TiebreaksBetter(a)
}

// scoreLanePlay is the score of tiles (0 for a square played through) laid
// in a lane from startCol, tilesPlayed of them from the rack.
func scoreLanePlay(ld *tilemapping.LetterDistribution, squares *[maxBoardDim]cachedSquare,
	tiles []tilemapping.MachineLetter, startCol, tilesPlayed int) int {
	var playedScoreTotal int
	var playthroughScoreTotal int
	var hookedCrossTotal int
	var playedCrossTotal int
	wordMultiplier := 1

	for i, ml := range tiles {
		sq := &squares[startCol+i]
		if ml == 0 {
			// Playthrough: score the existing board tile (no
			// multipliers; blanked board tiles score 0 via ld.Score).
			playthroughScoreTotal += ld.Score(sq.letter)
			continue
		}
		// Newly placed tile. Blanked placements score 0 via ld.Score.
		tileScore := ld.Score(ml)
		twm := int(sq.wordMul)
		cs := int(sq.crossScore)
		hookedCrossTotal += cs * twm
		isCrossWord := cs > 0 || sq.crossSet != board.TrivialCrossSet
		lm := int(sq.letterMul)
		if isCrossWord {
			playedCrossTotal += tileScore * twm * lm
		}
		playedScoreTotal += tileScore * lm
		wordMultiplier *= twm
	}

	bingoBonus := 0
	if tilesPlayed == game.RackTileLimit {
		bingoBonus = 50
	}

	return (playedScoreTotal+playthroughScoreTotal)*wordMultiplier +
		hookedCrossTotal + playedCrossTotal + bingoBonus
}

// carryRevalidate reports whether the rack's remembered play i is a play of
// this board, and if so its score and equity here. It is one when its lane
// has the tiles it was played through and none just before or after it,
// the squares it fills are empty and allow its tiles, and it covers an
// anchor: it then spells the word it spelled, with valid cross words, and
// the generator finds every such placement of a word. Its leave is what it
// was, since the rack is the same, and the entry is for this bag. A one-tile
// play down a column is one only if its tile makes no word across.
func (gen *GordonGenerator) carryRevalidate(rack *tilemapping.Rack, i int) (ok bool, score int, eq float64) {
	c := gen.carry
	entry := c.cur
	m := &entry.moves[i]
	if m.Action() != move.MoveTypePlay {
		return false, 0, 0
	}
	row, col, vertical := m.CoordsAndVertical()
	dir, lane, start := 0, row, col
	if vertical {
		dir, lane, start = 1, col, row
	}
	anchors := c.anchors[dir][lane]
	tiles := m.Tiles()
	dim := gen.boardDim
	end := start + len(tiles)
	if end > dim || anchors&((uint32(1)<<uint(end))-(uint32(1)<<uint(start))) == 0 {
		// (A lane without anchors is not loaded.)
		return false, 0, 0
	}
	squares := &gen.shadow.lanes[dir][lane]
	if start > 0 && squares[start-1].letter != 0 {
		return false, 0, 0
	}
	if end < dim && squares[end].letter != 0 {
		return false, 0, 0
	}
	through := &entry.through[i]
	// A one-tile play that also makes a word across its lane is the same
	// placement as a play of the other direction, and the generator makes
	// it in one form only: the horizontal one, unless a down lane's anchor
	// happens to take it in (see shadowPlayRight's isUnique). The play may
	// have been remembered from a board where the tile had no neighbour
	// across; here it would be a form the generator does not make, which
	// could win a tie against the form it does. So it is not taken.
	oneTileDown := vertical && m.TilesPlayed() == 1
	// The leave's index in the leave map: of each tile type, the low bits
	// of its slot, as many as are left.
	var used [carryMaxAlphabet]uint8
	for k, ml := range tiles {
		sq := &squares[start+k]
		if ml == 0 {
			if sq.letter == 0 || sq.letter != through[k] {
				return false, 0, 0
			}
			continue
		}
		if sq.letter != 0 || !sq.crossSet.Allowed(ml.Unblank()) {
			return false, 0, 0
		}
		if oneTileDown && sq.crossSet != board.TrivialCrossSet {
			return false, 0, 0
		}
		rackML := ml
		if ml.IsBlanked() {
			rackML = 0
		}
		used[rackML]++
	}
	idx := 0
	for ml, n := range rack.LetArr {
		if n == 0 {
			continue
		}
		left := n - int(used[ml])
		if left < 0 {
			return false, 0, 0
		}
		idx |= ((1 << uint(left)) - 1) << uint(gen.leavemap.LetterBaseIndex[ml])
	}
	score = scoreLanePlay(gen.letterDistribution, squares, tiles, start, m.TilesPlayed())
	return true, score, float64(score) + gen.leavemap.Values[idx]
}

// carrySeedAndSkip is called when every lane has been noted. It starts the
// generation from the best remembered play that is a play of this board
// too, if that beats what the generation has (an exchange, or nothing), and
// then marks every unchanged lane whose bound the generation's best play is
// no worse than.
func (gen *GordonGenerator) carrySeedAndSkip(rack *tilemapping.Rack) {
	c := gen.carry
	entry := c.cur
	dim := gen.boardDim
	seeded := false
	for i := 0; i < entry.numMoves; i++ {
		ok, score, eq := gen.carryRevalidate(rack, i)
		if !ok || (!gen.winner.IsEmpty() && eq < gen.winner.Equity()) {
			continue
		}
		c.scratch.CopyFrom(&entry.moves[i])
		c.scratch.SetScore(score)
		c.scratch.SetEquity(eq)
		if gen.winner.IsEmpty() || !carryNoWorse(gen.winner, &c.scratch) {
			gen.winner.CopyFrom(&c.scratch)
			seeded = true
		}
	}
	if seeded {
		if len(gen.plays) == 0 {
			gen.plays = append(gen.plays, gen.winner)
		} else {
			gen.plays[0] = gen.winner
		}
		c.seeded++
	}
	if gen.winner.IsEmpty() {
		return
	}
	if eq := gen.winner.Equity(); eq > gen.shadow.anchorFloor {
		gen.shadow.anchorFloor = eq
	}
	for dir := 0; dir < 2; dir++ {
		for row := 0; row < dim; row++ {
			f := c.found[dir][row]
			if f < 0 {
				continue
			}
			b, _ := entry.bound(dim, dir, row, int(f))
			if carryNoWorse(gen.winner, &entry.moves[b.moveIdx]) {
				c.skip[dir][row] = true
				c.lanesSkipped++
			}
		}
	}
}

// carryRemember is called when the generation is over, with its best move
// in gen.winner: that move bounds every lane the generation searched.
func (gen *GordonGenerator) carryRemember() {
	c := gen.carry
	entry := c.cur
	c.cur = nil
	if gen.quitEarly || gen.winner.IsEmpty() {
		return
	}
	// The move's slot: the one that already holds it, or the oldest.
	moveIdx := -1
	for i := 0; i < entry.numMoves; i++ {
		m := &entry.moves[i]
		if m.Equity() == gen.winner.Equity() && !m.TiebreaksBetter(gen.winner) &&
			!gen.winner.TiebreaksBetter(m) {
			moveIdx = i
			break
		}
	}
	if moveIdx < 0 {
		moveIdx = entry.nextMove
		entry.nextMove = (entry.nextMove + 1) % carryMovesPerRack
		if entry.numMoves < carryMovesPerRack {
			entry.numMoves++
		}
		entry.stamps[moveIdx]++
		entry.moves[moveIdx].CopyFrom(gen.winner)
	}
	dim := gen.boardDim
	if gen.winner.Action() == move.MoveTypePlay {
		// Its lane has an anchor, so it is loaded.
		row, col, vertical := gen.winner.CoordsAndVertical()
		dir, lane, start := 0, row, col
		if vertical {
			dir, lane, start = 1, col, row
		}
		squares := &gen.shadow.lanes[dir][lane]
		through := &entry.through[moveIdx]
		for k := range gen.winner.Tiles() {
			through[k] = squares[start+k].letter
		}
	}
	for dir := 0; dir < 2; dir++ {
		for row := 0; row < dim; row++ {
			if c.hashes[dir][row] == 0 || c.skip[dir][row] {
				continue
			}
			if f := c.found[dir][row]; f >= 0 {
				// The same contents are remembered with another bound;
				// keep the lower of the two.
				// (A bound whose move was just replaced is rewritten too.)
				b, _ := entry.bound(dim, dir, row, int(f))
				if b.stamp != entry.stamps[b.moveIdx] ||
					!carryNoWorse(gen.winner, &entry.moves[b.moveIdx]) {
					b.moveIdx = uint8(moveIdx)
					b.stamp = entry.stamps[moveIdx]
				}
				continue
			}
			slot := entry.nextBound[dir][row]
			entry.nextBound[dir][row] = (slot + 1) % carryBoundsPerLane
			b, kept := entry.bound(dim, dir, row, int(slot))
			b.hash = c.hashes[dir][row]
			b.epoch = entry.epoch
			b.stamp = entry.stamps[moveIdx]
			b.anchors = c.anchors[dir][row]
			b.moveIdx = uint8(moveIdx)
			copy(kept, gen.shadow.lanes[dir][row][:dim])
		}
	}
}

// genShadowCarried is genShadow for a generation that uses the memory: all
// lanes are loaded and looked up first, and then only the lanes that are not
// skipped are given bounds.
func (gen *GordonGenerator) genShadowCarried(rack *tilemapping.Rack) {
	dim := gen.boardDim

	for row := 0; row < dim; row++ {
		anchors, _ := gen.board.RowMasks(row, board.HorizontalDirection)
		if anchors != 0 {
			gen.cache.loadRow(gen.board, row, board.VerticalDirection, dim)
			gen.shadow.lanes[0][row] = gen.cache.squares
		}
		gen.carryNoteLane(0, row, anchors)
	}
	gen.board.Transpose()
	for row := 0; row < dim; row++ {
		anchors, _ := gen.board.RowMasks(row, board.VerticalDirection)
		if anchors != 0 {
			gen.cache.loadRow(gen.board, row, board.HorizontalDirection, dim)
			gen.shadow.lanes[1][row] = gen.cache.squares
		}
		gen.carryNoteLane(1, row, anchors)
	}

	gen.carrySeedAndSkip(rack)

	// The board is transposed: vertical lanes first.
	gen.vertical = true
	gen.shadowCarriedLanes(1, board.VerticalDirection)
	gen.board.Transpose()
	gen.vertical = false
	gen.shadowCarriedLanes(0, board.HorizontalDirection)

	gen.shadow.anchorHeap.heapifyAll()
}

func (gen *GordonGenerator) shadowCarriedLanes(lane int, dir board.BoardDirection) {
	c := gen.carry
	dim := gen.boardDim
	for row := 0; row < dim; row++ {
		anchors := c.anchors[lane][row]
		if anchors == 0 || c.skip[lane][row] {
			continue
		}
		gen.curRowIdx = row
		gen.lastAnchorCol = 100
		gen.cache.squares = gen.shadow.lanes[lane][row]
		for col := 0; col < dim; col++ {
			if anchors&(1<<uint(col)) != 0 {
				gen.shadowPlayForAnchor(row, col, gen.lastAnchorCol, dir)
				gen.lastAnchorCol = col
				if gen.cache.squares[col].letter != 0 {
					gen.lastAnchorCol++
				}
			}
		}
	}
}
