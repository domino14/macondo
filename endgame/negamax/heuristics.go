package negamax

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
	"os"
	"unsafe"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/tinymove/conversions"
)

// This file ports MAGPIE's endgame heuristics (src/impl/endgame.c). Both
// projects are GPL, so the implementation below is a direct translation.
//
// There are two surfaces, both gated on Solver.useHeuristics:
//
//   A. greedyPlayout replaces the static-spread leaf evaluation. From the leaf
//      position it keeps playing the best move for each side until the game
//      actually ends, then returns the resulting spread. This is far more
//      accurate than the current spread and lets the solver return good
//      answers at shallow ply depths.
//
//   B. The move-ordering estimate (see assignEstimates in solver.go) biases
//      which moves alpha-beta tries first.
//
// Both are further modulated by the opponent's "stuck fraction": the share of
// the opponent's rack score that cannot be played anywhere. When it is 0 --
// the common case -- surface A collapses to plain highest-score greedy and
// surface B collapses to a static estimate, so the expensive paths only run in
// genuinely stuck positions.

// playedTilesFaceValue sums the face values of the tiles a move places on the
// board, skipping play-through squares. Translated from MAGPIE's
// compute_played_tiles_face_value (endgame.c:1447).
//
// The tile layout is identical in both engines: tile group i lives at bits
// 20+6i, and bit 12+i marks that tile as a blank. MoveToTinyMove skips
// play-through tiles when encoding, so the first TilesPlayed() groups are
// exactly the newly-placed rack tiles. A blank is scored as the blank (0), not
// as the letter it was designated.
func playedTilesFaceValue(m *tinymove.SmallMove, ld *tilemapping.LetterDistribution) int {
	n := m.TilesPlayed()
	tm := uint64(m.TinyMove())
	faceValue := 0
	for i := 0; i < n; i++ {
		ml := tilemapping.MachineLetter((tm >> (20 + 6*i)) & 63)
		if tm&(1<<(12+i)) != 0 {
			ml = 0
		}
		faceValue += ld.Score(ml)
	}
	return faceValue
}

// conservationBonus penalizes playing tiles when the opponent has stuck tiles:
// the opponent is going to be left holding them at game end, so there is time
// to build, and dumping our own tiles only helps them. Translated from MAGPIE's
// compute_conservation_bonus (endgame.c:1464).
func conservationBonus(m *tinymove.SmallMove, ld *tilemapping.LetterDistribution, frac float32) int {
	n := m.TilesPlayed()
	faceValue := playedTilesFaceValue(m, ld)
	return int(float32(conservationTileWeight*n+conservationValueWeight*faceValue) * frac)
}

// orPlayedTiles sets, for every rack tile the move places, the bit for that
// tile's machine letter. Blanks set bit 0, since the blank is the rack tile
// that was actually used.
func orPlayedTiles(bv uint64, m *tinymove.SmallMove) uint64 {
	n := m.TilesPlayed()
	tm := uint64(m.TinyMove())
	for i := 0; i < n; i++ {
		ml := (tm >> (20 + 6*i)) & 63
		if tm&(1<<(12+i)) != 0 {
			ml = 0
		}
		bv |= 1 << ml
	}
	return bv
}

// stuckFractionFromBV returns the fraction of the rack's score that is stuck,
// where bit i of bv is set if machine letter i appears in some legal move.
// Translated from MAGPIE's stuck_tile_fraction_from_bv (endgame.c:93).
func stuckFractionFromBV(ld *tilemapping.LetterDistribution, rack *tilemapping.Rack, bv uint64) float32 {
	totalScore := 0
	stuckScore := 0
	for ml, count := range rack.LetArr {
		if count <= 0 {
			continue
		}
		score := count * ld.Score(tilemapping.MachineLetter(ml))
		totalScore += score
		if bv&(1<<uint(ml)) == 0 {
			stuckScore += score
		}
	}
	if totalScore == 0 {
		return 0
	}
	return float32(stuckScore) / float32(totalScore)
}

// bestSingleTilePlay finds the highest-scoring placement of a one-tile rack by
// scanning the board directly, instead of walking the word graph. Translated
// from MAGPIE's generate_single_tile_plays (endgame.c:1172).
//
// With one tile there is no word to build: legality is decided entirely by the
// two cross sets at the square, and the score needs no traversal because the
// perpendicular words' existing tiles are already summed in the cross scores.
// Returns false when the tile cannot be played anywhere.
//
// The caller must pass a rack holding exactly one tile; this does not check,
// and would silently score a placement of whichever tile it found first.
//
// Note the axis convention differs from MAGPIE's. Here GetCrossScore(r, c, D)
// is the sum of the existing tiles in the word running in direction D through
// (r, c) -- the argument names the cross word's own direction, which is how
// board.ScoreWord and TinyMoveToFullMove use it, and which board/cross_set.go
// documents for the matching cross *sets*. MAGPIE's cross_score is indexed by
// the perpendicular direction instead, so a literal transliteration scores
// wrong -- and wrong nearly everywhere, not just where words cross, because a
// square with no word in direction D stores crossScore(D) = 0, so the swap
// drops the whole word the tile was attaching to.
//
// Only the legality test is direction-blind: it intersects both cross sets,
// since one tile forms a word in both directions at once.
//
// Two things this deliberately does not do. It reports no play on an empty
// board, where an opening play would in fact be legal -- unreachable here,
// since an endgame always has tiles down. And when the tile is a blank it takes
// the lowest allowed designation, which can differ from the one full move
// generation would have picked; the score is identical either way (a blank is
// worth nothing) but the letter left on the board differs, so the rest of the
// rollout can diverge. Both are the same class of freedom as the tie-break
// between equal-scoring placements.
func bestSingleTilePlay(b *board.GameBoard, rack *tilemapping.Rack,
	ld *tilemapping.LetterDistribution) (tinymove.SmallMove, bool) {

	tile := loneTile(rack)
	score, row, col, found := scanLoneTile(b, tile, ld)
	if !found {
		return tinymove.SmallMove{}, false
	}
	return loneTileMove(b, tile, ld, row, col, score), true
}

// keptTile returns the one tile of the rack that the play m, which places
// all the others, leaves behind.
func keptTile(rack *tilemapping.Rack, m *tinymove.SmallMove) tilemapping.MachineLetter {
	sum := 0
	for ml, n := range rack.LetArr {
		sum += ml * int(n)
	}
	tm := uint64(m.TinyMove())
	for i, n := 0, m.TilesPlayed(); i < n; i++ {
		// A blank is tile 0 on the rack, whatever letter it plays as.
		if tm&(1<<(12+uint(i))) == 0 {
			sum -= int((tm >> (20 + 6*uint(i))) & 63)
		}
	}
	return tilemapping.MachineLetter(sum)
}

// loneTile returns the first tile type on the rack: the tile, when the rack
// holds just one.
func loneTile(rack *tilemapping.Rack) tilemapping.MachineLetter {
	for ml, count := range rack.LetArr {
		if count > 0 {
			return tilemapping.MachineLetter(ml)
		}
	}
	return 0
}

// loneTileFace is what the tile scores on a plain square.
func loneTileFace(tile tilemapping.MachineLetter, ld *tilemapping.LetterDistribution) int {
	if tile == 0 {
		return 0
	}
	return ld.Score(tile)
}

// loneTileAt scores the tile played alone on the empty square (r, c), and
// reports whether that is a legal play. It depends only on the square's two
// cross-sets and cross-scores and on which of its four neighbours hold
// tiles.
func loneTileAt(b *board.GameBoard, tile tilemapping.MachineLetter, tileFace int,
	r, c int) (int, bool) {

	dim := b.Dim()
	hasH := (c > 0 && b.HasLetter(r, c-1)) || (c < dim-1 && b.HasLetter(r, c+1))
	hasV := (r > 0 && b.HasLetter(r-1, c)) || (r < dim-1 && b.HasLetter(r+1, c))
	if !hasH && !hasV {
		return 0, false
	}
	idx := b.GetSqIdx(r, c)
	combined := uint64(b.GetCrossSetIdx(idx, board.HorizontalDirection)) &
		uint64(b.GetCrossSetIdx(idx, board.VerticalDirection))
	if tile == 0 {
		// A blank plays if the intersection admits any real letter. Bit 0
		// is the blank marker, not a letter, so shift it out.
		if combined>>1 == 0 {
			return 0, false
		}
	} else if combined&(1<<uint(tile)) == 0 {
		return 0, false
	}
	tileLS := tileFace * b.GetLetterMultiplier(idx)
	wm := b.GetWordMultiplier(idx)
	score := 0
	if hasH {
		score += (tileLS + b.GetCrossScoreIdx(idx, board.HorizontalDirection)) * wm
	}
	if hasV {
		score += (tileLS + b.GetCrossScoreIdx(idx, board.VerticalDirection)) * wm
	}
	return score, true
}

// nonzeroBytes returns a bit for each of x's eight bytes that is not zero,
// the lowest byte's in bit 0.
func nonzeroBytes(x uint64) uint32 {
	const low7 = 0x7f7f7f7f7f7f7f7f
	// The top bit of each byte: set if the byte's low seven bits are not
	// all zero (the add carries into it) or the byte has it set itself.
	t := ((x&low7 + low7) | x) &^ low7
	// Gather the eight top bits into the top byte.
	return uint32((t >> 7) * 0x0102040810204080 >> 56)
}

// scanLoneTile finds the highest-scoring square for the tile played alone:
// the first such square in row order.
func scanLoneTile(b *board.GameBoard, tile tilemapping.MachineLetter,
	ld *tilemapping.LetterDistribution) (bestScore, bestRow, bestCol int, found bool) {

	isBlank := tile == 0
	dim := b.Dim()
	squares := b.SquaresSlice()
	hCrossSets := b.CrossSetsForDir(board.HorizontalDirection)
	vCrossSets := b.CrossSetsForDir(board.VerticalDirection)
	rowMul := b.GetSqIdx(1, 0)
	colMul := b.GetSqIdx(0, 1)

	var occupied [board.MaxBoardDim]uint32
	if dim == 15 && colMul == 1 && len(squares) >= 14*rowMul+15 {
		// A row's 15 squares are 15 adjacent bytes: read them eight at a
		// time (the two reads share the eighth square).
		bytes := unsafe.Slice((*byte)(unsafe.Pointer(&squares[0])), len(squares))
		for r := 0; r < dim; r++ {
			base := r * rowMul
			lo := nonzeroBytes(binary.LittleEndian.Uint64(bytes[base:]))
			hi := nonzeroBytes(binary.LittleEndian.Uint64(bytes[base+7:]))
			occupied[r] = lo | hi<<7
		}
	} else {
		for r := 0; r < dim; r++ {
			var mask uint32
			base := r * rowMul
			for c := 0; c < dim; c++ {
				if squares[base+c*colMul] != 0 {
					mask |= 1 << uint(c)
				}
			}
			occupied[r] = mask
		}
	}
	tileFace := loneTileFace(tile, ld)

	full := uint32(1)<<uint(dim) - 1
	for r := 0; r < dim; r++ {
		base := r * rowMul
		// The empty squares next to a tile, across (h) and down (v), found by
		// shifting the occupancy masks instead of probing every square.
		hNbr := (occupied[r]<<1 | occupied[r]>>1) & full
		var vNbr uint32
		if r > 0 {
			vNbr |= occupied[r-1]
		}
		if r < dim-1 {
			vNbr |= occupied[r+1]
		}
		candidates := (hNbr | vNbr) &^ occupied[r]
		for candidates != 0 {
			c := bits.TrailingZeros32(candidates)
			candidates &= candidates - 1
			idx := base + c*colMul
			combined := uint64(hCrossSets[idx]) & uint64(vCrossSets[idx])
			if isBlank {
				if combined>>1 == 0 {
					continue
				}
			} else if combined&(1<<uint(tile)) == 0 {
				continue
			}
			tileLS := tileFace * b.GetLetterMultiplier(idx)
			wm := b.GetWordMultiplier(idx)
			score := 0
			if hNbr&(1<<uint(c)) != 0 {
				score += (tileLS + b.GetCrossScoreIdx(idx, board.HorizontalDirection)) * wm
			}
			if vNbr&(1<<uint(c)) != 0 {
				score += (tileLS + b.GetCrossScoreIdx(idx, board.VerticalDirection)) * wm
			}
			if found && score <= bestScore {
				continue
			}
			found = true
			bestScore, bestRow, bestCol = score, r, c
		}
	}
	return bestScore, bestRow, bestCol, found
}

// loneTileMove builds the play of the tile alone on (r, c), which scores
// score: the word's extent and, for a blank, its letter.
func loneTileMove(b *board.GameBoard, tile tilemapping.MachineLetter,
	ld *tilemapping.LetterDistribution, r, c, score int) tinymove.SmallMove {

	dim := b.Dim()
	hasH := (c > 0 && b.HasLetter(r, c-1)) || (c < dim-1 && b.HasLetter(r, c+1))
	vertical := !hasH
	start, end := c, c
	if !vertical {
		for start > 0 && b.HasLetter(r, start-1) {
			start--
		}
		for end < dim-1 && b.HasLetter(r, end+1) {
			end++
		}
	} else {
		start, end = r, r
		for start > 0 && b.HasLetter(start-1, c) {
			start--
		}
		for end < dim-1 && b.HasLetter(end+1, c) {
			end++
		}
	}
	isBlank := tile == 0
	chosen := tile
	if isBlank {
		// A blank scores 0 whatever it is designated, so every allowed
		// letter ties; take the lowest.
		idx := b.GetSqIdx(r, c)
		combined := uint64(b.GetCrossSetIdx(idx, board.HorizontalDirection)) &
			uint64(b.GetCrossSetIdx(idx, board.VerticalDirection))
		numLetters := int(ld.TileMapping().NumLetters())
		for ml := tilemapping.MachineLetter(1); int(ml) < numLetters; ml++ {
			if combined&(1<<uint(ml)) != 0 {
				chosen = ml
				break
			}
		}
	}

	// Encode exactly as AllPlaysSmallRecorder does: true board coordinates,
	// with the word's start along the play direction.
	var moveCode uint64
	moveCode |= uint64(chosen) << 20
	if isBlank {
		moveCode |= 1 << 12
	}
	row, col := r, start
	if vertical {
		moveCode |= 1
		row, col = start, c
	}
	moveCode |= uint64(col) << 1
	moveCode |= uint64(row) << 6

	return tinymove.TilePlayMove(tinymove.TinyMove(moveCode), int16(score), 1,
		uint8(end-start+1))
}

// boardPlayableTilesBV scans the board once and returns a bitvector of every
// machine letter that has a legal single-tile play somewhere. Translated from
// MAGPIE's board_get_playable_tiles_bv (board.h:899); a square is a candidate
// if it is empty and orthogonally adjacent to a tile, and a letter is playable
// there if both cross sets allow it.
//
// Rather than probe four neighbors for all dim^2 squares, we build a per-row
// occupancy bitmask and derive the candidate squares by shifting, which
// enumerates only the ~20-60 squares that actually border the played area.
//
// targetBV, when nonzero, is an early-exit set: the scan stops as soon as every
// bit in it is covered. Bit 0 (the blank) must not be included in it; blank
// playability is decided by the caller.
func boardPlayableTilesBV(b *board.GameBoard, targetBV uint64) uint64 {
	if b.TilesPlayed() == 0 {
		// No tiles on the board at all, so nothing constrains an opening play.
		// The endgame solver never sees this, but a 0 here would mean "every
		// tile is stuck", which is a much worse thing to be wrong about.
		return uint64(board.TrivialCrossSet)
	}
	dim := b.Dim()
	squares := b.SquaresSlice()
	hCrossSets := b.CrossSetsForDir(board.HorizontalDirection)
	vCrossSets := b.CrossSetsForDir(board.VerticalDirection)
	// Derive the flat-index strides rather than assuming row*dim+col: the board
	// swaps them under Transpose, and the cross-set arrays are not transposed
	// with it.
	rowMul := b.GetSqIdx(1, 0)
	colMul := b.GetSqIdx(0, 1)

	var occupied [board.MaxBoardDim]uint32
	if dim == 15 && colMul == 1 && len(squares) >= 14*rowMul+15 {
		// A row's 15 squares are 15 adjacent bytes: read them eight at a
		// time (the two reads share the eighth square).
		bytes := unsafe.Slice((*byte)(unsafe.Pointer(&squares[0])), len(squares))
		for r := 0; r < dim; r++ {
			base := r * rowMul
			lo := nonzeroBytes(binary.LittleEndian.Uint64(bytes[base:]))
			hi := nonzeroBytes(binary.LittleEndian.Uint64(bytes[base+7:]))
			occupied[r] = lo | hi<<7
		}
	} else {
		for r := 0; r < dim; r++ {
			var mask uint32
			base := r * rowMul
			for c := 0; c < dim; c++ {
				if squares[base+c*colMul] != 0 {
					mask |= 1 << uint(c)
				}
			}
			occupied[r] = mask
		}
	}

	full := uint32(1)<<uint(dim) - 1
	var playable uint64
	for r := 0; r < dim; r++ {
		candidates := occupied[r]<<1 | occupied[r]>>1
		if r > 0 {
			candidates |= occupied[r-1]
		}
		if r < dim-1 {
			candidates |= occupied[r+1]
		}
		candidates &^= occupied[r]
		candidates &= full
		base := r * rowMul
		for candidates != 0 {
			c := bits.TrailingZeros32(candidates)
			candidates &= candidates - 1
			idx := base + c*colMul
			playable |= uint64(hCrossSets[idx]) & uint64(vCrossSets[idx])
			if targetBV != 0 && playable&targetBV == targetBV {
				return playable
			}
		}
	}
	return playable
}

// oppStuckFraction returns the share of the opponent's rack score that cannot
// be played anywhere in this position, in [0, 1]. Translated from MAGPIE's
// compute_opp_stuck_fraction (endgame.c:533).
//
// The result depends only on the position, never on the path taken to reach
// it, which is what lets the leaf evaluation stay consistent across threads and
// transpositions. nodeKey is the position's Zobrist hash (board, both racks,
// turn, scoreless turns); it subsumes everything the fraction depends on, so we
// memoize on it. A nodeKey of 0 means the transposition table is off, in which
// case the memo is bypassed.
//
// Note this generates moves for the opponent, which clobbers mg's play list.
// Callers must have copied out any plays they still need.
func (s *Solver) oppStuckFraction(g *game.Game, mg movegen.MoveGenerator, nodeKey uint64, thread int) float32 {
	oppRack := g.RackFor(1 - s.solvingPlayer)
	if oppRack.NumTiles() == 0 {
		return 0
	}
	var slot *stuckEntry
	if nodeKey != 0 && thread < len(s.stuckMemo) {
		slot = &s.stuckMemo[thread][nodeKey&(stuckMemoSize-1)]
		if slot.key == nodeKey && slot.gen == s.stuckGen {
			return slot.frac
		}
	}

	var rackBV uint64
	for ml, count := range oppRack.LetArr {
		if count > 0 {
			rackBV |= 1 << uint(ml)
		}
	}
	if oppRack.NumTiles() == 1 {
		if lists := s.listsFor(thread, g, mg); lists != nil {
			// One tile: it is stuck exactly when it has no play alone, which
			// the path lists keep track of from position to position.
			var frac float32
			if _, ok := lists.bestLoneTilePlay(1 - s.solvingPlayer); !ok {
				frac = stuckFractionFromBV(g.Bag().LetterDistribution(), oppRack, 0)
			}
			if slot != nil {
				slot.key = nodeKey
				slot.frac = frac
				slot.gen = s.stuckGen
			}
			return frac
		}
	}
	// Cross-set scan: which rack tiles have a legal single-tile play? If they
	// all do, nobody is stuck and we can skip move generation entirely.
	playable := boardPlayableTilesBV(g.Board(), rackBV&^1)
	tilesBV := playable & rackBV
	// A blank is playable if any non-blank letter is playable.
	if rackBV&1 != 0 && playable>>1 != 0 {
		tilesBV |= 1
	}

	var frac float32
	switch {
	case tilesBV == rackBV:
		// Every rack tile plays somewhere; nothing is stuck.
	case oppRack.NumTiles() == 1:
		// With a single tile there are no longer plays to find, so the scan is
		// authoritative.
		frac = stuckFractionFromBV(g.Bag().LetterDistribution(), oppRack, tilesBV)
	default:
		// Some tiles look stuck, but a tile with no single-tile play may still
		// play inside a longer word. Fall back to real move generation. This is
		// the expensive branch; most positions return above.
		var plays []tinymove.SmallMove
		if lists := s.listsFor(thread, g, mg); lists != nil {
			plays = lists.get(1 - s.solvingPlayer).moves
		} else {
			mg.GenAll(oppRack, false)
			plays = mg.SmallPlays()
		}
		for i := range plays {
			tilesBV = orPlayedTiles(tilesBV, &plays[i])
			if tilesBV&rackBV == rackBV {
				break
			}
		}
		frac = stuckFractionFromBV(g.Bag().LetterDistribution(), oppRack, tilesBV)
	}

	if slot != nil {
		slot.key = nodeKey
		slot.frac = frac
		slot.gen = s.stuckGen
	}
	return frac
}

// playoutTail is the rest of a playout from a position one passed through:
// what it adds to the solving player's spread there, and how many moves it
// is. A key of 0 is an empty entry.
type playoutTail struct {
	key    uint64
	value  int32
	length int32
}

const (
	// playoutTailCacheSize is the number of entries of each thread's cache
	// of playout tails: half a megabyte, to stay in the thread's own cache.
	playoutTailCacheSize = 1 << 15
	// playoutTailSalt keeps a tail's key apart from its position's in the
	// transposition table, and playoutTailGenStep is what each solve adds.
	playoutTailSalt    = 0xC2B2AE3D27D4EB4F
	playoutTailGenStep = 0xD1B54A32D192ED03
)

// playoutTailsEverywhere caches the tails of playouts with no stuck tiles
// too.
var playoutTailsEverywhere = os.Getenv("ENDGAME_TAILS_EVERYWHERE") != ""

// greedyPlayout evaluates a leaf node by playing the game out greedily instead
// of returning the static spread. Translated from MAGPIE's
// negamax_greedy_leaf_playout (endgame.c:1740).
//
// The returned value is the spread from the perspective of the player on turn
// at the leaf, matching what the static evaluation returned, so negamax's
// negation propagates it up unchanged.
//
// The moves played out are recorded in pv, so that a principal variation that
// runs off the end of the search still shows how the game is expected to
// finish. They are marked as estimates when displayed.
//
// A playout with stuck tiles is long, the solving player laying its tiles
// down a few at a time, and it often passes through a position an earlier
// one reached by the same moves in another order. From there it plays the
// same moves: the rest of a playout depends only on the position and on the
// stuck fraction found at the leaf. So each position a playout passes
// through is remembered with what the rest added to the spread and how long
// it was, and a later playout that reaches it stops there. The idea is
// MAGPIE's (its playout tail cache). The thread's own small cache holds
// them, keyed by the position's hash with the fraction's bits. A playout cut
// short by its budget stores nothing, and a stored rest is used only where
// there is room for all of it, so the value is the one the playout would
// have reached. A principal-variation node plays its playout to the end, for
// its line.
func (s *Solver) greedyPlayout(g *game.Game, mg movegen.MoveGenerator, pv *PVLine,
	nodeKey uint64, thread int, pvNode bool) int16 {
	// Capture this before playing anything: handleConsecutiveScorelessTurns
	// advances the player on turn when the game ends on a double pass, and does
	// not put it back.
	leafPlayer := g.PlayerOnTurn()
	// The playout pushes states past the depth the search sized the stack for,
	// so never play more moves than the stack can hold. Solve leaves
	// MaxGreedyPlayoutPlies of headroom; callers of QuickAndDirtySolve size the
	// stack themselves and may leave less.
	budget := min(MaxGreedyPlayoutPlies, g.StateStackRemaining())
	if budget <= 0 {
		pv.Clear()
		return int16(g.SpreadFor(leafPlayer))
	}
	var playoutMoves [MaxGreedyPlayoutPlies]tinymove.SmallMove
	ld := g.Bag().LetterDistribution()
	// Recomputed from this position rather than inherited from the parent, so
	// that the leaf value is a pure function of the position.
	frac := s.oppStuckFraction(g, mg, nodeKey, thread)

	// Each player's rack at face value, kept up as the playout's moves take
	// tiles off them.
	rackScores := [2]int{g.RackFor(0).ScoreOn(ld), g.RackFor(1).ScoreOn(ld)}

	// played counts the playout's moves and onBoard those actually made on
	// the game; they differ by the going-out play, which is only scored.
	played, onBoard := 0, 0
	wentOut := false
	outGain := 0

	// The keys of the positions the playout passes through, for the cache
	// of tails, and the solving player's spread at each.
	cacheTails := s.playoutTailOptim && nodeKey != 0 && thread < len(s.playoutTails) &&
		(frac > 0 || playoutTailsEverywhere)
	var tails []playoutTail
	var tailKeys [MaxGreedyPlayoutPlies + 1]uint64
	var tailSpreads [MaxGreedyPlayoutPlies + 1]int32
	var tailSalt uint64
	key := nodeKey
	// A rest found in the cache: the playout stops there, except in check
	// mode, which plays on to see that it gets the same.
	tailHit, tailChecked := false, false
	var hit playoutTail
	hitAt := 0
	if cacheTails {
		tails = s.playoutTails[thread]
		tailSalt = playoutTailSalt ^ uint64(math.Float32bits(frac))<<32 ^
			uint64(s.stuckGen)*playoutTailGenStep
		tailKeys[0] = key ^ tailSalt
		tailSpreads[0] = int32(g.SpreadFor(s.solvingPlayer))
		if e := &tails[tailKeys[0]&(playoutTailCacheSize-1)]; !pvNode &&
			e.key == tailKeys[0] && int(e.length) <= budget {
			hit, hitAt = *e, 0
			if checkMoveLists {
				tailChecked = true
			} else {
				tailHit = true
			}
		}
	}
	if lists := s.listsFor(thread, g, mg); lists != nil {
		lists.inPlayout = true
		defer func() { lists.inPlayout = false }()
	}
	for !tailHit && played < budget && g.Playing() == pb.PlayState_PLAYING {
		onTurn := g.PlayerOnTurn()
		rack := g.RackFor(onTurn)
		var plays []tinymove.SmallMove
		if rack.NumTiles() == 1 {
			// One tile left: a board scan beats walking the word graph, and it
			// stays correct under conservation because every placement of the
			// same lone tile carries the identical conservation bonus (one tile,
			// one face value), so the highest-scoring placement is still the
			// one the picker below would choose.
			buf := &s.singleTileBuf[thread]
			n := 0
			var m tinymove.SmallMove
			var ok bool
			if lists := s.listsFor(thread, g, mg); lists != nil {
				m, ok = lists.bestLoneTilePlay(onTurn)
			} else {
				m, ok = bestSingleTilePlay(g.Board(), rack, ld)
			}
			if ok {
				buf[n] = m
				n++
			}
			buf[n] = tinymove.PassMove()
			n++
			plays = buf[:n]
		} else if lists := s.listsFor(thread, g, mg); lists != nil {
			// The list has no pass; give it the one GenAll would add at the
			// end. The list is the path lists' own, so the pass goes in a
			// copy only when there is no room behind the plays.
			l := lists.get(onTurn)
			l.moves = append(l.moves, tinymove.PassMove())
			plays = l.moves
			l.moves = l.moves[:len(l.moves)-1]
		} else {
			mg.GenAll(rack, false)
			plays = mg.SmallPlays()
		}
		if len(plays) == 0 {
			break
		}

		// When the opponent has stuck tiles and we are on turn, prefer
		// conservation: hold on to tiles rather than dumping them.
		conserve := frac > 0 && onTurn == s.solvingPlayer
		passPenalty := 0
		if conserve {
			// Passing instead of going out steers the game toward a
			// double-pass ending, where both players lose their rack value,
			// instead of gaining twice the opponent's rack. The difference is
			// (own rack + opp rack), prorated by the stuck fraction.
			//
			// MAGPIE gates this on an unrelated solver flag and so ships with
			// the penalty always zero; without it a pass (adjusted score 0)
			// beats every real play once the conservation bonus makes their
			// adjusted scores negative, and the playout degenerates into
			// double-passing.
			passPenalty = int(float32(rackScores[0]+rackScores[1]) * frac)
		}

		bestIdx := 0
		bestAdj := math.MinInt32
		for idx := range plays {
			adj := plays[idx].Score()
			if conserve {
				if plays[idx].IsPass() {
					adj -= passPenalty
				} else {
					adj -= conservationBonus(&plays[idx], ld, frac)
				}
			}
			if adj > bestAdj {
				bestAdj = adj
				bestIdx = idx
			}
		}

		// Play straight out of the generator's list rather than through a local
		// copy: taking the address of a local here would force it onto the heap
		// once per playout move, which at leaf-node frequency was the entire
		// allocation cost of the heuristics. plays stays valid until the next
		// GenAll, and we record the move by value before then.
		playoutMoves[played] = plays[bestIdx]
		if !plays[bestIdx].IsPass() && plays[bestIdx].TilesPlayed() == int(rack.NumTiles()) {
			// The play empties the rack and ends the game: its score plus
			// twice the other rack is all it changes, so there is no need to
			// put it on the board (and take it off again).
			outGain = plays[bestIdx].Score() + 2*rackScores[1-onTurn]
			if onTurn != leafPlayer {
				outGain = -outGain
			}
			wentOut = true
			played++
			break
		}
		if tiles := int(rack.NumTiles()); tiles >= 2 && plays[bestIdx].TilesPlayed() == tiles-1 {
			// The play keeps one tile, whose best play the playout will
			// want next. This position's list has all of that tile's
			// plays, so note the best now and spare a scan of the board.
			if lists := s.listsFor(thread, g, mg); lists != nil {
				lists.noteLonePlay(onTurn, keptTile(rack, &plays[bestIdx]))
			}
		}
		if !plays[bestIdx].IsPass() {
			rackScores[onTurn] -= playedTilesFaceValue(&plays[bestIdx], ld)
		}
		moveTiles, err := g.PlaySmallMove(&plays[bestIdx])
		played++
		onBoard++
		if err != nil {
			cacheTails = false
			break
		}
		// A position where the game is over has no rest to keep.
		if cacheTails && g.Playing() == pb.PlayState_PLAYING {
			key = s.ttable.Zobrist().AddMove(key, &playoutMoves[played-1], rack, moveTiles,
				onTurn == s.solvingPlayer, g.ScorelessTurns(), g.LastScorelessTurns())
			tailKeys[played] = key ^ tailSalt
			tailSpreads[played] = int32(g.SpreadFor(s.solvingPlayer))
			if e := &tails[tailKeys[played]&(playoutTailCacheSize-1)]; !pvNode && !tailChecked &&
				e.key == tailKeys[played] && int(e.length) <= budget-played {
				hit, hitAt = *e, played
				if checkMoveLists {
					tailChecked = true
				} else {
					tailHit = true
				}
			}
		}
	}
	pv.setPlayout(playoutMoves[:played])

	spread := g.SpreadFor(leafPlayer) + outGain
	// Whether the playout ran to the end of the game, or to a stored rest
	// that did.
	ended := tailHit || wentOut || g.Playing() != pb.PlayState_PLAYING
	if tailHit {
		spread = int(tailSpreads[hitAt] + hit.value)
		if leafPlayer != s.solvingPlayer {
			spread = -spread
		}
	} else if !wentOut && g.Playing() == pb.PlayState_PLAYING {
		// We ran out of budget before the game ended, which is rare. Estimate
		// who was closer to going out with MAGPIE's rack adjustment
		// (endgame.c:1932): twice the rack score off the holder, applied to
		// both sides. This is not real end-of-game scoring, it is an estimate.
		spread -= 2 * g.RackFor(leafPlayer).ScoreOn(ld)
		spread += 2 * g.RackFor(1-leafPlayer).ScoreOn(ld)
	}

	if cacheTails && ended {
		final := int32(spread)
		if leafPlayer != s.solvingPlayer {
			final = -final
		}
		total := played
		if tailHit {
			total += int(hit.length)
		}
		if tailChecked && (final != tailSpreads[hitAt]+hit.value ||
			total-hitAt != int(hit.length)) {
			panic(fmt.Sprintf("playout tail cache: the rest stored at step %d is %d in %d moves, played out it is %d in %d",
				hitAt, hit.value, hit.length, final-tailSpreads[hitAt], total-hitAt))
		}
		// The position a rest was found at holds it already, and the one
		// the game ended at was never keyed.
		for i := 0; i < played; i++ {
			e := &tails[tailKeys[i]&(playoutTailCacheSize-1)]
			e.key = tailKeys[i]
			e.value = final - tailSpreads[i]
			e.length = int32(total - i)
		}
	}

	for i := 0; i < onBoard; i++ {
		g.UnplayLastMove()
	}
	if played > 0 {
		s.nodes.addPlayout(thread, uint64(played))
	}
	return int16(spread)
}

// computeBuildChainValues rewards moves that build toward longer scoring
// sequences: when the opponent is stuck there is time to extend a word over
// several turns (ED -> RED -> IRED -> AIRED -> WAIRED), so a short play that is
// contained in a longer one is worth its own score plus the best containing
// play's value. Translated from MAGPIE's compute_build_chain_values
// (endgame.c:1501).
//
// Returns nil -- meaning "no build chain in play" -- unless the opponent is
// actually stuck, so the O(n^2) scan stays off the common path. The returned
// slice is per-thread scratch and is only valid until the next call on the same
// thread.
func (s *Solver) computeBuildChainValues(moves []tinymove.SmallMove, bd *board.GameBoard,
	frac float32, thread int) []int {

	if frac <= 0 || len(moves) <= 1 {
		return nil
	}
	n := len(moves)
	if cap(s.bcValues[thread]) < n {
		s.bcValues[thread] = make([]int, n)
		s.bcOrder[thread] = make([]int, n)
	}
	buildValues := s.bcValues[thread][:n]
	order := s.bcOrder[thread][:n]
	for i := range order {
		order[i] = i
	}
	// Insertion sort indices by tiles played, descending, so that when we get
	// to a move every longer move has already been finalized.
	for i := 1; i < n; i++ {
		key := order[i]
		tpKey := moves[key].TilesPlayed()
		j := i - 1
		for j >= 0 && moves[order[j]].TilesPlayed() < tpKey {
			order[j+1] = order[j]
			j--
		}
		order[j+1] = key
	}

	moveA := &s.bcMoveA[thread]
	moveB := &s.bcMoveB[thread]

	for oi := 0; oi < n; oi++ {
		i := order[oi]
		smA := &moves[i]
		buildValues[i] = smA.Score()
		if smA.IsPass() {
			continue
		}
		rowA, colA, vertA := smA.CoordsAndVertical()
		lenA := smA.PlayLength()
		tpA := smA.TilesPlayed()

		moveAReady := false
		bestExtension := 0
		for oj := 0; oj < oi; oj++ {
			j := order[oj]
			smB := &moves[j]
			if smB.IsPass() || smB.TilesPlayed() <= tpA {
				continue
			}
			rowB, colB, vertB := smB.CoordsAndVertical()
			if vertB != vertA {
				continue
			}
			lenB := smB.PlayLength()

			// A's span must nest inside B's span, in the same lane.
			var contained bool
			var offsetInB int
			if vertA {
				contained = colA == colB && rowA >= rowB && rowA+lenA <= rowB+lenB
				offsetInB = rowA - rowB
			} else {
				contained = rowA == rowB && colA >= colB && colA+lenA <= colB+lenB
				offsetInB = colA - colB
			}
			if !contained {
				continue
			}

			// Nesting spans isn't enough; the tiles have to actually agree.
			// Expand A lazily, on the first containment hit.
			if !moveAReady {
				conversions.TinyMoveToMove(smA.TinyMove(), bd, moveA)
				moveAReady = true
			}
			conversions.TinyMoveToMove(smB.TinyMove(), bd, moveB)
			tilesA := moveA.Tiles()
			tilesB := moveB.Tiles()
			tilesMatch := true
			for ti := 0; ti < lenA && ti < len(tilesA); ti++ {
				if tilesA[ti] == 0 {
					// Play-through square; the board supplies this tile, not A.
					continue
				}
				bi := ti + offsetInB
				if bi >= len(tilesB) || tilesA[ti] != tilesB[bi] {
					tilesMatch = false
					break
				}
			}
			if !tilesMatch {
				continue
			}
			if buildValues[j] > bestExtension {
				bestExtension = buildValues[j]
			}
		}
		if bestExtension > 0 {
			buildValues[i] += bestExtension
		}
	}
	return buildValues
}
