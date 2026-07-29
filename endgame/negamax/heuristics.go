package negamax

import (
	"math"
	"math/bits"

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
		mg.GenAll(oppRack, false)
		plays := mg.SmallPlays()
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
func (s *Solver) greedyPlayout(g *game.Game, mg movegen.MoveGenerator, pv *PVLine,
	nodeKey uint64, thread int) int16 {
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

	played := 0
	for played < budget && g.Playing() == pb.PlayState_PLAYING {
		onTurn := g.PlayerOnTurn()
		mg.GenAll(g.RackFor(onTurn), false)
		plays := mg.SmallPlays()
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
			passPenalty = int(float32(g.RackFor(onTurn).ScoreOn(ld)+
				g.RackFor(1-onTurn).ScoreOn(ld)) * frac)
		}

		best := plays[0]
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
				// Copy by value; the next GenAll overwrites the play list.
				best = plays[idx]
			}
		}

		_, err := g.PlaySmallMove(&best)
		playoutMoves[played] = best
		played++
		if err != nil {
			break
		}
	}
	pv.setPlayout(playoutMoves[:played])

	spread := g.SpreadFor(leafPlayer)
	if g.Playing() == pb.PlayState_PLAYING {
		// We ran out of budget before the game ended, which is rare. Estimate
		// who was closer to going out with MAGPIE's rack adjustment
		// (endgame.c:1932): twice the rack score off the holder, applied to
		// both sides. This is not real end-of-game scoring, it is an estimate.
		spread -= 2 * g.RackFor(leafPlayer).ScoreOn(ld)
		spread += 2 * g.RackFor(1-leafPlayer).ScoreOn(ld)
	}

	for i := 0; i < played; i++ {
		g.UnplayLastMove()
	}
	if played > 0 {
		s.nodes.Add(uint64(played))
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
