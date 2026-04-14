// wordmap_gen.go: WMP-based word retrieval for shadow-best-first move
// generation. Mirrors MAGPIE's wordmap_gen + record_wmp_plays_for_word +
// record_wmp_play.
//
// Given a WMP-recorded anchor (one with TilesToPlay, PlaythroughBlocks,
// WordLength, LeftmostStartCol, RightmostStartCol populated), wordmapGen
// asks the WMP for every blankless word that fits the anchor's
// (rack subset, word length) tuple, then for each word and start column
// it:
//
//  1. Checks the play against the board (cross-sets and playthrough
//     match), filling gen.strip with playthrough markers and
//     unblanked-letter placements.
//  2. Recursively decides which non-playthrough positions are designated
//     blanks vs real tiles, given how many blanks the subrack has.
//  3. Scores each fully-specified play and emits it via gen.playRecorder.
//
// This replaces recursive_gen for WMP anchors. The advantage over
// recursive_gen is that the WMP gives us the entire word list directly
// — no GADDAG traversal — and the per-(blocks, tiles) anchor structure
// avoids the redundant work that the previous integration suffered
// from.
package movegen

import (
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/wmp"
)

// wordmapGen runs MAGPIE's wordmap_gen for the given (already-loaded)
// WMP anchor and emits plays via gen.playRecorder. The anchor's row
// must be loaded into gen.cache before calling.
func (gen *GordonGenerator) wordmapGen(rack *tilemapping.Rack, anchor *Anchor) {
	wgen := &gen.wmpMoveGen

	// Decode the Anchor's packed uint8 fields into ints once so the
	// hot loop below doesn't pay the conversion repeatedly.
	tilesToPlay := int(anchor.TilesToPlay)
	wordLength := int(anchor.WordLength)
	leftmostStart := int(anchor.LeftmostStartCol)
	rightmostStart := int(anchor.RightmostStartCol)

	// Step 1: rebuild the WMP playthrough bit rack from the row cache
	// for this anchor. Mirrors set_playthrough_bit_rack.
	gen.setWMPPlaythroughFromRow(anchor)

	// Step 2: prepare the playthrough subrack array for this anchor.
	// PlaythroughSubracksInit reads only TilesToPlay and WordLength
	// from the anchor, so we construct a small wmp.Anchor with just
	// those two fields rather than translating the whole macondo
	// Anchor into a wmp.Anchor.
	wgen.PlaythroughSubracksInit(&wmp.Anchor{
		TilesToPlay: tilesToPlay,
		WordLength:  wordLength,
	})

	// All plays emitted by this anchor have exactly anchor.TilesToPlay
	// new tiles. Set tilesPlayed once so the bingo bonus and the
	// recorder's per-play state are consistent.
	gen.tilesPlayed = tilesToPlay

	numCombinations := wgen.NumSubrackCombinations()
	for subrackIdx := 0; subrackIdx < numCombinations; subrackIdx++ {
		// Look up the WMP entry and expand its words into the wmp
		// scratch buffer. If no words exist for this subrack, skip.
		if !wgen.GetSubrackWords(subrackIdx) {
			continue
		}
		numWords := wgen.NumWords()
		for wordIdx := 0; wordIdx < numWords; wordIdx++ {
			word := wgen.GetWord(wordIdx)
			for startCol := leftmostStart; startCol <= rightmostStart; startCol++ {
				if !gen.wordmapCheckPlaythroughAndCrosses(word, wordLength, startCol) {
					continue
				}
				gen.recordWMPPlaysForWord(rack, wgen, subrackIdx, startCol, wordLength, 0, 0)
				if gen.quitEarly {
					return
				}
			}
		}
	}
}

// setWMPPlaythroughFromRow walks the current row cache from
// anchor.RightmostStartCol to the right, rebuilding the wmp move gen's
// playthrough bit rack until anchor.PlaythroughBlocks blocks of
// consecutive board tiles have been consumed. Mirrors
// wmp_move_gen_set_playthrough_bit_rack.
func (gen *GordonGenerator) setWMPPlaythroughFromRow(anchor *Anchor) {
	wgen := &gen.wmpMoveGen
	wgen.ResetPlaythrough()
	want := int(anchor.PlaythroughBlocks)
	if want == 0 {
		return
	}
	inBlock := false
	blocksFound := 0
	for col := int(anchor.RightmostStartCol); col < gen.boardDim; col++ {
		ml := gen.cache.squares[col].letter
		if ml == 0 {
			if inBlock {
				if blocksFound == want {
					break
				}
				inBlock = false
			}
			continue
		}
		wgen.AddPlaythroughLetter(byte(ml.Unblank()))
		if !inBlock {
			inBlock = true
			blocksFound++
		}
	}
}

// wordmapCheckPlaythroughAndCrosses verifies that the candidate word
// (from the WMP) fits the board at the given start column: each empty
// square must allow word_letter via its cross-set, and each filled
// square must already contain (the unblanked form of) word_letter.
// On success it fills gen.strip[startCol..startCol+wordLength-1] with
// the new placements (raw word letters, no blank flag yet) and 0 for
// playthrough positions. Mirrors wordmap_gen_check_playthrough_and_crosses.
func (gen *GordonGenerator) wordmapCheckPlaythroughAndCrosses(word []byte, wordLength, startCol int) bool {
	for i := 0; i < wordLength; i++ {
		boardCol := startCol + i
		wordLetter := tilemapping.MachineLetter(word[i])
		sq := &gen.cache.squares[boardCol]
		if sq.letter == 0 {
			if !sq.crossSet.Allowed(wordLetter) {
				return false
			}
			gen.strip[boardCol] = wordLetter
		} else {
			if sq.letter.Unblank() != wordLetter {
				return false
			}
			gen.strip[boardCol] = 0
		}
	}
	return true
}

// recordWMPPlaysForWord recursively decides, for each non-playthrough
// position in the candidate word, whether the position is filled with
// a real tile from the rack or with a designated blank. When all
// blanks have been placed, the leaf calls recordWMPPlay to score and
// emit the play. Mirrors record_wmp_plays_for_word.
func (gen *GordonGenerator) recordWMPPlaysForWord(rack *tilemapping.Rack, wgen *wmp.MoveGen,
	subrackIdx, startCol, wordLength, blanksSoFar, pos int) {
	nonplaythroughTiles := wgen.GetNonplaythroughSubrack(subrackIdx)
	numBlanks := nonplaythroughTiles.GetLetter(0)
	if numBlanks == blanksSoFar {
		gen.recordWMPPlay(rack, startCol, wordLength)
		return
	}
	if pos >= wordLength {
		return
	}
	boardCol := startCol + pos
	ml := gen.strip[boardCol]
	if ml == 0 {
		// Playthrough position; nothing to decide here.
		gen.recordWMPPlaysForWord(rack, wgen, subrackIdx, startCol, wordLength, blanksSoFar, pos+1)
		return
	}
	canUnblanked, canBlanked := gen.wmpBlankPossibilities(nonplaythroughTiles, startCol, wordLength, pos)
	if canUnblanked {
		gen.recordWMPPlaysForWord(rack, wgen, subrackIdx, startCol, wordLength, blanksSoFar, pos+1)
	}
	if canBlanked {
		gen.strip[boardCol] = ml.Blank()
		gen.recordWMPPlaysForWord(rack, wgen, subrackIdx, startCol, wordLength, blanksSoFar+1, pos+1)
		gen.strip[boardCol] = ml
	}
}

// wmpBlankPossibilities mirrors get_blank_possibilities. For the
// non-playthrough position at pos in the candidate word, decides
// whether the position can be filled with a real tile from the
// subrack (canUnblanked) or with a designated blank (canBlanked),
// taking into account how many copies of word[pos] the subrack
// already contains and how many earlier positions of the same letter
// in the word have already claimed real-tile copies.
func (gen *GordonGenerator) wmpBlankPossibilities(nonplaythroughTiles *wmp.BitRack,
	startCol, wordLength, pos int) (canUnblanked, canBlanked bool) {
	targetLetter := gen.strip[startCol+pos]
	countOfTargetInSubrack := nonplaythroughTiles.GetLetter(byte(targetLetter))

	countBefore := 0
	countAtOrAfter := 0
	for p := 0; p < wordLength; p++ {
		ml := gen.strip[startCol+p]
		if ml == 0 || ml.IsBlanked() {
			// Playthrough or already-blanked: doesn't consume a
			// real copy from the subrack.
			continue
		}
		if ml != targetLetter {
			continue
		}
		if p < pos {
			countBefore++
		} else {
			countAtOrAfter++
		}
	}
	totalNonblank := countBefore + countAtOrAfter

	targetRemaining := countOfTargetInSubrack - countBefore
	canUnblanked = targetRemaining > 0
	canBlanked = totalNonblank > countOfTargetInSubrack
	return
}

// recordWMPPlay scores the fully-specified play in
// gen.strip[startCol..startCol+wordLength-1], takes the played tiles
// from the rack and updates the leavemap so the recorder fast path
// can read CurrentValue(), invokes gen.playRecorder, then restores
// the rack and leavemap. Mirrors record_wmp_play +
// update_best_move_or_insert_into_movelist_wmp.
func (gen *GordonGenerator) recordWMPPlay(rack *tilemapping.Rack, startCol, wordLength int) {
	score := gen.computeWMPScore(startCol, wordLength)

	// Take played tiles from the rack and update leavemap so the
	// recorder fast path sees the correct leave bitmask. We have to
	// undo this after recording.
	for i := 0; i < wordLength; i++ {
		ml := gen.strip[startCol+i]
		if ml == 0 {
			continue
		}
		var rackML tilemapping.MachineLetter
		if ml.IsBlanked() {
			rackML = 0
		} else {
			rackML = ml
		}
		rack.Take(rackML)
		if gen.leavemap.Initialized {
			gen.leavemap.TakeLetter(rackML, rack.LetArr[rackML])
		}
	}

	gen.playRecorder(gen, rack, startCol, startCol+wordLength-1, move.MoveTypePlay, score)

	// Restore in reverse order.
	for i := wordLength - 1; i >= 0; i-- {
		ml := gen.strip[startCol+i]
		if ml == 0 {
			continue
		}
		var rackML tilemapping.MachineLetter
		if ml.IsBlanked() {
			rackML = 0
		} else {
			rackML = ml
		}
		if gen.leavemap.Initialized {
			gen.leavemap.AddLetter(rackML, rack.LetArr[rackML])
		}
		rack.Add(rackML)
	}
}

// computeWMPScore computes the total score of the play in
// gen.strip[startCol..startCol+wordLength-1] using the row cache for
// per-square multipliers and cross-scores. Mirrors the scoring loop
// in record_wmp_play.
func (gen *GordonGenerator) computeWMPScore(startCol, wordLength int) int {
	var playedScoreTotal int
	var playthroughScoreTotal int
	var hookedCrossTotal int
	var playedCrossTotal int
	wordMultiplier := 1
	ld := gen.letterDistribution

	for i := 0; i < wordLength; i++ {
		col := startCol + i
		sq := &gen.cache.squares[col]
		ml := gen.strip[col]
		if ml == 0 {
			// Playthrough: score the existing board tile (no
			// multipliers; blanked board tiles score 0 via ld.Score).
			playthroughScoreTotal += ld.Score(sq.letter)
			continue
		}
		// Newly placed tile. Blanked placements score 0 via ld.Score.
		// Decode packed cachedSquare fields to int once.
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
	if gen.tilesPlayed == game.RackTileLimit {
		bingoBonus = 50
	}

	return (playedScoreTotal+playthroughScoreTotal)*wordMultiplier +
		hookedCrossTotal + playedCrossTotal + bingoBonus
}
