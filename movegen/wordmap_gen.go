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
	"bytes"
	"fmt"
	"math"
	"math/bits"

	"github.com/domino14/word-golib/tilemapping"

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

	// Step 2: tell the move gen the anchor's word length and tiles to play.
	wgen.PlaythroughSubracksInit(&wmp.Anchor{
		TilesToPlay: tilesToPlay,
		WordLength:  wordLength,
	})

	// All plays emitted by this anchor have exactly anchor.TilesToPlay
	// new tiles. Set tilesPlayed once so the bingo bonus and the
	// recorder's per-play state are consistent, and put it back to 0 after:
	// recursiveGen counts from what it finds, so a generation after this
	// one would start from this anchor's count.
	gen.tilesPlayed = tilesToPlay
	defer func() { gen.tilesPlayed = 0 }()

	// A subrack's plays all leave the same tiles, so with the recorder that
	// keeps only the best play (and prices a play as its score plus the leave
	// map's value, which is the value the subrack carries) the anchor's score
	// bound plus the subrack's leave bounds every one of them. A subrack whose
	// bound is below the best play found is skipped before its words are
	// looked up: the recorder would turn each of its plays away. MAGPIE does
	// the same (wordmap_gen_leave_prunes_subrack).
	pruneByLeave := gen.bestPlayOnly && !gen.wmpNoSubrackLeavePrune &&
		gen.leavemap.Initialized && !gen.board.IsEmpty()
	scoreBound := int32(anchor.HighestPossibleScore)
	// The anchor's score bases bound a subrack's plays by its own tiles (see
	// wmp.MoveGen.RecordPlacement); the subrack's bound is the least.
	bases := anchor.ScoreBases
	if gen.wmpNoScoreBases {
		bases = [wmp.ScoreBaseCount]int32{math.MaxInt32 / 2, math.MaxInt32 / 2}
	}
	gen.wmpSubrackScoreBound = math.MaxInt32

	// The subracks that may have plays, by the leave check against the best
	// play so far and by what the word map can say without a lookup. The
	// best play can only get better as the subracks are gone through, so
	// one the check drops now would be dropped when reached; one it keeps
	// is checked again then.
	pruneNow := pruneByLeave && !gen.winner.IsEmpty()
	cutoff := 0.0
	if pruneNow {
		cutoff = gen.winner.Equity()
	}
	candidates := wgen.CandidateSubracks(pruneNow, scoreBound, &bases, cutoff)
	if gen.wmpSelfCheck {
		gen.wordmapCheckCandidates(candidates, pruneNow, scoreBound, &bases, cutoff)
	}

	// The start columns as a bit set, and for each letter the columns it
	// can go on (worked out when the first words turn up).
	startCols := uint32(2)<<uint(rightmostStart) - uint32(1)<<uint(leftmostStart)
	letterColsLoaded := false
	letterCols := &gen.wmpLetterCols

	for ; candidates != 0; candidates &= candidates - 1 {
		subrackIdx := bits.TrailingZeros64(candidates)
		if pruneByLeave {
			bound := wmp.SubrackScoreBound(scoreBound, &bases, wgen.GetTileScoreSum(subrackIdx))
			gen.wmpSubrackScoreBound = bound
			if !gen.winner.IsEmpty() &&
				float64(bound)+wgen.GetLeaveValue(subrackIdx) < gen.winner.Equity() {
				continue
			}
		}
		found := wgen.GetSubrackWords(subrackIdx)
		if gen.wmpSelfCheck {
			gen.wordmapCheckWords(subrackIdx, found)
		}
		if !found {
			continue
		}
		if !letterColsLoaded {
			gen.wordmapLoadLetterCols(leftmostStart, rightmostStart+wordLength-1)
			letterColsLoaded = true
		}
		words := wgen.Words()
		for off := 0; off+wordLength <= len(words); off += wordLength {
			word := words[off : off+wordLength]
			// Every start column at once: the word fits at a column when
			// each of its letters can go on its square from there.
			fit := startCols
			for i, letter := range word {
				fit &= letterCols[letter&63] >> uint(i)
			}
			if gen.wmpSelfCheck {
				gen.wordmapCheckFit(word, leftmostStart, rightmostStart, fit)
			}
			for ; fit != 0; fit &= fit - 1 {
				startCol := bits.TrailingZeros32(fit)
				gen.wordmapMarkWord(word, startCol)
				gen.recordWMPPlaysForWord(rack, wgen, subrackIdx, startCol, wordLength, 0, 0)
				if gen.quitEarly {
					return
				}
			}
		}
	}
}

// wordmapCheckCandidates panics if a subrack left out of candidates for any
// reason but its leave has words by a plain lookup. For tests.
func (gen *GordonGenerator) wordmapCheckCandidates(candidates uint64, prune bool,
	scoreBound int32, bases *[wmp.ScoreBaseCount]int32, cutoff float64) {
	wgen := &gen.wmpMoveGen
	var buffer [wmp.ResultBufferSize]byte
	for idx := 0; idx < wgen.NumSubrackCombinations(); idx++ {
		if candidates&(1<<uint(idx)) != 0 {
			continue
		}
		if prune {
			bound := wmp.SubrackScoreBound(scoreBound, bases, wgen.GetTileScoreSum(idx))
			if float64(bound)+wgen.GetLeaveValue(idx) < cutoff {
				continue
			}
		}
		if n := wgen.ReferenceSubrackWords(idx, buffer[:]); n != 0 {
			panic(fmt.Sprintf("word map subrack %d (%d tiles, length %d) left out but has %d bytes of words",
				idx, wgen.TilesToPlay(), wgen.WordLength(), n))
		}
	}
}

// wordmapCheckWords panics if the words GetSubrackWords found for the
// subrack are not those of a plain lookup. For tests.
func (gen *GordonGenerator) wordmapCheckWords(subrackIdx int, found bool) {
	wgen := &gen.wmpMoveGen
	var buffer [wmp.ResultBufferSize]byte
	n := wgen.ReferenceSubrackWords(subrackIdx, buffer[:])
	if found != (n != 0) || (found && !bytes.Equal(wgen.Words(), buffer[:n])) {
		panic(fmt.Sprintf("word map subrack %d (%d tiles, length %d): found %v with words %v, a plain lookup gives %v",
			subrackIdx, wgen.TilesToPlay(), wgen.WordLength(), found, wgen.Words(), buffer[:n]))
	}
}

// wordmapCheckFit panics if fit, the start columns the letter masks say the
// word fits at, are not those a square-by-square check finds. For tests.
func (gen *GordonGenerator) wordmapCheckFit(word []byte, leftmostStart, rightmostStart int, fit uint32) {
	for startCol := leftmostStart; startCol <= rightmostStart; startCol++ {
		want := gen.wordmapCheckPlaythroughAndCrosses(word, len(word), startCol)
		if got := fit&(1<<uint(startCol)) != 0; got != want {
			panic(fmt.Sprintf("word map word %v at column %d of row %d (vertical %v): masks say fits=%v, squares say %v",
				word, startCol, gen.curRowIdx, gen.vertical, got, want))
		}
	}
}

// wordmapLoadLetterCols works out, for each letter, the columns from first
// to last of the loaded lane where a word may have that letter: the column
// of a tile that is the letter, or an empty square whose cross-set allows
// it. Bit col of gen.wmpLetterCols[letter] is set when it may. Mirrors
// wordmap_gen_load_letter_cols.
//
// Most squares away from the tiles allow every letter, so every letter is
// first given every column, and each column is then taken from the letters
// it does not allow; a column that allows fewer than half the letters (a
// tile's, usually) is instead left out at first and given to those it does.
func (gen *GordonGenerator) wordmapLoadLetterCols(first, last int) {
	numLetters := int(gen.letterDistribution.TileMapping().NumLetters())
	universe := uint64(1)<<uint(numLetters) - 2
	var colLetters [maxBoardDim]uint64
	most := uint32(0)
	few := uint32(0)
	for col := first; col <= last; col++ {
		sq := &gen.cache.squares[col]
		letters := uint64(sq.crossSet)
		if sq.letter != 0 {
			letters = 1 << (uint8(sq.letter.Unblank()) & 63)
		}
		letters &= universe
		if 2*bits.OnesCount64(letters) >= numLetters {
			most |= 1 << uint(col)
			colLetters[col] = universe &^ letters
		} else {
			few |= 1 << uint(col)
			colLetters[col] = letters
		}
	}
	letterCols := gen.wmpLetterCols[:numLetters&63]
	for ml := range letterCols {
		letterCols[ml] = most
	}
	for cols := most; cols != 0; cols &= cols - 1 {
		col := bits.TrailingZeros32(cols)
		for letters := colLetters[col]; letters != 0; letters &= letters - 1 {
			letterCols[bits.TrailingZeros64(letters)] &^= 1 << uint(col)
		}
	}
	for cols := few; cols != 0; cols &= cols - 1 {
		col := bits.TrailingZeros32(cols)
		for letters := colLetters[col]; letters != 0; letters &= letters - 1 {
			letterCols[bits.TrailingZeros64(letters)] |= 1 << uint(col)
		}
	}
}

// wordmapMarkWord writes a word that fits at startCol into gen.strip: the
// word's letter on each empty square (no blank flag yet) and 0 on each
// tile it plays through. Mirrors wordmap_gen_mark_word.
func (gen *GordonGenerator) wordmapMarkWord(word []byte, startCol int) {
	for i, letter := range word {
		col := startCol + i
		if gen.cache.squares[col].letter == 0 {
			gen.strip[col] = tilemapping.MachineLetter(letter)
		} else {
			gen.strip[col] = 0
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
//
// wordmapGen tests every start column at once with letter masks instead;
// this is what its self-check compares them with.
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
	if gen.wmpSelfCheck && int32(score) > gen.wmpSubrackScoreBound {
		panic(fmt.Sprintf("word map play scores %d, above its subrack's bound %d (row %d, start %d, length %d, vertical %v)",
			score, gen.wmpSubrackScoreBound, gen.curRowIdx, startCol, wordLength, gen.vertical))
	}

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
	return scoreLanePlay(gen.letterDistribution, &gen.cache.squares,
		gen.strip[startCol:startCol+wordLength], startCol, gen.tilesPlayed)
}
