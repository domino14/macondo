package movegen

// alpha_gen.go implements move generation for the WordSmog variant, where any
// anagram of a valid word is playable. It is a port of magpie's
// recursive_gen_alpha / go_on_alpha.
//
// The shape of the search is identical to the classic Gordon generator in
// movegen.go -- same anchors, same strip, same incremental scoring, same play
// recorder contract -- but there is no word graph cursor threaded through the
// recursion. Instead:
//
//   - candidate letters at an empty square come from the alphabet (filtered by
//     the cross set) rather than from the children of a KWG node;
//   - "does the main word spell something" is answered by running the tally of
//     tiles laid down so far through the alpha dawg, at every node of the
//     search, instead of reading an accepts bit;
//   - there is no dead-end signal and no GADDAG separator hop.
//
// The consequence is that this search has almost no structural pruning: only
// cross sets, the rack and the board edges bound it, and every permutation of
// a legal multiset is a legal (and distinct) play. Shadow play is what makes
// it tractable -- see shadow.go and the WordSmog bingo gate in shadowRecord.

import (
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/alphadawg"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
)

// SetWordSmog switches this generator into WordSmog mode, using kad (an alpha
// dawg, loaded from a .kad file) as the dictionary. Pass nil to switch back to
// classic generation.
//
// WMP acceleration is turned off in WordSmog mode: the word map is built from
// the classic lexicon, so its existence checks would prune anchors that are
// perfectly playable here.
func (gen *GordonGenerator) SetWordSmog(kad *kwg.KWG) {
	gen.alphaDawg = kad
	gen.isWordSmog = kad != nil
	if gen.isWordSmog {
		gen.setWMP(nil)
	}
}

// IsWordSmog reports whether this generator is in WordSmog mode.
func (gen *GordonGenerator) IsWordSmog() bool {
	return gen.isWordSmog
}

// alphaDistSize is the number of letters in the alphabet, including the blank.
func (gen *GordonGenerator) alphaDistSize() int {
	return int(gen.letterDistribution.TileMapping().NumLetters())
}

// recursiveGenAlpha is the WordSmog counterpart of recursiveGen.
func (gen *GordonGenerator) recursiveGenAlpha(col int, rack *tilemapping.Rack,
	leftstrip, rightstrip int, uniquePlay bool,
	baseScore int, crossScores int, wordMultiplier int) {

	if gen.quitEarly {
		return
	}

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

	if curLetter != 0 {
		// Play through the tile that's already here. It joins the tally of the
		// main word, undesignated.
		raw := curLetter.Unblank()
		gen.playedTally[raw]++
		gen.goOnAlpha(col, curLetter, rack, leftstrip, rightstrip, uniquePlay,
			baseScore+gen.letterDistribution.Score(curLetter),
			crossScores,
			wordMultiplier,
		)
		gen.playedTally[raw]--
		return
	}
	if rack.Empty() {
		return
	}

	lm := int(sq.letterMul)
	cs := int(sq.crossScore)
	wm := int(sq.wordMul)
	emptyAdjacent := crossSet == board.TrivialCrossSet
	nBlank := rack.LetArr[0]
	distSize := gen.alphaDistSize()

	// Unlike the classic generator, which walks the children of the current
	// KWG node, we have to consider every letter of the alphabet that the
	// cross set allows.
	for ml := tilemapping.MachineLetter(1); int(ml) < distSize; ml++ {
		if !crossSet.Allowed(ml) {
			continue
		}
		nMl := rack.LetArr[ml]
		if nMl == 0 && nBlank == 0 {
			continue
		}
		if nMl > 0 {
			rack.Take(ml)
			if gen.leavemap.Initialized {
				gen.leavemap.TakeLetter(ml, nMl-1)
			}
			gen.tilesPlayed++
			gen.playedTally[ml]++
			sml := gen.letterDistribution.Score(ml)
			addlCrossScore := 0
			if !emptyAdjacent {
				if wm > 1 {
					addlCrossScore = wm * (cs + sml)
				} else {
					addlCrossScore = cs + sml*lm
				}
			}
			gen.goOnAlpha(col, ml, rack, leftstrip, rightstrip, uniquePlay,
				baseScore+(sml*lm),
				crossScores+addlCrossScore,
				wordMultiplier*wm)

			gen.playedTally[ml]--
			gen.tilesPlayed--
			if gen.leavemap.Initialized {
				gen.leavemap.AddLetter(ml, nMl-1)
			}
			rack.Add(ml)
		}
		// check blank. The blank leaves the rack, but it's the letter it's
		// designated as that joins the alphagram tally.
		if nBlank > 0 {
			rack.Take(0)
			if gen.leavemap.Initialized {
				gen.leavemap.TakeLetter(0, nBlank-1)
			}
			gen.tilesPlayed++
			gen.playedTally[ml]++
			gen.goOnAlpha(col, ml.Blank(), rack, leftstrip, rightstrip, uniquePlay,
				baseScore,
				crossScores+cs*wm,
				wordMultiplier*wm)
			gen.playedTally[ml]--
			gen.tilesPlayed--
			if gen.leavemap.Initialized {
				gen.leavemap.AddLetter(0, nBlank-1)
			}
			rack.Add(0)
		}
	}
}

// goOnAlpha is the WordSmog counterpart of goOn.
func (gen *GordonGenerator) goOnAlpha(curCol int, L tilemapping.MachineLetter,
	rack *tilemapping.Rack, leftstrip, rightstrip int, uniquePlay bool,
	baseScore, crossScores, wordMultiplier int) {

	var bingoBonus int
	if gen.tilesPlayed == game.RackTileLimit {
		bingoBonus = 50
	}
	// The whole main word is legal iff the tiles it holds are an anagram of a
	// word. Unlike the classic generator's accepts bit, this costs a full
	// descent of the alpha dawg at every node of the search.
	accepts := alphadawg.Accepts(gen.alphaDawg, &gen.playedTally, gen.alphaDistSize())

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
		if curCol > 0 && curCol-1 != gen.lastAnchorCol {
			gen.recursiveGenAlpha(curCol-1, rack, leftstrip, rightstrip, uniquePlay,
				baseScore, crossScores, wordMultiplier)
		}
		// The classic generator crosses the GADDAG separator here to switch
		// from extending left to extending right. There is no separator in an
		// alpha dawg, so we always turn around.
		if noLetterDirectlyLeft && gen.curAnchorCol < gen.boardDim-1 {
			gen.recursiveGenAlpha(gen.curAnchorCol+1, rack, leftstrip, rightstrip, uniquePlay,
				baseScore, crossScores, wordMultiplier)
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
		if curCol < gen.boardDim-1 {
			gen.recursiveGenAlpha(curCol+1, rack, leftstrip, rightstrip, uniquePlay,
				baseScore, crossScores, wordMultiplier)
		}
	}
}

// genForAnchor runs the appropriate recursive generator for the anchor the
// generator is currently positioned at (curRowIdx / curAnchorCol / vertical
// already set).
func (gen *GordonGenerator) genForAnchor(rack *tilemapping.Rack) {
	col := gen.curAnchorCol
	if gen.isWordSmog {
		gen.playedTally = alphadawg.Tally{}
		gen.recursiveGenAlpha(col, rack, col, col, !gen.vertical, 0, 0, 1)
		return
	}
	gen.recursiveGen(col, rack, gen.gaddag.GetRootNodeIndex(), col, col, !gen.vertical, 0, 0, 1)
}
