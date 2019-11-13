package movegen

import (
	"fmt"
	"sync"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
)

// We move threaded versions of the recursive move generation
// functions to this file. We could have  branching logic in the movegen
// to figure out whether it's threaded (or just use a threading size of 1)
// but we are interested in generating moves as fast as possible and it's
// faster to just copy and paste, sadly.

// We use channels and other async constructs here.

func (gen *GordonGenerator) genByOrientationThreaded(rack *alphabet.Rack,
	dir board.BoardDirection) {

	dim := gen.board.Dim()

	work := make([][]int, gen.threads)

	for row := 0; row < dim; row++ {
		thread := row % gen.threads
		work[thread] = append(work[thread], row)
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})
	playChan := make(chan int)
	fmt.Println("WORK", work)
	for t := range work {
		// t is the thread index in the slice.
		// Spawn off a goroutine.
		wg.Add(1)
		gen.shared[t].rack.CopyFrom(rack)
		gen.shared[t].tilesPlayed = 0
		go func(t int) {
			defer wg.Done()

			for _, row := range work[t] {
				fmt.Println("ROW", row, "T", t)
				gen.shared[t].curRowIdx = row
				gen.shared[t].lastAnchorCol = 100
				for col := 0; col < dim; col++ {
					if gen.board.IsAnchor(row, col, dir) {
						fmt.Println("COL IS ANCHOR", col)
						gen.shared[t].curAnchorCol = col
						gen.genThreaded(col, alphabet.MachineWord([]alphabet.MachineLetter{}),
							gen.shared[t].rack, gen.gaddag.GetRootNodeIndex(), t, playChan)
						gen.shared[t].lastAnchorCol = col
						fmt.Println("LATAS")
					}
				}
			}
		}(t)
	}
	go func(stopChan chan struct{}) {
		for {
			select {
			case <-stop:
				return
			case t := <-playChan:
				play := gen.shared[t].latestPlay
				fmt.Println("THE PLAY TO RECORD IS", play)
				gen.RecordPlay(play.word, play.startRow, play.startCol,
					play.leave, play.tilesPlayed)
			}
		}
	}(stop)

	wg.Wait()
	// Make the play-recording goroutine stop.
	stop <- struct{}{}

}

// genThreaded is an implementation of the Gordon Gen function using threads
func (gen *GordonGenerator) genThreaded(col int, word alphabet.MachineWord,
	rack *alphabet.Rack, nodeIdx uint32, t int, playChan chan int) {

	var csDirection board.BoardDirection
	curRowIdx := gen.shared[t].curRowIdx
	// If a letter L is already on this square, then GoOn...
	curSquare := gen.board.GetSquare(curRowIdx, col)
	curLetter := curSquare.Letter()

	if gen.vertical {
		csDirection = board.HorizontalDirection
	} else {
		csDirection = board.VerticalDirection
	}
	crossSet := gen.board.GetCrossSet(curRowIdx, col, csDirection)

	if !curSquare.IsEmpty() {
		nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, curLetter.Unblank())
		gen.goOnThreaded(col, curLetter, word, rack, nnIdx, nodeIdx, t, playChan)

	} else if !rack.Empty() {
		for ml := alphabet.MachineLetter(0); ml < alphabet.MachineLetter(gen.numPossibleLetters); ml++ {
			if rack.LetArr[ml] == 0 {
				continue
			}
			if crossSet.Allowed(ml) {
				nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, ml)
				rack.Take(ml)
				gen.shared[t].tilesPlayed++
				gen.goOnThreaded(col, ml, word, rack, nnIdx, nodeIdx, t, playChan)
				rack.Add(ml)
				gen.shared[t].tilesPlayed--
			}

		}

		if rack.LetArr[alphabet.BlankMachineLetter] > 0 {
			// It's a blank. Loop only through letters in the cross-set.
			for i := 0; i < gen.numPossibleLetters; i++ {
				if crossSet.Allowed(alphabet.MachineLetter(i)) {
					nnIdx := gen.gaddag.NextNodeIdx(nodeIdx, alphabet.MachineLetter(i))
					rack.Take(alphabet.BlankMachineLetter)
					gen.shared[t].tilesPlayed++
					gen.goOnThreaded(col, alphabet.MachineLetter(i).Blank(), word, rack, nnIdx, nodeIdx,
						t, playChan)
					rack.Add(alphabet.BlankMachineLetter)
					gen.shared[t].tilesPlayed--
				}
			}
		}

	}
}

// goOnThreaded is an implementation of the Gordon GoOn function.
func (gen *GordonGenerator) goOnThreaded(curCol int, L alphabet.MachineLetter, word alphabet.MachineWord,
	rack *alphabet.Rack, newNodeIdx uint32, oldNodeIdx uint32, t int,
	playChan chan int) {

	curRowIdx := gen.shared[t].curRowIdx
	curAnchorCol := gen.shared[t].curAnchorCol
	tilesPlayed := gen.shared[t].tilesPlayed
	if curCol <= curAnchorCol {
		if !gen.board.GetSquare(curRowIdx, curCol).IsEmpty() {
			word = append([]alphabet.MachineLetter{alphabet.PlayedThroughMarker}, word...)
		} else {
			word = append([]alphabet.MachineLetter{L}, word...)
		}
		// if L on OldArc and no letter directly left, then record play.
		// roomToLeft is true unless we are right at the edge of the board.
		//roomToLeft := true
		noLetterDirectlyLeft := curCol == 0 ||
			gen.board.GetSquare(curRowIdx, curCol-1).IsEmpty()

		// Check to see if there is a letter directly to the left.
		if gen.gaddag.InLetterSet(L, oldNodeIdx) && noLetterDirectlyLeft && tilesPlayed > 0 {
			gen.shared[t].latestPlay.startRow = curRowIdx
			gen.shared[t].latestPlay.startCol = curCol
			gen.shared[t].latestPlay.word = word
			gen.shared[t].latestPlay.leave = rack.TilesOn()
			gen.shared[t].latestPlay.tilesPlayed = tilesPlayed
			fmt.Println("ONE", gen.shared[t].latestPlay)
			playChan <- t
		}
		if newNodeIdx == 0 {
			return
		}
		// Keep generating prefixes if there is room to the left, and don't
		// revisit an anchor we just saw.
		// This seems to work because we always shift direction afterwards, so we're
		// only looking at the first of a consecutive set of anchors going backwards,
		// and then always looking forward from then on.
		if curCol > 0 && curCol-1 != gen.shared[t].lastAnchorCol {
			gen.genThreaded(curCol-1, word, rack, newNodeIdx, t, playChan)
		}
		// Then shift direction.
		// Get the index of the SeparationToken
		separationNodeIdx := gen.gaddag.NextNodeIdx(newNodeIdx, alphabet.SeparationMachineLetter)
		// Check for no letter directly left AND room to the right (of the anchor
		// square)
		if separationNodeIdx != 0 && noLetterDirectlyLeft && curAnchorCol < gen.board.Dim()-1 {
			gen.genThreaded(curAnchorCol+1, word, rack, separationNodeIdx, t, playChan)
		}

	} else {
		if !gen.board.GetSquare(curRowIdx, curCol).IsEmpty() {
			word = append(word, alphabet.PlayedThroughMarker)
		} else {
			word = append(word, L)
		}

		noLetterDirectlyRight := curCol == gen.board.Dim()-1 ||
			gen.board.GetSquare(curRowIdx, curCol+1).IsEmpty()
		if gen.gaddag.InLetterSet(L, oldNodeIdx) && noLetterDirectlyRight && tilesPlayed > 0 {
			gen.shared[t].latestPlay.startRow = curRowIdx
			gen.shared[t].latestPlay.startCol = curCol - len(word) + 1
			gen.shared[t].latestPlay.word = word
			gen.shared[t].latestPlay.leave = rack.TilesOn()
			gen.shared[t].latestPlay.tilesPlayed = tilesPlayed
			fmt.Println("TWO", gen.shared[t].latestPlay)

			playChan <- t
		}
		if newNodeIdx != 0 && curCol < gen.board.Dim()-1 {
			// There is room to the right
			gen.genThreaded(curCol+1, word, rack, newNodeIdx, t, playChan)
		}
	}
}
