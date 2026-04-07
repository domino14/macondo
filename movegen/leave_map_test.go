// leave_map_test.go — ported from magpie's test/leave_map_test.c
package movegen

import (
	"math"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/leavemap"
)

func makeRack(letters map[tilemapping.MachineLetter]int) *tilemapping.Rack {
	rack := &tilemapping.Rack{LetArr: make([]int, tilemapping.MaxAlphabetSize+1)}
	for ml, count := range letters {
		rack.LetArr[ml] = count
	}
	return rack
}

func TestLeaveMapIndex(t *testing.T) {
	is := is.New(t)

	// Test with 7 distinct tiles (ABCDEFG): each gets one bit
	rack := makeRack(map[tilemapping.MachineLetter]int{
		1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,
	})

	var lm leavemap.LeaveMap
	lm.Init(rack)
	is.True(lm.Initialized)
	// Full rack = all 7 bits set = 127
	is.Equal(lm.CurrentIndex, 127)

	// Take A (ml=1): should clear bit 0 → index 126
	rack.Take(1)
	lm.TakeLetter(1, rack.LetArr[1])
	is.Equal(lm.CurrentIndex, 126)

	// Store value and verify retrieval
	lm.Values[lm.CurrentIndex] = 7.0
	is.Equal(lm.CurrentValue(), 7.0)

	// Add A back → index 127
	lm.AddLetter(1, rack.LetArr[1])
	rack.Add(1)
	is.Equal(lm.CurrentIndex, 127)

	// Take B (ml=2): should clear bit 1 → index 125
	rack.Take(2)
	lm.TakeLetter(2, rack.LetArr[2])
	is.Equal(lm.CurrentIndex, 125)
	lm.Values[lm.CurrentIndex] = 8.0
	is.Equal(lm.CurrentValue(), 8.0)

	// Restore
	lm.AddLetter(2, rack.LetArr[2])
	rack.Add(2)
	is.Equal(lm.CurrentIndex, 127)

	// Take E (ml=5): bit 4 → index 111
	rack.Take(5)
	lm.TakeLetter(5, rack.LetArr[5])
	is.Equal(lm.CurrentIndex, 111)
	lm.Values[lm.CurrentIndex] = 9.0

	lm.AddLetter(5, rack.LetArr[5])
	rack.Add(5)
	is.Equal(lm.CurrentIndex, 127)

	// Take G (ml=7): bit 6 → index 63
	rack.Take(7)
	lm.TakeLetter(7, rack.LetArr[7])
	is.Equal(lm.CurrentIndex, 63)
	lm.Values[lm.CurrentIndex] = 10.0

	lm.AddLetter(7, rack.LetArr[7])
	rack.Add(7)
	is.Equal(lm.CurrentIndex, 127)

	// Verify stored values are still accessible
	rack.Take(1)
	lm.TakeLetter(1, rack.LetArr[1])
	is.Equal(lm.CurrentValue(), 7.0)
	lm.AddLetter(1, rack.LetArr[1])
	rack.Add(1)

	rack.Take(5)
	lm.TakeLetter(5, rack.LetArr[5])
	is.Equal(lm.CurrentValue(), 9.0)
	lm.AddLetter(5, rack.LetArr[5])
	rack.Add(5)
}

func TestLeaveMapDuplicateTiles(t *testing.T) {
	is := is.New(t)

	// Test with duplicate tiles: DDDIIUU (3 D's, 2 I's, 2 U's)
	rack := makeRack(map[tilemapping.MachineLetter]int{4: 3, 9: 2, 21: 2})

	var lm leavemap.LeaveMap
	lm.Init(rack)
	is.True(lm.Initialized)
	is.Equal(lm.TotalTiles, 7)
	// All 7 bits set = 127
	is.Equal(lm.CurrentIndex, 127)

	// Store initial value
	lm.Values[127] = 100.0

	// Take D: 3→2, clears bit at base+2
	rack.Take(4)
	lm.TakeLetter(4, rack.LetArr[4])
	lm.Values[lm.CurrentIndex] = 11.0

	// Take U: 2→1, clears bit
	rack.Take(21)
	lm.TakeLetter(21, rack.LetArr[21])
	lm.Values[lm.CurrentIndex] = 12.0

	// Take D: 2→1
	rack.Take(4)
	lm.TakeLetter(4, rack.LetArr[4])
	lm.Values[lm.CurrentIndex] = 13.0

	// Take I: 2→1
	rack.Take(9)
	lm.TakeLetter(9, rack.LetArr[9])
	lm.Values[lm.CurrentIndex] = 14.0
	diuIndex := lm.CurrentIndex // Save: leave = D, I, U

	// Take D: 1→0
	rack.Take(4)
	lm.TakeLetter(4, rack.LetArr[4])
	lm.Values[lm.CurrentIndex] = 15.0

	// Take I: 1→0
	rack.Take(9)
	lm.TakeLetter(9, rack.LetArr[9])
	lm.Values[lm.CurrentIndex] = 16.0

	// Take U: 1→0 — empty leave
	rack.Take(21)
	lm.TakeLetter(21, rack.LetArr[21])
	is.Equal(lm.CurrentIndex, 0)
	lm.Values[lm.CurrentIndex] = 17.0

	// Add back D, I, U in a different order and verify same index
	lm.AddLetter(4, rack.LetArr[4])
	rack.Add(4)
	lm.AddLetter(9, rack.LetArr[9])
	rack.Add(9)
	lm.AddLetter(21, rack.LetArr[21])
	rack.Add(21)

	is.Equal(lm.CurrentIndex, diuIndex)
	is.Equal(lm.CurrentValue(), 14.0)

	// Continue adding back to verify full restoration
	lm.AddLetter(4, rack.LetArr[4])
	rack.Add(4)
	lm.AddLetter(9, rack.LetArr[9])
	rack.Add(9)
	lm.AddLetter(21, rack.LetArr[21])
	rack.Add(21)
	lm.AddLetter(4, rack.LetArr[4])
	rack.Add(4)

	is.Equal(lm.CurrentIndex, 127)
	is.Equal(lm.CurrentValue(), 100.0)
}

// TestLeaveMapValuesMatchKLV validates that the incremental KWG traversal
// produces the same leave values as KLV.LeaveValue for every rack subset.
// Tests multiple racks including duplicates and blanks.
func TestLeaveMapValuesMatchKLV(t *testing.T) {
	is := is.New(t)

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.WGLConfig())
	is.NoErr(err)

	calc, err := equity.NewCombinedStaticCalculator("NWL20", DefaultConfig, "", "")
	is.NoErr(err)
	klv := calc.KLV()
	is.True(klv != nil)

	racks := []string{
		"AEINRST", // common bingo rack
		"DDDIIUU", // duplicates
		"?AEINRS", // with blank
		"??EINRS", // two blanks
		"QZ",      // short rack
		"RETAINS", // another 7
		"AAAAAAA", // all same
		"EEEIILZ", // BenchmarkSim rack (3 E's, 2 I's)
		"AAAEOOS", // from sim: 3 A's, 2 O's
		"AAEFGIK", // from game 1 turn 1
		"EIMNORS", // game 0 turn 19 near-endgame
	}

	bd := testBoard()

	for _, rackStr := range racks {
		t.Run(rackStr, func(t *testing.T) {
			rack := tilemapping.RackFromString(rackStr, gd.GetAlphabet())

			gen := NewGordonGenerator(gd, bd, dist)
			gen.klv = klv
			gen.pegValues = calc.PEGValues()
			gen.tilesInBag = 80
			gen.oppRackScore = 20
			gen.shadow.tilesInBag = 80
			gen.shadow.oppRackScore = 20
			gen.equityCalculators = []equity.EquityCalculator{calc}

			gen.leavemap.Init(rack)
			is.True(gen.leavemap.Initialized)

			// Initialize bestLeaves
			for i := range gen.shadow.bestLeaves {
				gen.shadow.bestLeaves[i] = math.Inf(-1)
			}

			// Populate via incremental traversal (also sets bestLeaves)
			gen.populateLeaveMap(rack)

			// Verify every reachable entry
			verifyLeaveMapRecursive(t, gen, klv, rack, 0)

			// Verify bestLeaves from rawLeave matches computeBestLeaves
			lmBest := gen.shadow.bestLeaves
			gen.computeBestLeaves(rack)
			for i := 0; i <= gen.leavemap.TotalTiles; i++ {
				diff := lmBest[i] - gen.shadow.bestLeaves[i]
				if diff > 0.001 || diff < -0.001 {
					t.Errorf("bestLeaves[%d] mismatch: rawLeave=%.4f compute=%.4f diff=%.4f",
						i, lmBest[i], gen.shadow.bestLeaves[i], diff)
				}
			}

		})
	}
}

func testBoard() *board.GameBoard {
	return board.MakeBoard(board.CrosswordGameBoard)
}

// verifyLeaveMapRecursive checks every subset's leave map value against KLV.
func verifyLeaveMapRecursive(t *testing.T, gen *GordonGenerator, klv *equity.KLV,
	rack *tilemapping.Rack, ml tilemapping.MachineLetter) {
	t.Helper()
	for int(ml) < len(rack.LetArr) && rack.LetArr[ml] == 0 {
		ml++
	}
	if int(ml) == len(rack.LetArr) {
		// Leaf: compare leave map value against KLV.LeaveValue + peg
		numOnRack := int(rack.NumTiles())
		var expected float64
		if numOnRack > 0 {
			var leave [game.RackTileLimit]tilemapping.MachineLetter
			n := 0
			for lml := tilemapping.MachineLetter(0); int(lml) < len(rack.LetArr); lml++ {
				for j := 0; j < rack.LetArr[lml]; j++ {
					leave[n] = lml
					n++
				}
			}
			expected = klv.LeaveValue(leave[:n])
		}
		tilesPlayed := gen.leavemap.TotalTiles - numOnRack
		bagPlusSeven := gen.tilesInBag - tilesPlayed + 7
		if bagPlusSeven >= 0 && bagPlusSeven < len(gen.pegValues) {
			expected += gen.pegValues[bagPlusSeven]
		}

		got := gen.leavemap.Values[gen.leavemap.CurrentIndex]
		if diff := got - expected; diff > 0.001 || diff < -0.001 {
			t.Errorf("leave map mismatch at index %d (numOnRack=%d): got %.4f want %.4f (diff %.4f)",
				gen.leavemap.CurrentIndex, numOnRack, got, expected, diff)
		}
		return
	}
	verifyLeaveMapRecursive(t, gen, klv, rack, ml+1)
	numthis := rack.LetArr[ml]
	for i := 0; i < numthis; i++ {
		rack.Take(ml)
		gen.leavemap.TakeLetter(ml, rack.LetArr[ml])
		verifyLeaveMapRecursive(t, gen, klv, rack, ml+1)
	}
	for i := 0; i < numthis; i++ {
		gen.leavemap.AddLetter(ml, rack.LetArr[ml])
		rack.Add(ml)
	}
}
