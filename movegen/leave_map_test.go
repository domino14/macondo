// leave_map_test.go — ported from magpie's test/leave_map_test.c
package movegen

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
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

	var lm leaveMap
	lm.init(rack)
	is.True(lm.initialized)
	// Full rack = all 7 bits set = 127
	is.Equal(lm.currentIndex, 127)

	// Take A (ml=1): should clear bit 0 → index 126
	rack.Take(1)
	lm.takeLetter(1, rack.LetArr[1])
	is.Equal(lm.currentIndex, 126)

	// Store value and verify retrieval
	lm.values[lm.currentIndex] = 7.0
	is.Equal(lm.currentValue(), 7.0)

	// Add A back → index 127
	lm.addLetter(1, rack.LetArr[1])
	rack.Add(1)
	is.Equal(lm.currentIndex, 127)

	// Take B (ml=2): should clear bit 1 → index 125
	rack.Take(2)
	lm.takeLetter(2, rack.LetArr[2])
	is.Equal(lm.currentIndex, 125)
	lm.values[lm.currentIndex] = 8.0
	is.Equal(lm.currentValue(), 8.0)

	// Restore
	lm.addLetter(2, rack.LetArr[2])
	rack.Add(2)
	is.Equal(lm.currentIndex, 127)

	// Take E (ml=5): bit 4 → index 111
	rack.Take(5)
	lm.takeLetter(5, rack.LetArr[5])
	is.Equal(lm.currentIndex, 111)
	lm.values[lm.currentIndex] = 9.0

	lm.addLetter(5, rack.LetArr[5])
	rack.Add(5)
	is.Equal(lm.currentIndex, 127)

	// Take G (ml=7): bit 6 → index 63
	rack.Take(7)
	lm.takeLetter(7, rack.LetArr[7])
	is.Equal(lm.currentIndex, 63)
	lm.values[lm.currentIndex] = 10.0

	lm.addLetter(7, rack.LetArr[7])
	rack.Add(7)
	is.Equal(lm.currentIndex, 127)

	// Verify stored values are still accessible
	rack.Take(1)
	lm.takeLetter(1, rack.LetArr[1])
	is.Equal(lm.currentValue(), 7.0)
	lm.addLetter(1, rack.LetArr[1])
	rack.Add(1)

	rack.Take(5)
	lm.takeLetter(5, rack.LetArr[5])
	is.Equal(lm.currentValue(), 9.0)
	lm.addLetter(5, rack.LetArr[5])
	rack.Add(5)
}

func TestLeaveMapDuplicateTiles(t *testing.T) {
	is := is.New(t)

	// Test with duplicate tiles: DDDIIUU (3 D's, 2 I's, 2 U's)
	rack := makeRack(map[tilemapping.MachineLetter]int{4: 3, 9: 2, 21: 2})

	var lm leaveMap
	lm.init(rack)
	is.True(lm.initialized)
	is.Equal(lm.totalTiles, 7)
	// All 7 bits set = 127
	is.Equal(lm.currentIndex, 127)

	// Store initial value
	lm.values[127] = 100.0

	// Take D: 3→2, clears bit at base+2
	rack.Take(4)
	lm.takeLetter(4, rack.LetArr[4])
	lm.values[lm.currentIndex] = 11.0

	// Take U: 2→1, clears bit
	rack.Take(21)
	lm.takeLetter(21, rack.LetArr[21])
	lm.values[lm.currentIndex] = 12.0

	// Take D: 2→1
	rack.Take(4)
	lm.takeLetter(4, rack.LetArr[4])
	lm.values[lm.currentIndex] = 13.0

	// Take I: 2→1
	rack.Take(9)
	lm.takeLetter(9, rack.LetArr[9])
	lm.values[lm.currentIndex] = 14.0
	diuIndex := lm.currentIndex // Save: leave = D, I, U

	// Take D: 1→0
	rack.Take(4)
	lm.takeLetter(4, rack.LetArr[4])
	lm.values[lm.currentIndex] = 15.0

	// Take I: 1→0
	rack.Take(9)
	lm.takeLetter(9, rack.LetArr[9])
	lm.values[lm.currentIndex] = 16.0

	// Take U: 1→0 — empty leave
	rack.Take(21)
	lm.takeLetter(21, rack.LetArr[21])
	is.Equal(lm.currentIndex, 0)
	lm.values[lm.currentIndex] = 17.0

	// Add back D, I, U in a different order and verify same index
	lm.addLetter(4, rack.LetArr[4])
	rack.Add(4)
	lm.addLetter(9, rack.LetArr[9])
	rack.Add(9)
	lm.addLetter(21, rack.LetArr[21])
	rack.Add(21)

	is.Equal(lm.currentIndex, diuIndex)
	is.Equal(lm.currentValue(), 14.0)

	// Continue adding back to verify full restoration
	lm.addLetter(4, rack.LetArr[4])
	rack.Add(4)
	lm.addLetter(9, rack.LetArr[9])
	rack.Add(9)
	lm.addLetter(21, rack.LetArr[21])
	rack.Add(21)
	lm.addLetter(4, rack.LetArr[4])
	rack.Add(4)

	is.Equal(lm.currentIndex, 127)
	is.Equal(lm.currentValue(), 100.0)
}
