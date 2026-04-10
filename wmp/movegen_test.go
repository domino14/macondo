// movegen_test.go — ported from MAGPIE's test/wmp_move_gen_test.c
package wmp

import (
	"os"
	"testing"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/leavemap"
)

// wmpTestPath returns the path to a CSW24 WMP file for testing.
// Checks $MACONDO_WMP_FILE first, then $MACONDO_DATA_PATH/lexica/CSW24.wmp.
// Skips the test if neither is available.
func wmpTestPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("MACONDO_WMP_FILE"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if dp := os.Getenv("MACONDO_DATA_PATH"); dp != "" {
		p := dp + "/lexica/CSW24.wmp"
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("CSW24.wmp not available (set $MACONDO_WMP_FILE or $MACONDO_DATA_PATH)")
	return ""
}

// loadTestWMP returns the test WMP, skipping the test if the WMP
// data file is not available.
func loadTestWMP(t *testing.T) *WMP {
	t.Helper()
	path := wmpTestPath(t)
	w, err := LoadFromFile("CSW24", path)
	if err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}
	return w
}

// setDummyLeaveValues populates a leave map with leave_values[idx] =
// popcount(idx) for every index 0..127. Mirrors MAGPIE's
// set_dummy_leave_values: empty leave is 0, every additional tile in
// the leave adds 1.
func setDummyLeaveValues(lm *leavemap.LeaveMap) {
	for leaveIdx := 0; leaveIdx < 1<<RackSize; leaveIdx++ {
		bitsSet := 0
		for i := 0; i < RackSize; i++ {
			if leaveIdx&(1<<i) != 0 {
				bitsSet++
			}
		}
		lm.Values[leaveIdx] = float64(bitsSet)
	}
}

// TestWMPMoveGenInactive ports MAGPIE's test_wmp_move_gen_inactive.
func TestWMPMoveGenInactive(t *testing.T) {
	var mg MoveGen
	// Only wmp is checked by IsActive. nil wmp -> inactive.
	mg.Init(nil, nil, nil)
	if mg.IsActive() {
		t.Errorf("expected MoveGen with nil wmp to be inactive")
	}
}

// TestWMPMoveGenNonplaythroughExistence ports MAGPIE's
// test_nonplaythrough_existence. Rack VIVIFIC: there's IF (length 2)
// and VIVIFIC (length 7), and no valid words at lengths 3, 4, 5, 6.
func TestWMPMoveGenNonplaythroughExistence(t *testing.T) {
	w := loadTestWMP(t)
	ld := testEnglishLD(t)
	alph := ld.TileMapping()

	rack := tilemapping.RackFromString("VIVIFIC", alph)
	var lm leavemap.LeaveMap
	lm.Init(rack)
	lm.SetCurrentIndex(0)

	var mg MoveGen
	mg.Init(ld, rack, w)
	mg.ResetPlaythrough()
	if !mg.IsActive() {
		t.Fatalf("expected active MoveGen")
	}
	if mg.HasPlaythrough() {
		t.Errorf("expected no playthrough after reset")
	}

	// Set dummy leave values: leave_values[idx] = popcount(idx).
	// Even when checkLeaves=false the leave map index is still moved
	// during enumeration, matching MAGPIE.
	setDummyLeaveValues(&lm)

	mg.CheckNonplaythroughExistence(false, &lm)

	// IF (length 2)
	if !mg.NonplaythroughWordOfLengthExists(2) {
		t.Errorf("expected length-2 word to exist (IF)")
	}
	// no 3, 4, 5, or 6 letter words
	for length := 3; length <= 6; length++ {
		if mg.NonplaythroughWordOfLengthExists(length) {
			t.Errorf("expected no length-%d word", length)
		}
	}
	// VIVIFIC (length 7)
	if !mg.NonplaythroughWordOfLengthExists(7) {
		t.Errorf("expected length-7 word to exist (VIVIFIC)")
	}

	// With checkLeaves=false, all best-leave entries should be 0.
	bestLeaves := mg.NonplaythroughBestLeaveValues()
	for length := MinimumWordLength; length <= RackSize; length++ {
		leaveSize := RackSize - length
		if bestLeaves[leaveSize] != 0 {
			t.Errorf("checkLeaves=false: bestLeaves[%d] = %v, want 0",
				leaveSize, bestLeaves[leaveSize])
		}
	}

	// Now run again with checkLeaves=true.
	mg.CheckNonplaythroughExistence(true, &lm)
	if !mg.NonplaythroughWordOfLengthExists(2) {
		t.Errorf("expected length-2 word to exist (IF) with checkLeaves")
	}
	for length := 3; length <= 6; length++ {
		if mg.NonplaythroughWordOfLengthExists(length) {
			t.Errorf("expected no length-%d word with checkLeaves", length)
		}
	}
	if !mg.NonplaythroughWordOfLengthExists(7) {
		t.Errorf("expected length-7 word to exist (VIVIFIC) with checkLeaves")
	}

	// For each length where words exist, the best leave value should
	// equal the leave size (popcount of leave bitmask = number of
	// tiles in the leave = rackSize - wordLength).
	bestLeaves = mg.NonplaythroughBestLeaveValues()
	for wordLen := MinimumWordLength; wordLen <= RackSize; wordLen++ {
		if !mg.NonplaythroughWordOfLengthExists(wordLen) {
			continue
		}
		leaveSize := RackSize - wordLen
		want := float64(leaveSize)
		if bestLeaves[leaveSize] != want {
			t.Errorf("bestLeaves[%d] (wordLen %d) = %v, want %v",
				leaveSize, wordLen, bestLeaves[leaveSize], want)
		}
	}
}

// TestWMPMoveGenPlaythroughBingoExistence ports MAGPIE's
// test_playthrough_bingo_existence. Rack CHEESE? + various playthrough
// letters; verifies that the WMP has bingos for the right combinations.
func TestWMPMoveGenPlaythroughBingoExistence(t *testing.T) {
	w := loadTestWMP(t)
	ld := testEnglishLD(t)
	alph := ld.TileMapping()

	rack := tilemapping.RackFromString("CHEESE?", alph)
	var lm leavemap.LeaveMap
	lm.Init(rack)
	lm.SetCurrentIndex(0)

	var mg MoveGen
	mg.Init(ld, rack, w)
	mg.ResetPlaythrough()
	if !mg.IsActive() {
		t.Fatalf("expected active MoveGen")
	}
	if mg.HasPlaythrough() {
		t.Errorf("expected no playthrough after reset")
	}

	mlOf := func(s string) byte {
		v, err := alph.Val(s)
		if err != nil {
			t.Fatalf("alph.Val(%q): %v", s, err)
		}
		return byte(v)
	}

	// Add N as if shadowing left.
	mg.AddPlaythroughLetter(mlOf("N"))
	if !mg.HasPlaythrough() {
		t.Errorf("expected playthrough after adding N")
	}

	// CHEESE? + N = ENCHEErS (8-letter bingo).
	if !mg.CheckPlaythroughFullRackExistence() {
		t.Errorf("expected CHEESE?+N to find a bingo (ENCHEErS)")
	}

	// Save left state (just N), then add P as if shadowing right.
	mg.SavePlaythroughState()
	mg.AddPlaythroughLetter(mlOf("P"))

	// CHEESE? + NP = NIPCHEESE / PENNEECHS (9-letter bingo).
	if !mg.CheckPlaythroughFullRackExistence() {
		t.Errorf("expected CHEESE?+NP to find a bingo (NIPCHEESE/PENNEECHS)")
	}

	// Add Q. CHEESE?+NPQ has no bingo.
	mg.AddPlaythroughLetter(mlOf("Q"))
	if mg.CheckPlaythroughFullRackExistence() {
		t.Errorf("expected CHEESE?+NPQ to have no bingo")
	}

	// Restore the saved (left=N) state, then add I as if playing left.
	mg.RestorePlaythroughState()
	mg.AddPlaythroughLetter(mlOf("I"))
	// CHEESE? + NI = NIpCHEESE.
	if !mg.CheckPlaythroughFullRackExistence() {
		t.Errorf("expected CHEESE?+NI to find a bingo (NIpCHEESE)")
	}

	// Save left state (NI), then add P as if playing right.
	mg.SavePlaythroughState()
	mg.AddPlaythroughLetter(mlOf("P"))
	// CHEESE? + NIP = NIPCHEESEs.
	if !mg.CheckPlaythroughFullRackExistence() {
		t.Errorf("expected CHEESE?+NIP to find a bingo (NIPCHEESEs)")
	}

	// Add Q. CHEESE?+NIPQ has no bingo. (MAGPIE adds the Q without
	// re-checking but the assertion is implied; we keep it explicit.)
	mg.AddPlaythroughLetter(mlOf("Q"))
	if mg.CheckPlaythroughFullRackExistence() {
		t.Errorf("expected CHEESE?+NIPQ to have no bingo")
	}
}
