package automatic

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/matryer/is"
)

// writeTurns puts a per-turn log in a temp dir and returns its path.
func writeTurns(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "exp.txt")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// The plays below really are legal in sequence -- CAB across the star, then CAT
// and BIT hanging off its C and B, then an S making CABS -- because the replay
// validates words, and a fixture that cannot be played teaches nothing.
const (
	playCAB = " 8G CAB"  // C-A-B across the star
	playCAT = "G8 .AT"   // down from the C
	playBIT = "I8 .IT"   // down from the B
	playS   = " 8G ...S" // CAB becomes CABS
)

// The corpus is only worth anything if a replayed position is the position the
// run actually faced. This checks what can go wrong silently: which turn the
// snapshot stops at, whose leave is the answer, and that the two halves of a
// pair are told apart rather than replayed as one game.
func TestLoadCorpus(t *testing.T) {
	is := is.New(t)

	turns := turnHeaderWithInference +
		// half 1
		"p1,seed:a,1,ABCDEFG," + playCAB + ",20,20,3,DEFG,20.0,79,0,,,,,,,,\n" +
		"p2,seed:a,2,AHIJKLT," + playCAT + ",12,12,2,HIJKL,12.0,77,20,220,DEFG,0.02,0.01,1.0000,5,300,true\n" +
		"p1,seed:a,3,DEFGSOU," + playS + ",14,34,1,DEFGOU,14.0,76,12,,,,,,,,\n" +
		"p2,seed:a,4,HIJKLIT," + playBIT + ",10,22,2,HIJKL,10.0,74,34,220,DEFGOU,0.03,0.01,1.5850,2,300,false\n" +
		// half 2 -- the turn counter starting over is the only marker
		"p2,seed:a,1,ABCDEFG," + playCAB + ",20,20,3,DEFG,20.0,79,0,,,,,,,,\n" +
		"p1,seed:a,2,AHIJKLT," + playCAT + ",12,12,2,HIJKL,12.0,77,20,220,DEFG,0.04,0.01,2.0000,1,300,true\n"

	pos, skipped, err := LoadCorpusVerbose(DefaultConfig, writeTurns(t, turns),
		"NWL20", "english", "", CorpusFilter{AllLifts: true})
	is.NoErr(err)
	is.Equal(skipped, 0)
	// Three inferences: two in the first half, one in the second.
	is.Equal(len(pos), 3)

	// The snapshot ends with the play being read, so the true leave belongs to
	// whoever made that play.
	is.Equal(pos[0].TrueLeave, "DEFG")
	is.Equal(pos[0].OppRack, "ABCDEFG")
	is.Equal(pos[0].Turn, 1)
	is.Equal(pos[0].Half, 1)
	is.Equal(pos[0].LeaveLen, 4)
	is.Equal(pos[0].TilesRemaining, 79)
	is.True(pos[0].LoggedMeasured)
	is.Equal(pos[0].Game.Turn(), 1) // exactly one play made

	// The one-tile play, which is the bucket worth studying.
	is.Equal(pos[1].TrueLeave, "DEFGOU")
	is.Equal(pos[1].LeaveLen, 6)
	is.Equal(pos[1].Turn, 3)
	is.Equal(pos[1].Game.Turn(), 3)
	is.True(!pos[1].LoggedMeasured)
	is.Equal(pos[1].LoggedRank, 2)

	// The second half is its own game, not a continuation of the first.
	is.Equal(pos[2].Half, 2)
	is.Equal(pos[2].Turn, 1)
	is.Equal(pos[2].Game.Turn(), 1)
	is.Equal(pos[2].TrueLeave, "DEFG")

	// Each snapshot is independent: replaying on has not moved an earlier one.
	is.True(pos[0].Game != pos[1].Game)
	is.Equal(pos[0].Game.Turn(), 1)
}

// The filters are what make a targeted corpus possible: only the six-tile
// leaves, or only the turns the run got wrong, or only the pre-endgame.
func TestLoadCorpusFilters(t *testing.T) {
	is := is.New(t)

	turns := turnHeaderWithInference +
		"p1,seed:a,1,ABCDEFG," + playCAB + ",20,20,3,DEFG,20.0,79,0,,,,,,,,\n" +
		"p2,seed:a,2,AHIJKLT," + playCAT + ",12,12,2,HIJKL,12.0,77,20,220,DEFG,0.02,0.01,1.0000,5,300,true\n" +
		"p1,seed:a,3,DEFGSOU," + playS + ",14,34,1,DEFGOU,14.0,6,12,,,,,,,,\n" +
		"p2,seed:a,4,HIJKLIT," + playBIT + ",10,22,2,HIJKL,10.0,3,34,220,DEFGOU,0.03,0.01,-5.0000,2,300,false\n"
	path := writeTurns(t, turns)

	// Only the six-tile leave -- the one-tile-play bucket.
	six, err := LoadCorpus(DefaultConfig, path, "NWL20", "english", "",
		CorpusFilter{LeaveLen: 6, AllLifts: true})
	is.NoErr(err)
	is.Equal(len(six), 1)
	is.Equal(six[0].TrueLeave, "DEFGOU")

	// Only what the run got wrong.
	bad, err := LoadCorpus(DefaultConfig, path, "NWL20", "english", "",
		CorpusFilter{WorseThan: -2})
	is.NoErr(err)
	is.Equal(len(bad), 1)
	is.Equal(bad[0].TrueLeave, "DEFGOU")

	// Only the pre-endgame. The bag that counts is the one at the play being
	// read, not at the inference.
	late, err := LoadCorpus(DefaultConfig, path, "NWL20", "english", "",
		CorpusFilter{MaxBag: 7, AllLifts: true})
	is.NoErr(err)
	is.Equal(len(late), 1)
	is.Equal(late[0].TrueLeave, "DEFGOU")

	early, err := LoadCorpus(DefaultConfig, path, "NWL20", "english", "",
		CorpusFilter{MinBag: 8, AllLifts: true})
	is.NoErr(err)
	is.Equal(len(early), 1)
	is.Equal(early[0].TrueLeave, "DEFG")

	lim, err := LoadCorpus(DefaultConfig, path, "NWL20", "english", "",
		CorpusFilter{AllLifts: true, Limit: 1})
	is.NoErr(err)
	is.Equal(len(lim), 1)
}

// A log whose games will not replay must say so. Returning an empty corpus
// looks exactly like a filter that matched nothing, and means the opposite.
func TestLoadCorpusReportsUnreplayableGames(t *testing.T) {
	is := is.New(t)

	turns := turnHeaderWithInference +
		"p1,seed:a,1,ABCDEFG," + playCAB + ",20,20,3,DEFG,20.0,79,0,,,,,,,,\n" +
		// HIM here makes CH and BM down the board: not words, will not replay.
		"p2,seed:a,2,HIJKLMN, 9G HIM,18,18,3,JKLN,18.0,76,20,220,DEFG,0.02,0.01,1.0,5,300,true\n"

	_, _, err := LoadCorpusVerbose(DefaultConfig, writeTurns(t, turns),
		"NWL20", "english", "", CorpusFilter{AllLifts: true})
	is.True(err != nil)

	// A filter that simply matches nothing is not an error.
	none, err := LoadCorpus(DefaultConfig, writeTurns(t, turns),
		"NWL20", "english", "", CorpusFilter{LeaveLen: 7, AllLifts: true})
	is.NoErr(err)
	is.Equal(len(none), 0)
}
