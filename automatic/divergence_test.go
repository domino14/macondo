package automatic

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/matryer/is"
)

const gamesHeader = "gameID,HastyBot_score,NoLeaveBot_score,HastyBot_bingos," +
	"NoLeaveBot_bingos,HastyBot_turns,NoLeaveBot_turns,first,pair,divergent\n"

const turnHeader = "playerID,gameID,turn,rack,play,score,totalscore," +
	"tilesplayed,leave,equity,tilesremaining,oppscore\n"

// write a games/turn log pair into a temp dir and return the games path.
func writeLogs(t *testing.T, games, turns string) string {
	t.Helper()
	dir := t.TempDir()
	gamesPath := filepath.Join(dir, "games-exp.txt")
	if err := os.WriteFile(gamesPath, []byte(games), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "exp.txt"), []byte(turns), 0o644); err != nil {
		t.Fatal(err)
	}
	return gamesPath
}

func TestTurnLogFor(t *testing.T) {
	is := is.New(t)
	is.Equal(TurnLogFor("/tmp/x/games-exp.txt"), "/tmp/x/exp.txt")
	// Only the game log has a name we can derive the turn log from.
	is.Equal(TurnLogFor("/tmp/x/exp.txt"), "")
}

// The two halves share a game ID -- it comes from the seed they have in common
// -- so they are told apart by the turn counter starting over.
func TestSplitHalves(t *testing.T) {
	is := is.New(t)

	rows := []turnRow{{turn: 1}, {turn: 2}, {turn: 3}, {turn: 1}, {turn: 2}}
	first, second, ok := splitHalves(rows)
	is.True(ok)
	is.Equal(len(first), 3)
	is.Equal(len(second), 2)

	// A game with no second half cannot be compared.
	_, _, ok = splitHalves([]turnRow{{turn: 1}, {turn: 2}})
	is.True(!ok)
	_, _, ok = splitHalves(nil)
	is.True(!ok)
}

// The everyday case: identical racks, different plays. The report should name
// the turn, the shared rack, and what each bot did with it.
func TestAnalyzeDivergenceChoice(t *testing.T) {
	is := is.New(t)

	games := gamesHeader +
		"seed:aaa,400,350,2,1,12,12,HastyBot,0,true\n" +
		"seed:aaa,360,390,1,2,12,12,NoLeaveBot,0,true\n" +
		"seed:bbb,300,300,1,1,11,11,HastyBot,1,false\n" +
		"seed:bbb,300,300,1,1,11,11,NoLeaveBot,1,false\n"
	turns := turnHeader +
		// pair 0, first half
		"p1,seed:aaa,1,AEEFINP, 8G FE,10,10,2,AEINP,18.245,86,0\n" +
		"p2,seed:aaa,2,BDELMQU, 7C BLUMED,24,24,6,Q,24.000,84,10\n" +
		// pair 0, second half: same racks, and the seats have swapped
		"p2,seed:aaa,1,AEEFINP, 8G FANE,14,14,4,EIP,14.000,84,0\n" +
		"p1,seed:aaa,2,BDELMQU, 7C BLUMED,24,24,6,Q,24.000,84,14\n"

	gamesPath := writeLogs(t, games, turns)
	r, err := AnalyzeDivergence(gamesPath, TurnLogFor(gamesPath), 10)
	is.NoErr(err)

	is.Equal(r.Pairs, 2)
	is.Equal(r.Divergent, 1)
	is.Equal(r.ExactTies, 1) // pair 1 mirrors exactly
	is.Equal(r.Broken, 0)
	is.Equal(len(r.Details), 1)

	d := r.Details[0]
	is.Equal(d.Kind, DivergenceChoice)
	is.Equal(d.Turn, 1)
	is.Equal(d.Rack, "AEEFINP")
	// Seat 0 is played by the first bot in one half and the second in the other.
	is.Equal(d.BotA, "HastyBot")
	is.Equal(d.BotB, "NoLeaveBot")
	is.Equal(d.PlayA, "8G FE")
	is.Equal(d.PlayB, "8G FANE")
	is.Equal(d.ScoreDiff, 20) // (400-350) + (360-390)

	out := FormatDivergence(r)
	is.True(strings.Contains(out, "AEEFINP"))
	is.True(strings.Contains(out, "8G FANE"))
}

// Different racks mean the bag went out of step. That is a broken pairing, not
// a strategy difference, and has to be called out as such.
func TestAnalyzeDivergenceTiles(t *testing.T) {
	is := is.New(t)

	games := gamesHeader +
		"seed:aaa,400,350,2,1,12,12,HastyBot,0,true\n" +
		"seed:aaa,360,390,1,2,12,12,NoLeaveBot,0,true\n"
	turns := turnHeader +
		"p1,seed:aaa,1,AEEFINP, 8G FE,10,10,2,AEINP,18.245,86,0\n" +
		"p2,seed:aaa,2,BDELMQU, 7C BLUMED,24,24,6,Q,24.000,84,10\n" +
		"p2,seed:aaa,1,ZZZZZZZ, 8G ZA,22,22,2,ZZZZZ,22.000,84,0\n" +
		"p1,seed:aaa,2,BDELMQU, 7C BLUMED,24,24,6,Q,24.000,84,22\n"

	gamesPath := writeLogs(t, games, turns)
	r, err := AnalyzeDivergence(gamesPath, TurnLogFor(gamesPath), 10)
	is.NoErr(err)

	is.Equal(r.Broken, 1)
	d := r.Details[0]
	is.Equal(d.Kind, DivergenceTiles)
	is.Equal(d.RackA, "AEEFINP")
	is.Equal(d.RackB, "ZZZZZZZ")

	out := FormatDivergence(r)
	is.True(strings.Contains(out, "DIFFERENT racks"))
}

// A log without the pair columns is not a paired run, and saying so beats
// reporting zero divergences.
func TestAnalyzeDivergenceRejectsUnpairedLog(t *testing.T) {
	is := is.New(t)

	games := "gameID,HastyBot_score,NoLeaveBot_score,HastyBot_bingos," +
		"NoLeaveBot_bingos,HastyBot_turns,NoLeaveBot_turns,first\n" +
		"seed:aaa,400,350,2,1,12,12,HastyBot\n"
	gamesPath := writeLogs(t, games, turnHeader)
	_, err := AnalyzeDivergence(gamesPath, TurnLogFor(gamesPath), 10)
	is.True(err != nil)
	is.True(strings.Contains(err.Error(), "-gamepairs"))
}

// Handing over the per-turn log by mistake is easy to do and easy to diagnose.
func TestAnalyzeDivergenceRejectsTurnLog(t *testing.T) {
	is := is.New(t)

	dir := t.TempDir()
	path := filepath.Join(dir, "games-exp.txt")
	if err := os.WriteFile(path, []byte(turnHeader), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := AnalyzeDivergence(path, filepath.Join(dir, "exp.txt"), 10)
	is.True(err != nil)
	is.True(strings.Contains(err.Error(), "per-turn log"))
}
