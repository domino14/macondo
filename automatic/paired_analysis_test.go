package automatic

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/matryer/is"
)

func writeGamesLog(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "games-exp.txt")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// A log without the pair columns is analyzed exactly as before.
func TestUnpairedLogHasNoPairedResult(t *testing.T) {
	is := is.New(t)

	body := "gameID,A_score,B_score,A_bingos,B_bingos,A_turns,B_turns,first\n" +
		"g1,400,350,2,1,12,12,A\n" +
		"g2,360,390,1,2,12,12,B\n"
	r, err := AnalyzeLogFileData(writeGamesLog(t, body))
	is.NoErr(err)
	is.Equal(r.GamesPlayed, 2)
	is.True(r.Paired == nil)
}

// The pair, not the game, is the observation: the two halves are added
// together and the tests run over those sums.
func TestPairedResultBasics(t *testing.T) {
	is := is.New(t)

	// Pair 0: +50 then -30, so bot A is +20 over the pair.
	// Pair 1: -10 then -10, so bot A is -20.
	// Pair 2: +40 then -40, an exact tie that never diverged.
	body := gamesHeader +
		"g1,400,350,2,1,12,12,HastyBot,0,true\n" +
		"g1,360,390,1,2,12,12,NoLeaveBot,0,true\n" +
		"g2,300,310,1,1,11,11,HastyBot,1,true\n" +
		"g2,300,310,1,1,11,11,NoLeaveBot,1,true\n" +
		"g3,440,400,2,2,12,12,HastyBot,2,false\n" +
		"g3,400,440,2,2,12,12,NoLeaveBot,2,false\n"

	r, err := AnalyzeLogFileData(writeGamesLog(t, body))
	is.NoErr(err)
	is.True(r.Paired != nil)

	p := r.Paired
	is.Equal(p.Pairs, 3)
	is.Equal(p.Incomplete, 0)
	is.Equal(p.Divergent, 2)
	is.Equal(p.ExactTies, 1)

	// Mean of +20, -20, 0 is 0 -- and the tied pair is counted, not dropped.
	is.Equal(p.Margin.Iterations(), 3)
	is.True(math.Abs(p.Margin.Mean()) < 1e-9)

	perGame, _ := p.MarginPerGame()
	is.True(math.Abs(perGame) < 1e-9)
}

// Half a pair is not a result. Counting it would read one game's luck as a
// verdict, so it is left out and reported.
func TestPairedResultDropsIncompletePairs(t *testing.T) {
	is := is.New(t)

	body := gamesHeader +
		"g1,400,350,2,1,12,12,HastyBot,0,true\n" +
		"g1,360,390,1,2,12,12,NoLeaveBot,0,true\n" +
		"g2,500,300,3,0,12,12,HastyBot,1,true\n" // second half missing

	r, err := AnalyzeLogFileData(writeGamesLog(t, body))
	is.NoErr(err)

	p := r.Paired
	is.Equal(p.Pairs, 1)
	is.Equal(p.Incomplete, 1)
	// Only the complete pair contributes: +50 - 30 = +20.
	is.True(math.Abs(p.Margin.Mean()-20) < 1e-9)
	is.Equal(r.GamesPlayed, 3) // the per-game view still sees every game
}

// A bot against itself: every pair mirrors, so the margin is exactly zero with
// no uncertainty, the correlation is exactly -1, and nothing is significant.
func TestPairedResultSelfPlay(t *testing.T) {
	is := is.New(t)

	body := gamesHeader +
		"g1,400,350,2,1,12,12,HastyBot,0,false\n" +
		"g1,350,400,1,2,12,12,HastyBot1,0,false\n" +
		"g2,300,420,1,3,11,11,HastyBot,1,false\n" +
		"g2,420,300,3,1,11,11,HastyBot1,1,false\n" +
		"g3,455,455,2,2,12,12,HastyBot,2,false\n" +
		"g3,455,455,2,2,12,12,HastyBot1,2,false\n"

	r, err := AnalyzeLogFileData(writeGamesLog(t, body))
	is.NoErr(err)

	p := r.Paired
	is.Equal(p.Pairs, 3)
	is.Equal(p.Divergent, 0)
	is.Equal(p.ExactTies, 3)
	is.True(math.Abs(p.Margin.Mean()) < 1e-9)
	is.True(math.Abs(p.PairedSE()) < 1e-9) // known exactly
	is.True(math.Abs(p.WinShare.Mean()-0.5) < 1e-9)
	is.True(math.Abs(p.MarginPValue-1.0) < 1e-9)
	is.True(p.Correlation < -0.999)

	// The per-game view, by contrast, sees wild swings and claims real
	// uncertainty about a quantity that is known exactly.
	is.True(r.ScoreDiff.StandardError() > 10)

	out := FormatTable(r)
	is.True(strings.Contains(out, "no uncertainty at all"))
	is.True(strings.Contains(out, "No pair diverged"))
}

// Pairing only pays off when the halves anti-correlate, and the report says how
// much it paid.
func TestPairedResultBeatsNaiveStandardError(t *testing.T) {
	is := is.New(t)

	// Big tile swings that cancel within each pair, plus a steady 10-point
	// edge for bot A that survives them.
	var b strings.Builder
	b.WriteString(gamesHeader)
	swings := []int{200, -150, 90, -60, 170, -30, 120, -110, 40, -95}
	for i, s := range swings {
		// Half A: A wins by swing+10. Half B: A loses by swing, so the pair
		// nets +10 for A whatever the tiles did.
		b.WriteString(fmtRow("g", i, 400+s+10, 400, "HastyBot", i, true))
		b.WriteString(fmtRow("g", i, 400-s, 400, "NoLeaveBot", i, true))
	}
	r, err := AnalyzeLogFileData(writeGamesLog(t, b.String()))
	is.NoErr(err)

	p := r.Paired
	is.Equal(p.Pairs, 10)
	// Every pair nets +10 for bot A, so +5 per game with no spread at all.
	is.True(math.Abs(p.Margin.Mean()-10) < 1e-9)
	is.True(p.PairedSE() < 1e-9)
	is.True(p.NaiveScoreSE > 20) // the per-game view sees only the swings
	is.True(p.Correlation < -0.9)

	// Ten pairs all pointing the same way is the most ten pairs can say.
	is.True(math.Abs(p.MarginPValue-2.0/1024.0) < 1e-9)
}

func fmtRow(prefix string, pair, s1, s2 int, first string, pairIdx int, divergent bool) string {
	div := "false"
	if divergent {
		div = "true"
	}
	return strings.Join([]string{
		prefix + itoa(pair),
		itoa(s1), itoa(s2), "1", "1", "12", "12", first, itoa(pairIdx), div,
	}, ",") + "\n"
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var d []byte
	for i > 0 {
		d = append([]byte{byte('0' + i%10)}, d...)
		i /= 10
	}
	if neg {
		return "-" + string(d)
	}
	return string(d)
}
