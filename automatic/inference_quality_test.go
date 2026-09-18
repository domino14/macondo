package automatic

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/matryer/is"
)

const turnHeaderWithInference = "playerID,gameID,turn,rack,play,score,totalscore," +
	"tilesplayed,leave,equity,tilesremaining,oppscore" + InferenceLogColumns + "\n"

func writeTurnLog(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "exp.txt")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// row builds a per-turn row; pass "" for the inference columns to make it a row
// from the bot that does not infer.
func row(nick, infer string) string {
	base := nick + ",seed:a,3,ABCDEFG, 8G AB,20,20,2,CDEFG,20.000,80,10"
	return base + infer + "\n"
}

// The headline is the mean lift, but the rest of the report has to line up with
// it: how often inference beat the prior, and how often the truth was measured
// rather than imputed.
func TestAnalyzeInferenceQuality(t *testing.T) {
	is := is.New(t)

	body := turnHeaderWithInference +
		// lift +1 (posterior twice the prior), measured
		row("p1", ",200,AB,0.02,0.01,1.0000,5,300,true") +
		// lift -1, imputed
		row("p1", ",200,CD,0.005,0.01,-1.0000,50,300,false") +
		// lift +2, imputed, and a longer leave so the by-length breakdown has
		// more than one bucket to show
		row("p1", ",200,EFG,0.04,0.01,2.0000,2,300,false") +
		// inference ran but had nothing to grade
		row("p1", ",0,,,,,,,") +
		// the other bot: the same columns, left empty, so the file stays a table
		row("p2", emptyInferenceFields)

	q, err := AnalyzeInferenceQuality(writeTurnLog(t, body))
	is.NoErr(err)

	is.Equal(q.Inferences, 3)
	is.Equal(q.Unscored, 1)
	is.Equal(q.LiftBits.Iterations(), 3)
	// (1 - 1 + 2) / 3
	is.True(math.Abs(q.LiftBits.Mean()-2.0/3.0) < 1e-9)
	is.True(math.Abs(q.LiftPercentile(50)-1.0) < 1e-9)
	is.Equal(q.Better, 2)
	is.Equal(q.Worse, 1)
	is.Equal(q.Measured, 1)
	is.Equal(q.RuledOut, 0)

	// Ranks become percentiles so leave spaces of different sizes compare.
	is.True(q.RankPct.Mean() > 0)
	is.True(math.Abs(q.Leaves.Mean()-300) < 1e-9)

	out := FormatInferenceQuality(q)
	is.True(strings.Contains(out, "bits per inference"))
	is.True(strings.Contains(out, "2 tile(s)"))
}

// A posterior that gives the true leave no weight has no ratio to report. It is
// counted separately rather than averaged in as negative infinity, since it is
// the failure that matters most: the simmer never samples the truth.
func TestAnalyzeInferenceQualityRuledOut(t *testing.T) {
	is := is.New(t)

	body := turnHeaderWithInference +
		row("p1", ",200,AB,0.02,0.01,1.0000,5,300,true") +
		row("p1", ",200,ZZ,0,0.01,NaN,0,300,false")

	q, err := AnalyzeInferenceQuality(writeTurnLog(t, body))
	is.NoErr(err)
	is.Equal(q.Inferences, 2)
	is.Equal(q.RuledOut, 1)
	is.Equal(q.LiftBits.Iterations(), 1) // the ruled-out one is not averaged
	is.True(math.Abs(q.LiftBits.Mean()-1.0) < 1e-9)
	is.True(strings.Contains(FormatInferenceQuality(q), "ruled the true leave out"))
}

// Handing over the wrong file, or a log from a run with no inferring bot, should
// say so rather than report zeros.
func TestAnalyzeInferenceQualityRejectsWrongInput(t *testing.T) {
	is := is.New(t)

	_, err := AnalyzeInferenceQuality(writeTurnLog(t, gamesHeader))
	is.True(err != nil)
	is.True(strings.Contains(err.Error(), "per-turn log"))

	noInference := "playerID,gameID,turn,rack,play,score,totalscore," +
		"tilesplayed,leave,equity,tilesremaining,oppscore\n" + row("p1", "")
	// (a run with no inferring bot at all: narrower header, matching rows)
	_, err = AnalyzeInferenceQuality(writeTurnLog(t, noInference))
	is.True(err != nil)
	is.True(strings.Contains(err.Error(), "no inference columns"))
}
