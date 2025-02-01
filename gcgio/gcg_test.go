package gcgio

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/matryer/is"
	"github.com/stretchr/testify/assert"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

var DefaultConfig = config.DefaultConfig()

var goldenFileUpdate bool

func init() {
	flag.BoolVar(&goldenFileUpdate, "update", false, "update golden files")
}

func slurp(filename string) string {
	contents, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal(err)
	}
	return string(contents)
}

func updateGolden(filename string, bts []byte) {
	// write the bts to filename
	os.WriteFile(filename, bts, 0600)
}

func compareGoldenJSON(t *testing.T, goldenFile string, actualRepr []byte) {
	expected := slurp(goldenFile)
	if goldenFileUpdate {
		updateGolden(goldenFile, actualRepr)
	} else {
		assert.JSONEq(t, expected, string(actualRepr))
	}
}

func TestParseGCGs(t *testing.T) {

	testcases := []struct {
		name       string
		gcgfile    string
		goldenfile string
		lexicon    string
	}{
		{"regular", "vs_andy.gcg", "vs_andy.json", "TWL06"},
		{"other", "doug_v_emely.gcg", "doug_v_emely.json", "NWL18"},
		{"withdrawn phony bingo", "josh2.gcg", "josh2.json", "CSW19"},
		{"challenge bonus", "vs_frentz.gcg", "vs_frentz.json", "CSW12"},
	}

	for _, tc := range testcases {
		history, err := ParseGCG(DefaultConfig, filepath.Join("testdata", tc.gcgfile))
		assert.Nil(t, err)
		assert.NotNil(t, history)
		history.Lexicon = tc.lexicon

		repr, err := json.MarshalIndent(history, "", "  ")
		assert.Nil(t, err)
		compareGoldenJSON(t, filepath.Join("testdata", tc.goldenfile), repr)
	}

}

func TestParseSpecialChar(t *testing.T) {
	history, err := ParseGCG(DefaultConfig, "./testdata/name_iso8859-1.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, history)
	assert.Equal(t, "césar", history.Players[0].Nickname)
	assert.Equal(t, "hércules", history.Players[1].Nickname)
}

func TestParseSpecialUTF8NoHeader(t *testing.T) {
	history, err := ParseGCG(DefaultConfig, "./testdata/name_utf8_noheader.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, history)
	// Since there was no encoding header, the name gets all messed up:
	assert.Equal(t, "cÃ©sar", history.Players[0].Nickname)
}

func TestParseSpecialUTF8WithHeader(t *testing.T) {
	history, err := ParseGCG(DefaultConfig, "./testdata/name_utf8_with_header.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, history)
	assert.Equal(t, "césar", history.Players[0].Nickname)
}

func TestParseUnsupportedEncoding(t *testing.T) {
	history, err := ParseGCG(DefaultConfig, "./testdata/name_weird_encoding_with_header.gcg")
	assert.NotNil(t, err)
	assert.Nil(t, history)
}

func TestParseDOSMode(t *testing.T) {
	// file has CRLF carriage returns. we should handle it.
	history, err := ParseGCG(DefaultConfig, "./testdata/utf8_dos.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, history)
	assert.Equal(t, "angwantibo", history.Players[0].Nickname)
	assert.Equal(t, "Michal_Josko", history.Players[1].Nickname)
}

func TestToGCG(t *testing.T) {
	history, err := ParseGCG(DefaultConfig, "./testdata/doug_v_emely.gcg")

	assert.Nil(t, err)
	assert.NotNil(t, history)

	gcgstr, err := GameHistoryToGCG(history, false)
	assert.Nil(t, err)

	// ignore encoding line:
	linesNew := strings.Split(gcgstr, "\n")[1:]
	linesOld := strings.Split(slurp("./testdata/doug_v_emely.gcg"), "\n")

	assert.Equal(t, len(linesNew), len(linesOld))
	for idx, ln := range linesNew {
		assert.Equal(t, strings.Fields(ln), strings.Fields(linesOld[idx]))
	}
}

func TestNewFromHistoryExcludePenultimatePass(t *testing.T) {
	is := is.New(t)

	rules, err := game.NewBasicGameRules(
		DefaultConfig,
		"",
		board.CrosswordGameLayout,
		"english",
		game.CrossScoreOnly,
		"")
	is.NoErr(err)

	gameHistory, err := ParseGCG(DefaultConfig, "./testdata/guy_vs_bot_almost_complete.gcg")
	is.NoErr(err)
	is.Equal(len(gameHistory.Events), 25)

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	alph := g.Alphabet()
	g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	is.NoErr(err)
	is.True(g != nil)
	err = g.PlayToTurn(25)
	is.NoErr(err)
	is.True(g.Playing() == pb.PlayState_PLAYING)
	is.Equal(g.RackLettersFor(1), "U")

	m := move.NewScoringMoveSimple(6, "11D", ".U", alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)

	m = move.NewPassMove(alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)

	gcgstr, err := GameHistoryToGCG(g.History(), false)
	assert.Nil(t, err)

	// ignore encoding line:
	linesNew := strings.Split(gcgstr, "\n")[1:]
	linesOld := strings.Split(slurp("./testdata/guy_vs_bot.gcg"), "\n")

	assert.Equal(t, len(linesNew), len(linesOld))
	for idx, ln := range linesNew {
		assert.Equal(t, strings.Fields(ln), strings.Fields(linesOld[idx]))
	}
}

func TestNewFromHistoryExcludePenultimateChallengeTurnLoss(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		DefaultConfig,
		"",
		board.CrosswordGameLayout,
		"english",
		game.CrossScoreOnly,
		"")
	is.NoErr(err)

	gameHistory, err := ParseGCG(DefaultConfig, "./testdata/guy_vs_bot_almost_complete.gcg")
	is.NoErr(err)
	is.Equal(len(gameHistory.Events), 25)

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	alph := g.Alphabet()
	g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	is.NoErr(err)
	is.True(g != nil)
	err = g.PlayToTurn(25)
	is.NoErr(err)
	is.True(g.Playing() == pb.PlayState_PLAYING)
	is.Equal(g.RackLettersFor(1), "U")

	m := move.NewScoringMoveSimple(6, "11D", ".U", alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)

	m = move.NewUnsuccessfulChallengePassMove(alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)

	gcgstr, err := GameHistoryToGCG(g.History(), false)
	assert.Nil(t, err)

	// ignore encoding line:
	linesNew := strings.Split(gcgstr, "\n")[1:]
	linesOld := strings.Split(slurp("./testdata/guy_vs_bot.gcg"), "\n")

	assert.Equal(t, len(linesNew), len(linesOld))
	for idx, ln := range linesNew {
		assert.Equal(t, strings.Fields(ln), strings.Fields(linesOld[idx]))
	}
}

func TestDuplicateNicknames(t *testing.T) {
	reader := strings.NewReader(`#character-encoding UTF-8
#player1 dougie Doungy B
#player2 dougie Cesar D
>dougie: FOO 8D FOO +12 12`)
	history, err := ParseGCGFromReader(DefaultConfig, reader)
	assert.Nil(t, history)
	assert.Equal(t, errDuplicateNames, err)
}

func TestPragmaWrongPlace(t *testing.T) {
	reader := strings.NewReader(`#character-encoding UTF-8
#player1 dougie Doungy B
#player2 cesar Cesar D
>dougie: FOO 8H FOO +12 12
#lexicon OSPD4`)
	history, err := ParseGCGFromReader(DefaultConfig, reader)
	assert.Nil(t, history)
	assert.Equal(t, errPragmaPrecedeEvent, err)
}

func TestIsBingo(t *testing.T) {
	reader := strings.NewReader(`#character-encoding UTF-8
#lexicon CSW19
#player1 dougie Doungy B
#player2 cesar Cesar D
>dougie: FOODIES 8D FOODIES +80 80
>cesar: ABCDEFG D7 E. +5 5
`)
	history, err := ParseGCGFromReader(DefaultConfig, reader)
	assert.Nil(t, err)
	assert.True(t, history.Events[0].IsBingo)
	assert.False(t, history.Events[1].IsBingo)
}
