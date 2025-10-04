package gcgio

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
	"github.com/rs/zerolog/log"
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
	contents, err := os.ReadFile(filename)
	if err != nil {
		panic(err)
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
		game, err := ParseGCG(DefaultConfig, filepath.Join("testdata", tc.gcgfile))
		assert.Nil(t, err)
		assert.NotNil(t, game)
		history := game.GenerateSerializableHistory()
		history.Lexicon = tc.lexicon

		repr, err := json.MarshalIndent(history, "", "  ")
		assert.Nil(t, err)
		compareGoldenJSON(t, filepath.Join("testdata", tc.goldenfile), repr)
	}

}

func TestParseSpecialChar(t *testing.T) {
	game, err := ParseGCG(DefaultConfig, "./testdata/name_iso8859-1.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, game)
	history := game.GenerateSerializableHistory()
	assert.Equal(t, "césar", history.Players[0].Nickname)
	assert.Equal(t, "hércules", history.Players[1].Nickname)
}

func TestParseSpecialUTF8NoHeader(t *testing.T) {
	game, err := ParseGCG(DefaultConfig, "./testdata/name_utf8_noheader.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, game)
	history := game.GenerateSerializableHistory()
	// Since there was no encoding header, the name gets all messed up:
	assert.Equal(t, "cÃ©sar", history.Players[0].Nickname)
}

func TestParseSpecialUTF8WithHeader(t *testing.T) {
	game, err := ParseGCG(DefaultConfig, "./testdata/name_utf8_with_header.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, game)
	history := game.GenerateSerializableHistory()
	assert.Equal(t, "césar", history.Players[0].Nickname)
}

func TestParseUnsupportedEncoding(t *testing.T) {
	game, err := ParseGCG(DefaultConfig, "./testdata/name_weird_encoding_with_header.gcg")
	assert.NotNil(t, err)
	assert.Nil(t, game)
}

func TestParseDOSMode(t *testing.T) {
	// file has CRLF carriage returns. we should handle it.
	game, err := ParseGCG(DefaultConfig, "./testdata/utf8_dos.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, game)
	history := game.GenerateSerializableHistory()
	assert.Equal(t, "angwantibo", history.Players[0].Nickname)
	assert.Equal(t, "Michal_Josko", history.Players[1].Nickname)
}

func TestToGCG(t *testing.T) {
	game, err := ParseGCG(DefaultConfig, "./testdata/doug_v_emely.gcg")

	assert.Nil(t, err)
	assert.NotNil(t, game)
	history := game.GenerateSerializableHistory()

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

	parsedGame, err := ParseGCG(DefaultConfig, "./testdata/guy_vs_bot_almost_complete.gcg")
	is.NoErr(err)
	gameHistory := parsedGame.GenerateSerializableHistory()
	is.Equal(len(gameHistory.Events), 25)

	g, err := game.NewFromHistory(gameHistory, rules, 25)
	is.NoErr(err)
	is.True(g != nil)
	alph := g.Alphabet()
	g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	// XXX: The problem below is that the history is generated from the
	// game at turn zero (see NewFromHistory call above). So the last known
	// racks in that call are the first turn racks.
	history := g.GenerateSerializableHistory()
	err = g.PlayToTurn(25, history.LastKnownRacks)
	is.NoErr(err)
	is.True(g.Playing() == pb.PlayState_PLAYING)
	is.Equal(g.RackLettersFor(1), "U")

	m := move.NewScoringMoveSimple(6, "11D", ".U", "", alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)

	l, err := tilemapping.ToMachineWord("", alph)
	is.NoErr(err)
	m = move.NewPassMove(l, alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)

	history = g.GenerateSerializableHistory()
	gcgstr, err := GameHistoryToGCG(history, false)
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

	parsedGame, err := ParseGCG(DefaultConfig, "./testdata/guy_vs_bot_almost_complete.gcg")
	is.NoErr(err)
	gameHistory := parsedGame.GenerateSerializableHistory()
	is.Equal(len(gameHistory.Events), 25)
	log.Info().Interface("gameHistory", gameHistory).Msg("generated-gh")

	g, err := game.NewFromHistory(gameHistory, rules, 25)
	alph := g.Alphabet()
	g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	is.NoErr(err)
	is.True(g != nil)
	history := g.GenerateSerializableHistory()
	log.Info().Interface("history2", history).Msg("generated-gh-2")

	err = g.PlayToTurn(25, history.LastKnownRacks)
	is.NoErr(err)
	is.True(g.Playing() == pb.PlayState_PLAYING)
	is.Equal(g.RackLettersFor(1), "U")

	m := move.NewScoringMoveSimple(6, "11D", ".U", "", alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)

	l, err := tilemapping.ToMachineWord("", alph)
	is.NoErr(err)
	m = move.NewUnsuccessfulChallengePassMove(l, alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)

	history = g.GenerateSerializableHistory()
	gcgstr, err := GameHistoryToGCG(history, false)
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
	game, err := ParseGCGFromReader(DefaultConfig, reader)
	assert.Nil(t, game)
	assert.Equal(t, errDuplicateNames, err)
}

func TestPragmaWrongPlace(t *testing.T) {
	reader := strings.NewReader(`#character-encoding UTF-8
#player1 dougie Doungy B
#player2 cesar Cesar D
>dougie: FOO 8H FOO +12 12
#lexicon OSPD4`)
	game, err := ParseGCGFromReader(DefaultConfig, reader)
	assert.Nil(t, game)
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
	game, err := ParseGCGFromReader(DefaultConfig, reader)
	assert.Nil(t, err)
	history := game.GenerateSerializableHistory()
	assert.True(t, history.Events[0].IsBingo)
	assert.False(t, history.Events[1].IsBingo)
}
