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

	"github.com/domino14/word-golib/tilemapping"
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

func TestEmptyDescriptionAndTitle(t *testing.T) {
	// Trailing spaces after #description or #title should not cause a parse error.
	reader := strings.NewReader("#character-encoding UTF-8\n#description    \n#title    \n#lexicon CSW19\n#player1 p1 Player One\n#player2 p2 Player Two\n")
	history, err := ParseGCGFromReader(DefaultConfig, reader)
	assert.Nil(t, err)
	assert.Equal(t, "", history.Description)
	assert.Equal(t, "", history.Title)
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

// TestWriteLastKnownRacks pins the rule for #rack1/#rack2: a game still in
// progress records the racks so a round-trip keeps them, and a finished game
// does not, since no one is on turn and the final scoring events already
// account for the leftover tiles.
func TestWriteLastKnownRacks(t *testing.T) {
	is := is.New(t)

	history, err := ParseGCG(DefaultConfig, "./testdata/doug_v_emely.gcg")
	is.NoErr(err)
	history.LastKnownRacks = []string{"ACEINRT", ""}

	// As parsed, the game is over: it has final scores.
	is.True(len(history.FinalScores) > 0)
	done, err := GameHistoryToGCG(history, true)
	is.NoErr(err)
	is.True(!strings.Contains(done, "#rack1"))

	// The same history with its ending removed is a game in progress.
	history.FinalScores = nil
	history.PlayState = pb.PlayState_PLAYING
	inProgress, err := GameHistoryToGCG(history, true)
	is.NoErr(err)
	is.True(strings.Contains(inProgress, "#rack1 ACEINRT"))
	is.True(!strings.Contains(inProgress, "#rack2"))

	// And it survives the round trip, which is the point of writing it.
	reparsed, err := ParseGCGFromReader(DefaultConfig, strings.NewReader(inProgress))
	is.NoErr(err)
	is.Equal(reparsed.LastKnownRacks[0], "ACEINRT")
}

func TestEventsThatRequireAPrecedingPlay(t *testing.T) {
	// The parser creates its internal game lazily, on the first
	// move/pass/exchange/rack token. A gcg whose first event line is a scoring
	// adjustment has no game to apply that adjustment to, and used to
	// nil-deref rather than report a parse error. Scrabblecam emits exactly
	// the end-rack-points case when a game is finalized with no recorded
	// moves.
	testcases := []struct {
		name string
		gcg  string
	}{
		{"end rack points", `#player1 bob Bob
#player2 alice Alice
>alice: (AEIOU) +10 10`},
		{"challenge bonus", `#player1 bob Bob
#player2 alice Alice
>alice: AEIOU (challenge) +5 5`},
		{"last rack penalty", `#player1 bob Bob
#player2 alice Alice
>alice: AEIOU (AEIOU) -5 -5`},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			history, err := ParseGCGFromReader(DefaultConfig, strings.NewReader(tc.gcg))
			assert.Nil(t, history)
			assert.Equal(t, errEventWithoutPlay, err)
		})
	}
}

func TestPhonyTilesReturnedWithoutGame(t *testing.T) {
	// A time penalty appends an event without creating the game, so this
	// case's len(Events) == 0 check does not cover a nil game on its own.
	reader := strings.NewReader(`#player1 bob Bob
#player2 alice Alice
>bob: ABC (time) -10 -10
>alice: DEF -- -5 0`)
	history, err := ParseGCGFromReader(DefaultConfig, reader)
	assert.Nil(t, history)
	assert.EqualError(t, err, "malformed gcg; phony tiles returned without play")
}

func TestEndRackPointsAfterPlay(t *testing.T) {
	// The same end-rack-points line parses fine once a play has created the
	// game, and still ends it.
	reader := strings.NewReader(`#player1 bob Bob
#player2 alice Alice
>bob: ACDEFGT 8H CAT +10 10
>bob: (AEIOU) +5 15`)
	history, err := ParseGCGFromReader(DefaultConfig, reader)
	assert.Nil(t, err)
	assert.Equal(t, pb.PlayState_GAME_OVER, history.PlayState)
}

// TestEventsWithoutRacks covers event lines that leave out the rack, as a GCG
// does when its annotator did not know it: one run of spaces or two stand
// where the rack would be. A line that shows tiles takes them for its rack; a
// pass shows none.
func TestEventsWithoutRacks(t *testing.T) {
	const opening = "#player1 alice Alice\n#player2 bob Bob\n>alice: ACDEFGT 8H CAT +10 10\n"
	testcases := []struct {
		name      string
		line      string
		evtType   pb.GameEvent_Type
		rack      string
		exchanged string
	}{
		{"play", ">bob:  I7 B.D +12 12", pb.GameEvent_TILE_PLACEMENT_MOVE, "BD", ""},
		{"play, one space", ">bob: I7 B.D +12 12", pb.GameEvent_TILE_PLACEMENT_MOVE, "BD", ""},
		{"play naming the tile it plays through", ">bob: I7 BAD +12 12", pb.GameEvent_TILE_PLACEMENT_MOVE, "BD", ""},
		{"play with a blank", ">bob: I7 b.D +9 9", pb.GameEvent_TILE_PLACEMENT_MOVE, "?D", ""},
		{"exchange", ">bob:  -QZ +0 0", pb.GameEvent_EXCHANGE, "QZ", "QZ"},
		{"exchange, one space", ">bob: -QZ +0 0", pb.GameEvent_EXCHANGE, "QZ", "QZ"},
		{"pass", ">bob:  - +0 0", pb.GameEvent_PASS, "", ""},
		{"pass, one space", ">bob: - +0 0", pb.GameEvent_PASS, "", ""},
		{"phony", ">alice: -- -10 0", pb.GameEvent_PHONY_TILES_RETURNED, "", ""},
		{"challenge bonus, one space", ">alice: (challenge) +5 15", pb.GameEvent_CHALLENGE_BONUS, "", ""},
		{"last rack penalty", ">bob: (EIOU) -4 -4", pb.GameEvent_END_RACK_PENALTY, "EIOU", ""},
		// Lines with a rack read as they did. A play's tiles are never a
		// "-..." or a "(...)", so a rack is not taken for a play's position.
		{"pass with a rack", ">bob: U - +0 0", pb.GameEvent_PASS, "U", ""},
		{"exchange with a rack", ">bob: SEQSPOT -QO +0 0", pb.GameEvent_EXCHANGE, "SEQSPOT", "QO"},
		{"challenge bonus with a rack", ">alice: DEFG (challenge) +5 15", pb.GameEvent_CHALLENGE_BONUS, "DEFG", ""},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			is := is.New(t)
			history, err := ParseGCGFromReader(DefaultConfig, strings.NewReader(opening+tc.line+"\n"))
			is.NoErr(err)
			evt := history.Events[len(history.Events)-1]
			is.Equal(evt.Type, tc.evtType)
			is.Equal(evt.Rack, tc.rack)
			is.Equal(evt.Exchanged, tc.exchanged)
		})
	}
}

// TestPlayWithParentheses: a play's tiles can start and end with tiles played
// through in parentheses, as "(Q)I(S)" does, and the line is read as a play
// with or without its rack; only a lone "(...)" marks another event.
func TestPlayWithParentheses(t *testing.T) {
	is := is.New(t)
	tokenFor := func(line string) Token {
		for _, datum := range GCGRegexes {
			if datum.regex.MatchString(line) {
				return datum.token
			}
		}
		return UndefinedToken
	}
	is.Equal(tokenFor(">bob: BCDEFGI H8 (Q)I +11 11"), MoveToken)
	is.Equal(tokenFor(">bob: H8 (Q)I +11 11"), MoveToken)
	is.Equal(tokenFor(">bob: BCDEFGI H8 (Q)I(S) +12 12"), MoveToken)
	is.Equal(tokenFor(">bob: H8 (Q)I(S) +12 12"), MoveToken)
	is.Equal(tokenFor(">bob: DEFG (challenge) +5 15"), ChallengeBonusToken)
	is.Equal(tokenFor(">bob: (challenge) +5 15"), ChallengeBonusToken)
}

func TestExchangeOfANumberWithoutRack(t *testing.T) {
	// Without the rack there are no tiles to count the exchanged ones from.
	reader := strings.NewReader(`#player1 alice Alice
#player2 bob Bob
>alice: ACDEFGT 8H CAT +10 10
>bob: -3 +0 0`)
	history, err := ParseGCGFromReader(DefaultConfig, reader)
	assert.Nil(t, history)
	assert.EqualError(t, err, "error in line `>bob: -3 +0 0`: an exchange given as a number of tiles needs the rack")
}

// TestRoundTripWithoutRacks: a game whose racks were left out is written with
// the passes' racks still left out, reads back the same, and replays.
func TestRoundTripWithoutRacks(t *testing.T) {
	is := is.New(t)
	history, err := ParseGCGFromReader(DefaultConfig, strings.NewReader(`#player1 alice Alice
#player2 bob Bob
>alice: ACDEFGT 8H CAT +10 10
>bob: - +0 0
>alice: DEFGIOR 9G DOG +11 21
>bob: J6 OA. +3 3
>alice: EFGINRR 10I FIG +14 35
>bob: - +0 3`))
	is.NoErr(err)

	gcg, err := GameHistoryToGCG(history, false)
	is.NoErr(err)
	is.True(strings.Contains(gcg, ">bob:  - +0 0\n"))
	is.True(strings.Contains(gcg, ">bob: OA J6 OA. +3 3\n"))
	reparsed, err := ParseGCGFromReader(DefaultConfig, strings.NewReader(gcg))
	is.NoErr(err)
	is.Equal(len(reparsed.Events), len(history.Events))
	for i, evt := range reparsed.Events {
		is.Equal(evt.Type, history.Events[i].Type)
		is.Equal(evt.Rack, history.Events[i].Rack)
		is.Equal(evt.PlayedTiles, history.Events[i].PlayedTiles)
	}

	rules, err := game.NewBasicGameRules(
		DefaultConfig, "", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	g, err := game.NewFromHistory(reparsed, rules, 0)
	is.NoErr(err)
	for turn := 0; turn <= len(reparsed.Events); turn++ {
		is.NoErr(g.PlayToTurn(turn))
	}
}
