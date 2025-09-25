package game

import (
	"encoding/json"
	"io"
	"os"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/testhelpers"
)

var DefaultConfig = config.DefaultConfig()

func TestNewGame(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	is.NoErr(err)
	game, err := NewGame(rules, players)
	is.NoErr(err)
	game.StartGame()
	is.Equal(game.bag.TilesRemaining(), 86)
}

func TestBackup(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	is.NoErr(err)
	game, _ := NewGame(rules, players)

	game.StartGame()
	// Some positive number.
	game.SetStateStackLength(5)
	game.SetBackupMode(SimulationMode)
	// Overwrite the player on turn to be JD:
	game.SetPlayerOnTurn(0)
	alph := game.Alphabet()
	game.SetRackFor(0, tilemapping.RackFromString("ACEOTV?", alph))

	m := move.NewScoringMoveSimple(20, "H7", "AVOCET", "?", alph)
	game.PlayMove(m, false, 0)

	is.Equal(game.stackPtr, 1)
	is.Equal(game.players[0].points, 20)
	is.Equal(game.players[1].points, 0)
	is.Equal(game.bag.TilesRemaining(), 80)

	game.UnplayLastMove()
	is.Equal(game.stackPtr, 0)
	is.Equal(game.players[0].points, 0)
	is.Equal(game.players[1].points, 0)
	is.Equal(game.bag.TilesRemaining(), 86)
	is.Equal(game.players[0].rackLetters(), "?ACEOTV")
}

func TestValidate(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}

	rules, err := NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	is.NoErr(err)

	g, _ := NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetPlayerOnTurn(0)
	g.SetRackFor(0, tilemapping.RackFromString("HIS", alph))
	g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	m := move.NewScoringMoveSimple(12, "H7", "HIS", "", alph)
	words, err := g.ValidateMove(m)
	is.NoErr(err)
	is.Equal(len(words), 1)
	g.PlayMove(m, true, 0)
	is.Equal(g.history.Events[len(g.history.Events)-1].WordsFormed,
		[]string{"HIS"})
	g.SetRackFor(1, tilemapping.RackFromString("OIK", alph))
	m = move.NewScoringMoveSimple(13, "G8", "OIK", "", alph)
	words, err = g.ValidateMove(m)
	is.NoErr(err)
	is.Equal(len(words), 3)
	g.PlayMove(m, true, 0)
	is.Equal(g.history.Events[len(g.history.Events)-1].WordsFormed,
		[]string{"OIK", "OI", "IS"})

	g.SetRackFor(0, tilemapping.RackFromString("ADITT", alph))
	m = move.NewScoringMoveSimple(22, "10E", "DI.TAT", "", alph)
	words, err = g.ValidateMove(m)
	is.NoErr(err)
	is.Equal(len(words), 2)
	g.PlayMove(m, true, 0)
	is.Equal(g.history.Events[len(g.history.Events)-1].WordsFormed,
		[]string{"DIKTAT", "HIST"})
}

func TestPlayToTurnWithPhony(t *testing.T) {
	is := is.New(t)

	rules, err := NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	is.NoErr(err)
	jsonFile, err := os.Open("./testdata/history1.json")
	is.NoErr(err)
	defer jsonFile.Close()

	bytes, err := io.ReadAll(jsonFile)
	is.NoErr(err)
	gameHistory := &pb.GameHistory{}
	err = json.Unmarshal(bytes, gameHistory)
	is.NoErr(err)

	g, err := NewFromHistory(gameHistory, rules, len(gameHistory.Events))
	is.NoErr(err)

	is.Equal(g.RackFor(0).TilesOn().UserVisible(testhelpers.EnglishAlphabet()),
		"EEHKNOQ")
	is.Equal(g.RackFor(1).TilesOn().UserVisible(testhelpers.EnglishAlphabet()),
		"?DEMOOW")
	log.Debug().Interface("lex", g.lexicon.Name()).Interface("wf", g.lastWordsFormed).Msg("info")
	// Player 0 challenges opponent's phony
	valid, err := g.ChallengeEvent(0, 1000)
	is.NoErr(err)
	is.Equal(valid, false)

	// check that game rolled back successfully
	is.Equal(len(g.History().Events), 3)
	is.Equal(g.History().Events[2].Type, pb.GameEvent_PHONY_TILES_RETURNED)

	// The tiles in the phony "DORMINE" should be gone.
	// An already empty tile to the left of DORMINE*
	is.Equal(g.Board().GetLetter(6, 6).IsPlayedTile(), false)
	// The D in DORMINE:
	is.Equal(g.Board().GetLetter(6, 7).IsPlayedTile(), false)

	// p1 gets their phony tiles back
	is.Equal(g.RackFor(1).TilesOn().UserVisible(testhelpers.EnglishAlphabet()),
		"DEIMNOR")
	// p0 still has their rack
	is.Equal(g.RackFor(0).TilesOn().UserVisible(testhelpers.EnglishAlphabet()),
		"EEHKNOQ")
}

func TestToCGP(t *testing.T) {
	is := is.New(t)

	rules, err := NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	is.NoErr(err)
	jsonFile, err := os.Open("./testdata/history1.json")
	is.NoErr(err)
	defer jsonFile.Close()

	bytes, err := io.ReadAll(jsonFile)
	is.NoErr(err)
	gameHistory := &pb.GameHistory{}
	err = json.Unmarshal(bytes, gameHistory)
	is.NoErr(err)

	g, err := NewFromHistory(gameHistory, rules, len(gameHistory.Events))
	is.NoErr(err)
	is.Equal(g.ToCGP(false), "15/15/15/15/15/15/7DORMINE1/5IBEX6/15/15/15/15/15/15/15 EEHKNOQ/?DEMOOW 26/75 0 lex CSW19;")
}

func TestToCGPWithOppRack(t *testing.T) {
	is := is.New(t)

	rules, err := NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	is.NoErr(err)
	jsonFile, err := os.Open("../puzzles/testdata/phony_tiles_history.json")
	is.NoErr(err)
	defer jsonFile.Close()

	bytes, err := io.ReadAll(jsonFile)
	is.NoErr(err)
	gameHistory := &pb.GameHistory{}
	err = json.Unmarshal(bytes, gameHistory)
	is.NoErr(err)

	g, err := NewFromHistory(gameHistory, rules, 31)
	is.NoErr(err)
	is.Equal(g.ToCGP(true), "15/15/12L2/12O1V/12U1O/1L10I1T/KI2G1Q2B2ERE/E2FOGIE1R3AD/MUNI3WEANING1/B2A3E1PO2AH/1VINE2R2R3A/1I1C3S1OM3U/1THEN9L/1AAS10E/2J11R AEERSTT/YD 127/334 1 lex CSW19; ld english;")
	// try a phony with a blank.
	jsonFile, err = os.Open("../gcgio/testdata/josh2.json")
	is.NoErr(err)
	defer jsonFile.Close()

	bytes, err = io.ReadAll(jsonFile)
	is.NoErr(err)
	gameHistory = &pb.GameHistory{}
	err = json.Unmarshal(bytes, gameHistory)
	is.NoErr(err)
	gameHistory.ChallengeRule = pb.ChallengeRule_DOUBLE
	g, err = NewFromHistory(gameHistory, rules, 23)
	is.NoErr(err)
	// opp has blank in rack
	is.Equal(g.ToCGP(true), "15/9J5/5F3UT4/1SQUARER1TAD3/5I3EMO3/5ZEK2EW3/6MITT1N3/7DOWLY3/5OX1POI4/3ALBUGoS5/3HAO1U7/2CIG2L7/2O2HALON5/1DIETARY7/EINA11 CENNRRT/DEPONE? 316/224 1 lex CSW19;")

}
