package game

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/move"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddagmaker"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/matryer/is"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	for _, lex := range []string{"NWL18"} {
		gdgPath := filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag")
		if _, err := os.Stat(gdgPath); os.IsNotExist(err) {
			gaddagmaker.GenerateGaddag(filepath.Join(DefaultConfig.LexiconPath, lex+".txt"), true, true)
			err = os.Rename("out.gaddag", gdgPath)
			if err != nil {
				panic(err)
			}
		}
	}
	os.Exit(m.Run())
}

func TestNewGame(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
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
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
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
	game.SetRackFor(0, alphabet.RackFromString("ACEOTV?", alph))

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
	is.Equal(game.players[0].rackLetters, "ACEOTV?")
}

func TestValidate(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}

	rules, err := NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	is.NoErr(err)

	g, _ := NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetPlayerOnTurn(0)
	g.SetRackFor(0, alphabet.RackFromString("HIS", alph))
	g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	m := move.NewScoringMoveSimple(12, "H7", "HIS", "", alph)
	words, err := g.ValidateMove(m)
	is.NoErr(err)
	is.Equal(len(words), 1)
	g.PlayMove(m, true, 0)
	is.Equal(g.history.Events[len(g.history.Events)-1].WordsFormed,
		[]string{"HIS"})
	g.SetRackFor(1, alphabet.RackFromString("OIK", alph))
	m = move.NewScoringMoveSimple(13, "G8", "OIK", "", alph)
	words, err = g.ValidateMove(m)
	is.NoErr(err)
	is.Equal(len(words), 3)
	g.PlayMove(m, true, 0)
	is.Equal(g.history.Events[len(g.history.Events)-1].WordsFormed,
		[]string{"OIK", "OI", "IS"})

	g.SetRackFor(0, alphabet.RackFromString("ADITT", alph))
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
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	is.NoErr(err)
	jsonFile, err := os.Open("./testdata/history1.json")
	is.NoErr(err)
	defer jsonFile.Close()

	bytes, err := ioutil.ReadAll(jsonFile)
	is.NoErr(err)
	gameHistory := &pb.GameHistory{}
	err = json.Unmarshal(bytes, gameHistory)
	is.NoErr(err)

	g, err := NewFromHistory(gameHistory, rules, len(gameHistory.Events))
	is.NoErr(err)

	is.Equal(g.RackFor(0).TilesOn().UserVisible(alphabet.EnglishAlphabet()),
		"EEHKNOQ")
	is.Equal(g.RackFor(1).TilesOn().UserVisible(alphabet.EnglishAlphabet()),
		"DEMOOW?")
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
	is.Equal(g.RackFor(1).TilesOn().UserVisible(alphabet.EnglishAlphabet()),
		"DEIMNOR")
	// p0 still has their rack
	is.Equal(g.RackFor(0).TilesOn().UserVisible(alphabet.EnglishAlphabet()),
		"EEHKNOQ")
}
