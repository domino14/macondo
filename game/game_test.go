package game

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/move"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddagmaker"
	pb "github.com/domino14/macondo/rpc/api/proto"
	"github.com/matryer/is"
)

var DefaultConfig = &config.Config{
	StrategyParamsPath:        os.Getenv("STRATEGY_PARAMS_PATH"),
	LexiconPath:               os.Getenv("LEXICON_PATH"),
	DefaultLexicon:            "NWL18",
	DefaultLetterDistribution: "English",
}

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
	rules, err := NewGameRules(DefaultConfig, board.CrosswordGameBoard, "NWL18",
		"English")
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
	rules, _ := NewGameRules(DefaultConfig, board.CrosswordGameBoard, "NWL18",
		"English")
	game, _ := NewGame(rules, players)

	game.StartGame()
	// Some positive number.
	game.SetStateStackLength(5)

	alph := rules.gaddag.GetAlphabet()
	game.SetRackFor(0, alphabet.RackFromString("ACEOTV?", alph))

	m := move.NewScoringMoveSimple(20, "H7", "AVOCET", "?", alph)
	game.PlayMove(m, true, false)

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
