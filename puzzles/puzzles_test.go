package puzzles

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/matryer/is"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	for _, lex := range []string{"CSW19"} {
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

func TestBestEquity(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}

	rules, err := game.NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	g, _ := game.NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetPlayerOnTurn(0)

	g.SetRackFor(0, alphabet.RackFromString("AEIQRST", alph))
	m := move.NewScoringMoveSimple(22, "8G", "QI", "AERST", alph)
	words, err := g.ValidateMove(m)
	is.NoErr(err)
	is.Equal(len(words), 1)
	g.PlayMove(m, true, 0)

	g.SetRackFor(1, alphabet.RackFromString("AFGKORT", alph))
	m = move.NewScoringMoveSimple(17, "9H", "FORK", "AGT", alph)
	words, err = g.ValidateMove(m)
	is.NoErr(err)
	is.Equal(len(words), 2)
	g.PlayMove(m, true, 0)

	puzzles, err := CreatePuzzlesFromGame(&DefaultConfig, g)
	is.NoErr(err)
	is.Equal(len(puzzles), 1)
	is.Equal(puzzles[0].TurnNumber, int32(1))
	is.Equal(puzzles[0].GameId, g.Uid())
	// a game move doesn't have WordsFormed, so need to look at the actual
	// tiles played:
	is.Equal(puzzles[0].Answer.PlayedTiles, "KOFTGAR.")
	is.Equal(puzzles[0].Answer.Position, "H1")
	is.Equal(len(puzzles[0].Tags), 1)
	is.Equal(puzzles[0].Tags[0], pb.PuzzleTag_EQUITY)
}
