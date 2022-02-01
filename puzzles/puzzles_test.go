package puzzles

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/matryer/is"
)

var DefaultConfig = config.DefaultConfig()

func TestBestEquity(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}

	rules, err := game.NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, game.VarClassic)
	is.NoErr(err)

	g, _ := game.NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetPlayerOnTurn(0)

	g.SetRackFor(0, alphabet.RackFromString("AEIQRST", alph))
	m := move.NewScoringMoveSimple(22, "8G", "QI", "", alph)
	words, err := g.ValidateMove(m)
	is.NoErr(err)
	is.Equal(len(words), 1)
	g.PlayMove(m, true, 0)

	g.SetRackFor(1, alphabet.RackFromString("AFGKORT", alph))
	m = move.NewScoringMoveSimple(17, "9H", "FORK", "", alph)
	words, err = g.ValidateMove(m)
	is.NoErr(err)
	is.Equal(len(words), 2)
	g.PlayMove(m, true, 0)

	puzzles, err := CreatePuzzlesFromGame(&DefaultConfig, g)
	is.NoErr(err)
	is.Equal(len(puzzles), 1)
	is.Equal(puzzles[0].TurnNumber, 1)
	is.Equal(puzzles[0].Type, pb.PuzzleType_BEST_EQUITY)
	is.Equal(puzzles[0].Answer.WordsFormed, []string{"KOFTGARI"})
}
