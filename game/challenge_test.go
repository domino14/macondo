package game_test

import (
	"testing"

	airunner "github.com/domino14/macondo/ai/runner"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"

	"github.com/matryer/is"
)

func TestChallengeVoid(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout, "NWL18",
		"English")
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)
	alph := game.Alphabet()
	game.StartGame()
	game.SetPlayerOnTurn(0)
	game.SetRackFor(0, alphabet.RackFromString("EFFISTW", alph))
	game.SetChallengeRule(pb.ChallengeRule_VOID)
	m := move.NewScoringMoveSimple(90, "8C", "SWIFFET", "", alph)
	_, err = game.ValidateMove(m)
	is.Equal(err.Error(), "the play contained illegal words: SWIFFET")
}

func TestChallengeDoubleIsLegal(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, _ := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout, "NWL18",
		"English")
	g, _ := game.NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetPlayerOnTurn(0)
	g.SetRackFor(0, alphabet.RackFromString("IFFIEST", alph))
	g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	m := move.NewScoringMoveSimple(84, "8C", "IFFIEST", "", alph)
	_, err := g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)
	legal, err := g.ChallengeEvent(0, 0)
	is.NoErr(err)
	is.True(legal)
	is.Equal(len(g.History().Events), 2)
	is.Equal(g.History().Events[1].Type, pb.GameEvent_UNSUCCESSFUL_CHALLENGE_TURN_LOSS)
}

func TestChallengeDoubleIsIllegal(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, _ := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout, "NWL18",
		"English")
	g, _ := game.NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	g.SetPlayerOnTurn(0)
	g.SetRackFor(0, alphabet.RackFromString("IFFIEST", alph))
	g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	m := move.NewScoringMoveSimple(84, "8C", "IFFITES", "", alph)
	_, err := g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)
	legal, err := g.ChallengeEvent(0, 0)
	is.NoErr(err)
	is.True(!legal)
	is.Equal(len(g.History().Events), 2)
	is.Equal(g.History().Events[1].Type, pb.GameEvent_PHONY_TILES_RETURNED)
}

func TestChallengeEndOfGamePlusFive(t *testing.T) {
	is := is.New(t)

	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../gcgio/testdata/some_isc_game.gcg")
	is.NoErr(err)
	rules, _ := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout, "NWL18",
		"English")

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	g.SetChallengeRule(pb.ChallengeRule_FIVE_POINT)
	err = g.PlayToTurn(21)
	is.NoErr(err)
	alph := g.Alphabet()
	m := move.NewScoringMoveSimple(22, "3K", "ABBE", "", alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)
	legal, err := g.ChallengeEvent(0, 0)
	is.NoErr(err)
	is.True(legal)
	is.Equal(g.Playing(), pb.PlayState_GAME_OVER)
	is.Equal(g.PointsForNick("arcadio"), 364)
	is.Equal(g.PointsForNick("úrsula"), 409)
}

func TestChallengeEndOfGamePhony(t *testing.T) {
	is := is.New(t)

	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../gcgio/testdata/some_isc_game.gcg")
	is.NoErr(err)
	rules, _ := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout, "NWL18",
		"English")

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	g.SetChallengeRule(pb.ChallengeRule_FIVE_POINT)
	err = g.PlayToTurn(21)
	is.NoErr(err)
	alph := g.Alphabet()
	m := move.NewScoringMoveSimple(22, "3K", "ABEB", "", alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)
	is.Equal(g.Playing(), pb.PlayState_WAITING_FOR_FINAL_PASS)
	legal, err := g.ChallengeEvent(0, 0)
	is.NoErr(err)
	is.True(!legal)
	is.Equal(g.Playing(), pb.PlayState_PLAYING)

	is.Equal(g.PointsForNick("arcadio"), 364)
	is.Equal(g.PointsForNick("úrsula"), 372)
	is.Equal(g.NickOnTurn(), "arcadio")
}

func TestChallengeTripleUnsuccessful(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, _ := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout, "NWL18",
		"English")
	g, _ := game.NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetPlayerOnTurn(0)
	g.SetRackFor(0, alphabet.RackFromString("IFFIEST", alph))
	g.SetChallengeRule(pb.ChallengeRule_TRIPLE)
	m := move.NewScoringMoveSimple(84, "8C", "IFFIEST", "", alph)
	_, err := g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)
	legal, err := g.ChallengeEvent(0, 0)
	is.NoErr(err)
	is.True(legal)
	is.Equal(len(g.History().Events), 1)
	is.Equal(g.History().PlayState, pb.PlayState_GAME_OVER)
	is.Equal(g.History().Winner, int32(0))
}

func TestChallengeTripleSuccessful(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, _ := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout, "NWL18",
		"English")
	g, _ := game.NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	g.SetPlayerOnTurn(0)
	g.SetRackFor(0, alphabet.RackFromString("IFFIEST", alph))
	g.SetChallengeRule(pb.ChallengeRule_TRIPLE)
	m := move.NewScoringMoveSimple(84, "8C", "IFFISET", "", alph)
	_, err := g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)
	legal, err := g.ChallengeEvent(0, 0)
	is.NoErr(err)
	is.True(!legal)
	is.Equal(len(g.History().Events), 2)
	is.Equal(g.History().PlayState, pb.PlayState_GAME_OVER)
	is.Equal(g.History().Winner, int32(1))
}
