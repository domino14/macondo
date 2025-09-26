package game_test

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

func TestChallengeVoid(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)
	alph := game.Alphabet()
	game.StartGame()
	game.SetPlayerOnTurn(0)
	game.SetRackFor(0, tilemapping.RackFromString("EFFISTW", alph))
	game.SetChallengeRule(pb.ChallengeRule_VOID)
	m := move.NewScoringMoveSimple(90, "8C", "SWIFFET", "", alph)
	_, err = game.ValidateMove(m)
	is.Equal(err.Error(), "the play contained invalid words: SWIFFET")
}

func TestChallengeDoubleIsLegal(t *testing.T) {
	is := is.New(t)
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	g, _ := game.NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetPlayerOnTurn(0)
	err = g.SetRackFor(0, tilemapping.RackFromString("IFFIEST", alph))
	is.NoErr(err)
	g.SyncRacksToHistory()
	g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	m := move.NewScoringMoveSimple(84, "8C", "IFFIEST", "", alph)
	_, err = g.ValidateMove(m)
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
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	for range 1000 {
		g, _ := game.NewGame(rules, players)
		alph := g.Alphabet()
		g.StartGame()
		g.SetBackupMode(game.InteractiveGameplayMode)
		g.SetStateStackLength(1)
		g.SetPlayerOnTurn(0)
		err = g.SetRackFor(0, tilemapping.RackFromString("IFFIEST", alph))
		is.NoErr(err)
		g.SyncRacksToHistory()
		g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
		m := move.NewScoringMoveSimple(84, "8C", "IFFITES", "", alph)
		_, err = g.ValidateMove(m)
		is.NoErr(err)
		err = g.PlayMove(m, true, 0)
		is.NoErr(err)
		legal, err := g.ChallengeEvent(0, 0)
		is.NoErr(err)
		is.True(!legal)
		is.Equal(len(g.History().Events), 2)
		is.Equal(g.History().Events[1].Type, pb.GameEvent_PHONY_TILES_RETURNED)
	}
}

func TestChallengeEndOfGamePlusFive(t *testing.T) {
	is := is.New(t)

	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../gcgio/testdata/some_isc_game.gcg")
	is.NoErr(err)
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

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

	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../gcgio/testdata/some_isc_game.gcg")
	is.NoErr(err)
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

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
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	g, _ := game.NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetPlayerOnTurn(0)
	err = g.SetRackFor(0, tilemapping.RackFromString("IFFIEST", alph))
	is.NoErr(err)
	g.SyncRacksToHistory()
	g.SetChallengeRule(pb.ChallengeRule_TRIPLE)
	m := move.NewScoringMoveSimple(84, "8C", "IFFIEST", "", alph)
	_, err = g.ValidateMove(m)
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
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	g, _ := game.NewGame(rules, players)
	alph := g.Alphabet()
	g.StartGame()
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	g.SetPlayerOnTurn(0)
	err = g.SetRackFor(0, tilemapping.RackFromString("IFFIEST", alph))
	is.NoErr(err)
	g.SyncRacksToHistory()
	g.SetChallengeRule(pb.ChallengeRule_TRIPLE)
	m := move.NewScoringMoveSimple(84, "8C", "IFFISET", "", alph)
	_, err = g.ValidateMove(m)
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

func TestChallengeRestoreRack(t *testing.T) {
	is := is.New(t)
	pos := "4REORIENT3/1Q3LI2NOO3/1U1BAL4V4/1ICE6AA3/1N8TE3/1O7WE4/1L5V1ED4/S6OFT5/H5IXIA5/OP1BHUT1F6/JORAM2AE6/I4A1D7/5WUZ7/5E1E7/COGNiSED7 AEIINST/ADGGKPS 248/385 0 lex NWL23;"
	g, err := cgp.ParseCGP(DefaultConfig, pos)
	g.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	alph := g.Alphabet()
	is.NoErr(err)
	m := move.NewScoringMoveSimple(74, "M3", "ISATINE", "", alph)
	_, err = g.ValidateMove(m)
	is.NoErr(err)
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)

	err = g.SetRacksForBoth([]*tilemapping.Rack{
		tilemapping.RackFromString("GNPRUY?", alph),
		tilemapping.RackFromString("MATLIKE", alph),
	})
	is.NoErr(err)
	g.SyncRacksToHistory()

	m2 := move.NewScoringMoveSimple(82, "L9", "MATLIKE", "", alph)
	_, err = g.ValidateMove(m2)
	is.NoErr(err)
	err = g.PlayMove(m2, true, 0)
	is.NoErr(err)

	legal, err := g.ChallengeEvent(0, 0)
	is.NoErr(err)
	is.True(!legal)
	is.Equal(g.RackLettersFor(0), "?GNPRUY")
	is.Equal(g.RackLettersFor(1), "AEIKLMT")
}
