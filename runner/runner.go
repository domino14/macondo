package runner

import (
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

// Basic game. Set racks, make moves

type GameRunner struct {
	game.Game
}

// NewGameRunnerFromRules is a good entry point
func NewGameRunnerFromRules(opts *GameOptions, players []*pb.PlayerInfo, rules *game.GameRules) (*GameRunner, error) {
	g, err := game.NewGame(rules, players)
	if err != nil {
		return nil, err
	}
	if opts.FirstIsAssigned {
		g.SetNextFirst(opts.GoesFirst)
	} else {
		// game determines it.
		g.SetNextFirst(-1)
	}
	g.StartGame()
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	g.SetChallengeRule(opts.ChallengeRule)
	ret := &GameRunner{*g}
	return ret, nil
}

func (g *GameRunner) SetPlayerRack(playerid int, letters string) error {
	rack := alphabet.RackFromString(letters, g.Alphabet())
	return g.SetRackFor(playerid, rack)
}

func (g *GameRunner) SetCurrentRack(letters string) error {
	return g.SetPlayerRack(g.PlayerOnTurn(), letters)
}

func (g *GameRunner) NewPassMove(playerid int) (*move.Move, error) {
	rack := g.RackFor(playerid)
	m := move.NewPassMove(rack.TilesOn(), g.Alphabet())
	return m, nil
}

func (g *GameRunner) NewChallengeMove(playerid int) (*move.Move, error) {
	rack := g.RackFor(playerid)
	m := move.NewChallengeMove(rack.TilesOn(), g.Alphabet())
	return m, nil
}

func (g *GameRunner) NewExchangeMove(playerid int, letters string) (*move.Move, error) {
	alph := g.Alphabet()
	rack := g.RackFor(playerid)
	tiles, err := alphabet.ToMachineWord(letters, alph)
	if err != nil {
		return nil, err
	}
	leaveMW, err := game.Leave(rack.TilesOn(), tiles)
	if err != nil {
		return nil, err
	}
	m := move.NewExchangeMove(tiles, leaveMW, alph)
	return m, nil
}

func (g *GameRunner) NewPlacementMove(playerid int, coords string, word string) (*move.Move, error) {
	coords = strings.ToUpper(coords)
	rack := g.RackFor(playerid).String()
	return g.CreateAndScorePlacementMove(coords, word, rack)
}

func (g *GameRunner) MoveFromEvent(evt *pb.GameEvent) (*move.Move, error) {
	return game.MoveFromEvent(evt, g.Alphabet(), g.Board())
}

func (g *GameRunner) IsPlaying() bool {
	return g.Playing() == pb.PlayState_PLAYING
}
