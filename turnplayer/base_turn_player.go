package turnplayer

import (
	"strings"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

// Basic game. Set racks, make moves

type BaseTurnPlayer struct {
	*game.Game
}

// BaseTurnPlayerFromRules is a good entry point
func BaseTurnPlayerFromRules(opts *GameOptions, players []*pb.PlayerInfo, rules *game.GameRules) (*BaseTurnPlayer, error) {
	g, err := game.NewGame(rules, players)
	if err != nil {
		return nil, err
	}

	g.StartGame()
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	g.SetChallengeRule(opts.ChallengeRule)
	ret := &BaseTurnPlayer{g}
	return ret, nil
}

func (p *BaseTurnPlayer) SetPlayerRack(playerid int, letters string) error {
	rack := tilemapping.RackFromString(letters, p.Alphabet())
	err := p.SetRackFor(playerid, rack)
	if err != nil {
		return err
	}
	return nil
}

func (p *BaseTurnPlayer) SetCurrentRack(letters string) error {
	return p.SetPlayerRack(p.PlayerOnTurn(), letters)
}

func (p *BaseTurnPlayer) NewPassMove(playerid int) (*move.Move, error) {
	rack := p.RackFor(playerid)
	m := move.NewPassMove(rack.TilesOn(), p.Alphabet())
	return m, nil
}

func (p *BaseTurnPlayer) NewChallengeMove(playerid int) (*move.Move, error) {
	rack := p.RackFor(playerid)
	m := move.NewChallengeMove(rack.TilesOn(), p.Alphabet())
	return m, nil
}

func (p *BaseTurnPlayer) NewExchangeMove(playerid int, letters string) (*move.Move, error) {
	alph := p.Alphabet()
	rack := p.RackFor(playerid)
	tiles, err := tilemapping.ToMachineWord(letters, alph)
	if err != nil {
		return nil, err
	}
	leaveMW, err := tilemapping.Leave(rack.TilesOn(), tiles, true)
	if err != nil {
		return nil, err
	}
	m := move.NewExchangeMove(tiles, leaveMW, alph)
	return m, nil
}

func (p *BaseTurnPlayer) NewPlacementMove(playerid int, coords string, word string, transpose bool) (*move.Move, error) {
	coords = strings.ToUpper(coords)
	rack := p.RackFor(playerid).String()
	return p.CreateAndScorePlacementMove(coords, word, rack, transpose)
}

func (p *BaseTurnPlayer) MoveFromEvent(evt *pb.GameEvent) (*move.Move, error) {
	return game.MoveFromEvent(evt, p.Alphabet(), p.Board())
}

func (p *BaseTurnPlayer) IsPlaying() bool {
	return p.Playing() != pb.PlayState_GAME_OVER
}

func (p *BaseTurnPlayer) SetGame(g *game.Game) {
	p.Game = g
}
