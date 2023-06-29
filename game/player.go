package game

import (
	"fmt"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/tilemapping"
	"github.com/rs/zerolog/log"
)

type playerState struct {
	pb.PlayerInfo

	rack   *tilemapping.Rack
	points int
	bingos int
	turns  int

	// to minimize allocs:
	placeholderRack []tilemapping.MachineLetter
}

func newPlayerState(nickname, userid, realname string) *playerState {
	return &playerState{
		PlayerInfo: pb.PlayerInfo{
			Nickname: nickname,
			UserId:   userid,
			RealName: realname,
		},
		placeholderRack: make([]tilemapping.MachineLetter, RackTileLimit),
	}
}

func (p *playerState) resetScore() {
	p.points = 0
	p.bingos = 0
	p.turns = 0
}

func (p *playerState) throwRackIn(bag *tilemapping.Bag) {
	log.Trace().Str("rack", p.rack.String()).Str("player", p.Nickname).
		Msg("throwing rack in")
	bag.PutBack(p.rack.TilesOn())
	p.rack.Set([]tilemapping.MachineLetter{})
}

func (p *playerState) setRackTiles(tiles []tilemapping.MachineLetter, alph *tilemapping.TileMapping) {
	p.rack.Set(tiles)
}

func (p *playerState) rackLetters() string {
	return p.rack.String()
}

func (p *playerState) stateString(myturn bool) string {
	onturn := ""
	if myturn {
		onturn = "-> "
	}
	rackLetters := p.rackLetters()
	if !myturn {
		// Don't show rack letters.
		rackLetters = ""
	}
	return fmt.Sprintf("%4v%20v%9v %4v", onturn, p.Nickname, rackLetters, p.points)
}

type playerStates []*playerState

func (p playerStates) resetRacks() {
	for idx := range p {
		p[idx].rack.Clear()
	}
}

func (p playerStates) resetScore() {
	for idx := range p {
		p[idx].resetScore()
	}
}
