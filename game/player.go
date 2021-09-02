package game

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/rs/zerolog/log"
)

type playerState struct {
	pb.PlayerInfo

	rack        *alphabet.Rack
	rackLetters string
	points      int
	bingos      int
	turns       int
}

func (p *playerState) resetScore() {
	p.points = 0
	p.bingos = 0
	p.turns = 0
}

func (p *playerState) throwRackIn(bag *alphabet.Bag) {
	log.Debug().Str("rack", p.rack.String()).Str("player", p.Nickname).
		Msg("throwing rack in")
	bag.PutBack(p.rack.TilesOn())
	p.rack.Set([]alphabet.MachineLetter{})
	p.rackLetters = ""
}

func (p *playerState) setRackTiles(tiles []alphabet.MachineLetter, alph *alphabet.Alphabet) {
	p.rack.Set(tiles)
	p.rackLetters = alphabet.MachineWord(tiles).UserVisible(alph)
}

func (p playerState) stateString(myturn bool) string {
	onturn := ""
	if myturn {
		onturn = "-> "
	}
	rackLetters := p.rackLetters
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
		p[idx].rackLetters = ""
	}
}

func (p playerStates) resetScore() {
	for idx := range p {
		p[idx].resetScore()
	}
}
