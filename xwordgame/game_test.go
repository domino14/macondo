package xwordgame

import (
	"testing"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/stretchr/testify/assert"
)

func TestGenBestStaticTurn(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america2018.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()

	game.players[0].rack = movegen.RackFromString("DRRIRDF", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	assert.Equal(t, move.MoveTypeExchange, game.movegen.Plays()[0].Action())
}

func TestGenBestStaticTurn2(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america2018.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()

	game.players[0].rack = movegen.RackFromString("COTTTV?", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	assert.Equal(t, move.MoveTypeExchange, game.movegen.Plays()[0].Action())
}

func TestGenBestStaticTurn3(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america2018.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()

	game.players[0].rack = movegen.RackFromString("INNRUVW", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	// assert.Equal(t, move.MoveTypeExchange, game.movegen.Plays()[0].Action())
}

func TestGenBestStaticTurn4(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america2018.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()
	// this rack has so much equity that the player might pass.
	game.players[0].rack = movegen.RackFromString("CDEERS?", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	assert.Equal(t, move.MoveTypeEndgameTiles, game.movegen.Plays()[0].Action())
}
