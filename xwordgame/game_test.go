package xwordgame

import (
	"log"
	"testing"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/stretchr/testify/assert"
)

func TestGenBestStaticTurn(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()

	game.players[0].rack = movegen.RackFromString("DRRIRDF", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	// XXX: Fix this test. It doesn't work because our evaluator is not very good.
	// assert.Equal(t, move.MoveTypeExchange, game.movegen.Plays()[0].Action())
}

func TestGenBestStaticTurn2(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()

	game.players[0].rack = movegen.RackFromString("COTTTV?", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	log.Println(game.movegen.Plays())
	assert.Equal(t, move.MoveTypeExchange, game.movegen.Plays()[0].Action())
}
