package xwordgame

import (
	"log"
	"testing"

	"github.com/domino14/macondo/board"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/stretchr/testify/assert"
)

func TestGenBestStaticTurn(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america2018.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()

	game.players[0].rack = alphabet.RackFromString("DRRIRDF", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	assert.Equal(t, move.MoveTypeExchange, game.movegen.Plays()[0].Action())
}

func TestGenBestStaticTurn2(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america2018.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()

	game.players[0].rack = alphabet.RackFromString("COTTTV?", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	assert.Equal(t, move.MoveTypeExchange, game.movegen.Plays()[0].Action())
}

func TestGenBestStaticTurn3(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america2018.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()

	game.players[0].rack = alphabet.RackFromString("INNRUVW", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	// assert.Equal(t, move.MoveTypeExchange, game.movegen.Plays()[0].Action())
}

func TestGenBestStaticTurn4(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america2018.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()
	// this rack has so much equity that the player might pass/exchange.
	game.players[0].rack = alphabet.RackFromString("CDEERS?", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	assert.Equal(t, move.MoveTypePlay, game.movegen.Plays()[0].Action())
}

func TestGenBestStaticTurn5(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america2018.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.Reset()
	game.players[0].rack = alphabet.RackFromString("ADNNRST", game.alph)
	game.movegen.GenAll(game.players[0].rack)
	log.Println(game.movegen.Plays())
	// It tries to play STRAND >:(
	// XXX: FIX
	// assert.NotEqual(t, 6, game.movegen.Plays()[0].TilesPlayed())
}

func TestGenBestStaticTurn6(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america2018.gaddag")
	game := &XWordGame{}

	game.Init(gd)
	game.movegen.SetBoardToGame(gd.GetAlphabet(), board.VsMacondo1)
	game.players[0].rack = alphabet.RackFromString("APRS?", game.alph)
	game.movegen.SetOppRack(alphabet.RackFromString("ENNR", game.alph))
	game.movegen.GenAll(game.players[0].rack)
	log.Println(game.movegen.Plays())
	assert.Equal(t, "F10 .cARPS", game.movegen.Plays()[0].ShortDescription())
}
