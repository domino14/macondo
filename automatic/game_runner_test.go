package automatic

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
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	r := &GameRunner{}

	r.Init(gd)
	r.movegen.Reset()
	r.game.SetRackFor(0, alphabet.RackFromString("DRRIRDF", gd.GetAlphabet()))
	r.movegen.GenAll(r.game.RackFor(0))
	assert.Equal(t, move.MoveTypeExchange, r.movegen.Plays()[0].Action())
}

func TestGenBestStaticTurn2(t *testing.T) {
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	r := &GameRunner{}

	r.Init(gd)
	r.movegen.Reset()
	r.game.SetRackFor(0, alphabet.RackFromString("COTTTV?", gd.GetAlphabet()))
	r.movegen.GenAll(r.game.RackFor(0))
	assert.Equal(t, move.MoveTypeExchange, r.movegen.Plays()[0].Action())
}

// func TestGenBestStaticTurn3(t *testing.T) {
// 	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
// 	if err != nil {
// 		t.Errorf("expected err to be nil, got %v", err)
// 	}
// 	game := &XWordGame{}

// 	game.Init(gd)
// 	game.movegen.Reset()

// 	game.players[0].rack = alphabet.RackFromString("INNRUVW", game.alph)
// 	game.movegen.GenAll(game.players[0].rack)
// 	// assert.Equal(t, move.MoveTypeExchange, game.movegen.Plays()[0].Action())
// }

func TestGenBestStaticTurn4(t *testing.T) {
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	r := &GameRunner{}

	r.Init(gd)
	r.movegen.Reset()
	// this rack has so much equity that the player might pass/exchange.
	r.game.SetRackFor(0, alphabet.RackFromString("CDEERS?", gd.GetAlphabet()))
	r.movegen.GenAll(r.game.RackFor(0))
	assert.Equal(t, move.MoveTypePlay, r.movegen.Plays()[0].Action())
}

// func TestGenBestStaticTurn5(t *testing.T) {
// 	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
// 	if err != nil {
// 		t.Errorf("expected err to be nil, got %v", err)
// 	}
// 	game := &XWordGame{}

// 	game.Init(gd)
// 	game.movegen.Reset()
// 	game.players[0].rack = alphabet.RackFromString("ADNNRST", game.alph)
// 	game.movegen.GenAll(game.players[0].rack)
// 	log.Println(game.movegen.Plays())
// 	// It tries to play STRAND >:(
// 	// XXX: FIX
// 	// assert.NotEqual(t, 6, game.movegen.Plays()[0].TilesPlayed())
// }

func TestGenBestStaticTurn6(t *testing.T) {
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	r := &GameRunner{}

	r.Init(gd)
	r.movegen.SetBoardToGame(gd.GetAlphabet(), board.VsMacondo1)
	r.game.SetRackFor(0, alphabet.RackFromString("APRS?", gd.GetAlphabet()))

	r.movegen.SetOppRack(alphabet.RackFromString("ENNR", gd.GetAlphabet()))
	r.movegen.GenAll(r.game.RackFor(0))
	log.Println(r.movegen.Plays())
	assert.Equal(t, "F10 .cARPS", r.movegen.Plays()[0].ShortDescription())
}
