package automatic

import (
	"log"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/board"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
)

func TestGenBestStaticTurn(t *testing.T) {
	is := is.New(t)
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	is.NoErr(err)
	r := &GameRunner{}

	r.Init(gd)
	r.game.SetRackFor(0, alphabet.RackFromString("DRRIRDF", gd.GetAlphabet()))
	bestPlay := r.genBestStaticTurn(0)
	is.Equal(move.MoveTypeExchange, bestPlay.Action())
}

func TestGenBestStaticTurn2(t *testing.T) {
	is := is.New(t)
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	is.NoErr(err)
	r := &GameRunner{}

	r.Init(gd)
	r.game.SetRackFor(0, alphabet.RackFromString("COTTTV?", gd.GetAlphabet()))
	bestPlay := r.genBestStaticTurn(0)
	is.Equal(move.MoveTypeExchange, bestPlay.Action())
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
	is := is.New(t)
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	is.NoErr(err)
	r := &GameRunner{}

	r.Init(gd)
	// this rack has so much equity that the player might pass/exchange.
	r.game.SetRackFor(0, alphabet.RackFromString("CDEERS?", gd.GetAlphabet()))
	bestPlay := r.genBestStaticTurn(0)
	is.Equal(move.MoveTypePlay, bestPlay.Action())
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
	is := is.New(t)
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	is.NoErr(err)
	r := &GameRunner{}

	r.Init(gd)
	tilesInPlay := r.game.Board().SetToGame(gd.GetAlphabet(), board.VsMacondo1)
	r.game.Board().GenAllCrossSets(gd, r.game.Bag().LetterDistribution())
	r.game.Bag().RemoveTiles(tilesInPlay.OnBoard)
	r.game.SetRackFor(0, alphabet.RackFromString("APRS?", gd.GetAlphabet()))
	r.game.SetRackFor(1, alphabet.RackFromString("ENNR", gd.GetAlphabet()))
	is.Equal(r.game.Bag().TilesRemaining(), 0)
	bestPlay := r.genBestStaticTurn(0)
	log.Println(r.movegen.Plays())
	is.Equal("F10 .cARPS", bestPlay.ShortDescription())
}
