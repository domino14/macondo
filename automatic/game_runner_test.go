package automatic

import (
	"log"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/move"
)

func TestGenBestStaticTurn(t *testing.T) {
	is := is.New(t)
	runner := NewGameRunner(nil, &DefaultConfig)
	runner.StartGame()
	runner.game.SetRackFor(0, alphabet.RackFromString("DRRIRDF", runner.alphabet))
	bestPlay := runner.genBestStaticTurn(0)
	is.Equal(move.MoveTypeExchange, bestPlay.Action())
}

func TestGenBestStaticTurn2(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, &DefaultConfig)
	runner.StartGame()
	runner.game.SetRackFor(0, alphabet.RackFromString("COTTTV?", runner.alphabet))
	bestPlay := runner.genBestStaticTurn(0)
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
	runner := NewGameRunner(nil, &DefaultConfig)
	runner.StartGame()
	// this rack has so much equity that the player might pass/exchange.
	runner.game.SetRackFor(0, alphabet.RackFromString("CDEERS?", runner.alphabet))
	bestPlay := runner.genBestStaticTurn(0)
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
	runner := NewGameRunner(nil, &DefaultConfig)
	runner.StartGame()
	runner.game.ThrowRacksIn()

	tilesInPlay := runner.game.Board().SetToGame(runner.alphabet, board.VsMacondo1)
	bd := runner.game.Board()
	cross_set.GenAllCrossSets(bd, runner.gaddag, runner.game.Bag().LetterDistribution())

	err := runner.game.Bag().RemoveTiles(tilesInPlay.OnBoard)
	is.NoErr(err)

	runner.game.SetRackFor(0, alphabet.RackFromString("APRS?", runner.alphabet))
	runner.game.SetRackFor(1, alphabet.RackFromString("ENNR", runner.alphabet))

	is.Equal(runner.game.RackLettersFor(0), "APRS?")
	is.Equal(runner.game.RackLettersFor(1), "ENNR")

	is.Equal(runner.game.Bag().TilesRemaining(), 0)
	bestPlay := runner.genBestStaticTurn(0)
	log.Println(runner.movegen.Plays())
	is.Equal("F10 .cARPS", bestPlay.ShortDescription())
}
