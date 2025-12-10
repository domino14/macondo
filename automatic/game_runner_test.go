package automatic

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

func TestGenBestStaticTurn(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, DefaultConfig)
	runner.StartGame(0)
	runner.game.SetRackFor(0, tilemapping.RackFromString("DRRIRDF", runner.alphabet))
	bestPlay := runner.genBestStaticTurn(0)
	is.Equal(move.MoveTypeExchange, bestPlay.Action())
}

func TestGenBestStaticTurn2(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, DefaultConfig)
	runner.StartGame(0)
	runner.game.SetRackFor(0, tilemapping.RackFromString("COTTTV?", runner.alphabet))
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

// 	game.players[0].rack = tilemapping.RackFromString("INNRUVW", game.alph)
// 	game.movegen.GenAll(game.players[0].rack)
// 	// assert.Equal(t, move.MoveTypeExchange, game.movegen.Plays()[0].Action())
// }

func TestGenBestStaticTurn4(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, DefaultConfig)
	runner.StartGame(0)
	// this rack has so much equity that the player might pass/exchange.
	runner.game.SetRackFor(0, tilemapping.RackFromString("CDEERS?", runner.alphabet))
	bestPlay := runner.genBestStaticTurn(0)
	is.Equal(move.MoveTypePlay, bestPlay.Action())
}

func TestGenBestStaticTurn6(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, DefaultConfig)
	runner.StartGame(0)
	runner.game.ThrowRacksIn()

	tilesInPlay := runner.game.Board().SetToGame(runner.alphabet, board.VsMacondo1)
	bd := runner.game.Board()
	cross_set.GenAllCrossSets(bd, runner.gaddag, runner.game.Bag().LetterDistribution())

	err := runner.game.Bag().RemoveTiles(tilesInPlay.OnBoard)
	is.NoErr(err)

	runner.game.SetRackFor(0, tilemapping.RackFromString("APRS?", runner.alphabet))
	runner.game.SetRackFor(1, tilemapping.RackFromString("ENNR", runner.alphabet))

	is.Equal(runner.game.RackLettersFor(0), "?APRS")
	is.Equal(runner.game.RackLettersFor(1), "ENNR")

	is.Equal(runner.game.Bag().TilesRemaining(), 0)
	bestPlay := runner.genBestStaticTurn(0)
	is.Equal("F10 .cARPS", bestPlay.ShortDescription())
}

func TestStochasticTurn(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, DefaultConfig)
	runner.StartGame(0)
	// this rack has so much equity that the player might pass/exchange.
	runner.game.SetRackFor(0, tilemapping.RackFromString("ADFGHIJ", runner.alphabet))
	runner.aiplayers[0].MoveGenerator().(*movegen.GordonGenerator).SetRecordNTopPlays(2)
	runner.aiplayers[0].MoveGenerator().(*movegen.GordonGenerator).SetGame(runner.game)
	plays := map[string]int{}
	for i := 0; i < 10000; i++ {
		bestPlay := runner.genStochasticStaticTurn(0)
		plays[bestPlay.ShortDescription()]++
	}
	// JIHAD should get picked around 59% of the time.
	is.True(plays[" 8D JIHAD"] > 5500 && plays[" 8D JIHAD"] < 6500)
	is.True(plays[" 8D HADJI"] > 3500 && plays[" 8D HADJI"] < 4500)
}

func TestBagSeeding(t *testing.T) {
	is := is.New(t)

	// Create two games with the same seed
	seed1 := [32]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}

	runner1 := NewGameRunner(nil, DefaultConfig)
	runner1.StartGameWithSeed(0, seed1)

	runner2 := NewGameRunner(nil, DefaultConfig)
	runner2.StartGameWithSeed(0, seed1)

	// They should have identical racks
	rack1 := runner1.game.RackLettersFor(0)
	rack2 := runner2.game.RackLettersFor(0)
	t.Logf("Runner1 P0: %s, Runner2 P0: %s", rack1, rack2)
	is.Equal(rack1, rack2)

	rack1 = runner1.game.RackLettersFor(1)
	rack2 = runner2.game.RackLettersFor(1)
	t.Logf("Runner1 P1: %s, Runner2 P1: %s", rack1, rack2)
	is.Equal(rack1, rack2)

	// Create a game with a different seed
	seed2 := [32]byte{32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
		16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}

	runner3 := NewGameRunner(nil, DefaultConfig)
	runner3.StartGameWithSeed(0, seed2)

	// It should have different racks
	rack3 := runner3.game.RackLettersFor(0)
	t.Logf("Runner3 P0: %s (should differ from %s)", rack3, rack1)
	is.True(rack1 != rack3)
}
