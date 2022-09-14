package automatic

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/matryer/is"
)

func TestGenBestStaticTurn(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, &DefaultConfig)
	runner.StartGame()
	runner.game.SetRacksForBoth([]*alphabet.Rack{
		alphabet.RackFromString("DDFIRRR", runner.alphabet),
		alphabet.RackFromString("AEINRST", runner.alphabet),
	})
	is.Equal(runner.game.RackLettersFor(0), "DDFIRRR")
	is.Equal(runner.game.RackLettersFor(1), "AEINRST")
	is.Equal(runner.game.Bag().TilesRemaining(), 86)

	bestPlay := runner.genBestStaticTurn(0)
	is.Equal(move.MoveTypeExchange, bestPlay.Action())
}

func TestGenBestStaticTurn2(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, &DefaultConfig)
	runner.StartGame()
	runner.game.SetRacksForBoth([]*alphabet.Rack{
		alphabet.RackFromString("COTTTV?", runner.alphabet),
		alphabet.RackFromString("AEINRST", runner.alphabet),
	})
	is.Equal(runner.game.RackLettersFor(0), "COTTTV?")
	is.Equal(runner.game.RackLettersFor(1), "AEINRST")
	is.Equal(runner.game.Bag().TilesRemaining(), 86)

	bestPlay := runner.genBestStaticTurn(0)
	is.Equal(move.MoveTypeExchange, bestPlay.Action())
}

func TestGenBestStaticTurn3(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, &DefaultConfig)
	runner.StartGame()
	runner.game.SetRacksForBoth([]*alphabet.Rack{
		alphabet.RackFromString("INNRUVW", runner.alphabet),
		alphabet.RackFromString("AEINRST", runner.alphabet),
	})
	is.Equal(runner.game.RackLettersFor(0), "INNRUVW")
	is.Equal(runner.game.RackLettersFor(1), "AEINRST")
	is.Equal(runner.game.Bag().TilesRemaining(), 86)

	bestPlay := runner.genBestStaticTurn(0)
	is.Equal(move.MoveTypeExchange, bestPlay.Action())
}

func TestGenBestStaticTurn4(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, &DefaultConfig)
	runner.StartGame()
	// CDEERS? has so much equity that the player could pass/exchange
	// if an opponent's high-equity rack plays through D, E, R, or S.
	// genBestStaticTurn does not "simulate" that dynamic possibility.
	runner.game.SetRacksForBoth([]*alphabet.Rack{
		alphabet.RackFromString("CDEERS?", runner.alphabet),
		alphabet.RackFromString("AEINRST", runner.alphabet),
	})
	is.Equal(runner.game.RackLettersFor(0), "CDEERS?")
	is.Equal(runner.game.RackLettersFor(1), "AEINRST")
	is.Equal(runner.game.Bag().TilesRemaining(), 86)

	bestPlay := runner.genBestStaticTurn(0)
	is.Equal(move.MoveTypePlay, bestPlay.Action())
}

func TestGenBestStaticTurn5(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, &DefaultConfig)
	runner.StartGame()
	runner.game.SetRacksForBoth([]*alphabet.Rack{
		alphabet.RackFromString("ADNNRST", runner.alphabet),
		alphabet.RackFromString("AEINRST", runner.alphabet),
	})
	is.Equal(runner.game.RackLettersFor(0), "ADNNRST")
	is.Equal(runner.game.RackLettersFor(1), "AEINRST")
	is.Equal(runner.game.Bag().TilesRemaining(), 86)

	bestPlay := runner.genBestStaticTurn(0)
	is.Equal(" 8G DARN", bestPlay.ShortDescription())
	is.Equal(move.MoveTypePlay, bestPlay.Action())
}

func TestGenBestStaticTurn6(t *testing.T) {
	is := is.New(t)

	runner := NewGameRunner(nil, &DefaultConfig)
	runner.StartGame()
	runner.game.ThrowRacksIn()

	tilesInPlay := runner.game.Board().SetToGame(runner.alphabet, board.VsMacondo1)
	// Recalculate the board's anchors/cross-sets
	runner.game.RecalculateBoard()

	err := runner.game.Bag().RemoveTiles(tilesInPlay.OnBoard)
	is.NoErr(err)

	runner.game.SetRacksForBoth([]*alphabet.Rack{
		alphabet.RackFromString("APRS?", runner.alphabet),
		alphabet.RackFromString("ENNR", runner.alphabet),
	})
	is.Equal(runner.game.RackLettersFor(0), "APRS?")
	is.Equal(runner.game.RackLettersFor(1), "ENNR")
	is.Equal(runner.game.Bag().TilesRemaining(), 0)

	bestPlay := runner.genBestStaticTurn(0)
	is.Equal("F10 .cARPS", bestPlay.ShortDescription())
}
