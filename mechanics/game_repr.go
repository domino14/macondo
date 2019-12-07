package mechanics

import (
	"os"
	"path/filepath"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/rs/zerolog/log"
)

var LexiconPath = os.Getenv("LEXICON_PATH")

// XXX: Make a config class that contains all these values.
const (
	LeaveFile = "leave_values_112719.idx.gz"
)

// A GameRepr is a user- and machine-friendly representation of a full
// Crossword Game. This differs from an XWordGame in that the latter is
// more of an instantaneous state of the game at any given time. It is
// therefore possible to determine the state from any turn in a GameRepr.
type GameRepr struct {
	Turns       []Turn    `json:"turns"`
	Players     []*Player `json:"players"`
	Version     int       `json:"version"`
	OriginalGCG string    `json:"originalGCG"`
	// Based on lexica, we will determine an alphabet.
	Lexicon string `json:"lexicon,omitempty"`
}

// StateFromRepr takes in a game representation and a turn number, and
// outputs a full XWordGame state at that turn number. turnnum can be
// equal to any number from -1 to the full number of turns minus 1.
// When -1, it will default to the end of the game (the last turn).
// Turns start at 0 for the purposes of this API.
func StateFromRepr(repr *GameRepr, defaultLexicon string, turnnum int) *XWordGame {
	game := &XWordGame{}

	gdFilename := filepath.Join(LexiconPath, "gaddag", repr.Lexicon+".gaddag")
	// XXX: Later make this lexicon-dependent.
	dist := alphabet.EnglishLetterDistribution()

	gd, err := gaddag.LoadGaddag(gdFilename)
	if err != nil {
		log.Warn().Msgf("The loaded file contained no lexicon information. "+
			"Defaulting to %v (use loadlex to override)", defaultLexicon)
		gd, err = gaddag.LoadGaddag(filepath.Join(LexiconPath, "gaddag",
			defaultLexicon+".gaddag"))
		if err != nil {
			log.Fatal().Msgf("Could not load default lexicon; please be sure " +
				"the LEXICON_PATH environment variable is correct.")
		}
	}
	game.Init(gd, dist)
	// strategy := strategy.NewExhaustiveLeaveStrategy(game.Bag(), gd.LexiconName(),
	// 	gd.GetAlphabet(), LeaveFile)
	game.players = players(repr.Players)
	if turnnum == -1 {
		turnnum = len(repr.Turns)
	}
	game.PlayGameToTurn(repr, turnnum)
	return game
}

func (g *XWordGame) PlayGameToTurn(repr *GameRepr, turnnum int) {
	for t := 0; t < turnnum; t++ {

	}
}
