package mechanics

import (
	"os"
	"path/filepath"

	"github.com/domino14/macondo/move"

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
	// for t := 0; t < turnnum; t++ {
	// 	m := genMove(repr.Turns[t], g.alph)

	// }
}

// Calculate the leave from the rack and the made play.
func leave(rack alphabet.MachineWord, play alphabet.MachineWord) alphabet.MachineWord {
	rackmls := map[alphabet.MachineLetter]int{}
	for _, t := range rack {
		rackmls[t]++
	}
	for _, t := range play {
		if rackmls[t] != 0 {
			// It should never be 0 unless the GCG is malformed somehow.
			rackmls[t]--
		}
	}
	leave := []alphabet.MachineLetter{}
	for k, v := range rackmls {
		if v > 0 {
			for i := 0; i < v; i++ {
				leave = append(leave, k)
			}
		}
	}
	return leave
}

// Generate a move from a turn
func genMove(t Turn, alph *alphabet.Alphabet) *move.Move {
	// Have to use type assertions here, but that's OK...
	var m *move.Move
	switch v := t.(type) {
	case *TilePlacementTurn:
		// Calculate tiles, leave, tilesPlayed
		tiles, err := alphabet.ToMachineWord(v.Play, alph)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil
		}
		rack, err := alphabet.ToMachineWord(v.Rack, alph)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil
		}
		leaveMW := leave(rack, tiles)
		m = move.NewScoringMove(v.Score, tiles, leaveMW, v.Direction == "v",
			len(rack)-len(leaveMW), alph, int(v.Row), int(v.Column), v.Position)

	case *PassingTurn:
		rack, err := alphabet.ToMachineWord(v.Rack, alph)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil
		}

		if len(v.Exchanged) > 0 {
			tiles, err := alphabet.ToMachineWord(v.Exchanged, alph)
			if err != nil {
				log.Error().Err(err).Msg("")
				return nil
			}
			leaveMW := leave(rack, tiles)
			m = move.NewExchangeMove(tiles, leaveMW, alph)
		} else {
			m = move.NewPassMove(rack)
		}

	case *ScoreAdditionTurn:

	case *ScoreSubtractionTurn:

	}
	return m
}
