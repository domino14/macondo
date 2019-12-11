package mechanics

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

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
	for idx := range repr.Players {
		game.players[idx].Nickname = repr.Players[idx].Nickname
		game.players[idx].RealName = repr.Players[idx].RealName
		game.players[idx].PlayerNumber = repr.Players[idx].PlayerNumber
	}

	game.PlayGameToTurn(repr, turnnum)
	return game
}

func (g *XWordGame) playTurn(repr *GameRepr, turnnum int) []alphabet.MachineLetter {

	playedTiles := []alphabet.MachineLetter(nil)

	for evtIdx := range repr.Turns[turnnum] {

		m := genMove(repr.Turns[turnnum][evtIdx], g.alph)

		switch m.Action() {
		case move.MoveTypePlay:
			g.board.PlayMove(m, g.gaddag, g.bag)
			g.players[g.onturn].points += m.Score()
			// Add tiles to playedTilesList
			for _, t := range m.Tiles() {
				if t != alphabet.PlayedThroughMarker {
					// Note that if a blank is played, the blanked letter
					// is added to the played tiles (and not the blank itself)
					// The RemoveTiles function below handles this later.
					playedTiles = append(playedTiles, t)
				}
			}
		case move.MoveTypeChallengeBonus, move.MoveTypeEndgameTiles,
			move.MoveTypePhonyTilesReturned, move.MoveTypeLostTileScore,
			move.MoveTypeLostScoreOnTime:

			// The score should already have the proper sign at creation time.
			g.players[g.onturn].points += m.Score()
		case move.MoveTypeExchange, move.MoveTypePass:
			// Nothing.
		}

	}

	g.onturn = (g.onturn + 1) % len(g.players)
	g.turnnum++
	return playedTiles
}

func (g *XWordGame) PlayGameToTurn(repr *GameRepr, turnnum int) error {
	g.board.Clear()
	g.bag.Refill()
	g.players.resetScore()
	g.turnnum = 0
	g.onturn = 0
	playedTiles := []alphabet.MachineLetter(nil)
	if turnnum < 0 || turnnum > len(repr.Turns) {
		return fmt.Errorf("game has %v turns, you have chosen a turn outside the range",
			len(repr.Turns))
	}
	var t int
	for t = 0; t < turnnum; t++ {
		addlTiles := g.playTurn(repr, t)
		playedTiles = append(playedTiles, addlTiles...)
	}
	var err error
	var rack alphabet.MachineWord

	if t < len(repr.Turns) {
		rack, err = alphabet.ToMachineWord(repr.Turns[t][0].GetRack(), g.alph)
		if err != nil {
			log.Error().Err(err).Msg("")
			return err
		}
		g.players[g.onturn].rack.Set(rack)
		g.players[g.onturn].rackLetters = alphabet.MachineWord(rack).UserVisible(g.alph)
		g.players[g.onturn].rack.Set([]alphabet.MachineLetter(nil))
		g.players[(g.onturn+1)%2].rackLetters = ""
	}

	// Now update the bag.
	// XXX: the RemoveTiles function is not smart, and thus expensive
	// It should not be used in any sort of simulation.
	g.bag.RemoveTiles(playedTiles)

	// What is the rack of the player on turn now?
	g.bag.RemoveTiles(rack)
	return nil
}

func (g *XWordGame) ToDisplayText() string {
	bt := g.Board().ToDisplayText(g.alph)
	// We need to insert rack, player, bag strings into the above string.
	bts := strings.Split(bt, "\n")
	hpadding := 3
	vpadding := 1
	for p := 0; p < len(g.players); p++ {
		bts[p+vpadding] = bts[p+vpadding] + strings.Repeat(" ", hpadding) +
			g.players[p].stateString(g.onturn == p)
	}
	bag := g.bag.Peek()

	bts[vpadding+3] = bts[vpadding+3] + strings.Repeat(" ", hpadding) +
		fmt.Sprintf("Bag + unseen: (%d)", len(bag))
	vpadding = 6
	sort.Slice(bag, func(i, j int) bool {
		return bag[i] < bag[j]
	})

	bagDisp := []string{}
	cCtr := 0
	bagStr := ""
	for i := 0; i < len(bag); i++ {
		bagStr += string(bag[i].UserVisible(g.alph)) + " "
		cCtr++
		if cCtr == 15 {
			bagDisp = append(bagDisp, bagStr)
			bagStr = ""
			cCtr = 0
		}
	}
	if bagStr != "" {
		bagDisp = append(bagDisp, bagStr)
	}

	for p := vpadding; p < vpadding+len(bagDisp); p++ {
		bts[p] = bts[p] + strings.Repeat(" ", hpadding) + bagDisp[p-vpadding]
	}

	return strings.Join(bts, "\n")

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

// Generate a move from an event
func genMove(e Event, alph *alphabet.Alphabet) *move.Move {
	// Have to use type assertions here, but that's OK...
	var m *move.Move

	rack, err := alphabet.ToMachineWord(e.GetRack(), alph)
	if err != nil {
		log.Error().Err(err).Msg("")
		return nil
	}

	switch v := e.(type) {
	case *TilePlacementEvent:
		// Calculate tiles, leave, tilesPlayed
		tiles, err := alphabet.ToMachineWord(v.Play, alph)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil
		}

		leaveMW := leave(rack, tiles)
		m = move.NewScoringMove(v.Score, tiles, leaveMW, v.Direction == "v",
			len(rack)-len(leaveMW), alph, int(v.Row), int(v.Column), v.Position)

	case *PassingEvent:
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

	case *ScoreAdditionEvent:
		if v.Bonus > 0 {
			// Challenge bonus
			// hmm
			m = move.NewBonusScoreMove(move.MoveTypeChallengeBonus,
				rack, v.Bonus)

		} else {
			// Endgame points
			m = move.NewBonusScoreMove(move.MoveTypeEndgameTiles,
				rack, v.EndRackPoints)
		}
	case *ScoreSubtractionEvent:
		// This either happens for:
		// - game over after 6 passes
		// - phony came off the board
		// - international rules at the end of a game
		// - time penalty
		var mt move.MoveType
		// XXX: these are strings because we can't import from gcgio module
		// otherwise there's a circular import. That means I probably
		// screwed something up with the design.
		if v.Type == "lost_challenge" {
			mt = move.MoveTypePhonyTilesReturned
		} else if v.Type == "end_rack_penalty" {
			mt = move.MoveTypeLostTileScore
		} else if v.Type == "time_penalty" {
			mt = move.MoveTypeLostScoreOnTime
		}
		m = move.NewLostScoreMove(mt, rack, v.LostScore)
	default:
		log.Error().Msgf("Unhandled event %v", e)

	}
	return m
}
