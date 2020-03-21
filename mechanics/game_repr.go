package mechanics

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
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
	Turns       []Turn       `json:"turns"`
	Players     []PlayerInfo `json:"players"`
	Version     int          `json:"version"`
	OriginalGCG string       `json:"originalGCG"`
	// Based on lexica, we will determine an alphabet.
	Lexicon string `json:"lexicon,omitempty"`
}

// StateFromRepr takes in a game representation and a turn number, and
// outputs a full XWordGame state at that turn number. turnnum can be
// equal to any number from 0 to the full number of turns.
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
		game.players[idx].info = repr.Players[idx]
	}

	game.PlayGameToTurn(repr, turnnum)
	return game
}

func (g *XWordGame) playTurn(repr *GameRepr, turnnum int) []alphabet.MachineLetter {

	playedTiles := []alphabet.MachineLetter(nil)
	challengedOffPlay := false
	// Check for the special case where a player played a phony that was
	// challenged off. We don't want to process this at all.
	if len(repr.Turns[turnnum]) == 2 {
		if repr.Turns[turnnum][0].GetType() == RegMove &&
			repr.Turns[turnnum][1].GetType() == LostChallenge {
			challengedOffPlay = true
		}
	}
	if !challengedOffPlay {
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
	}
	g.onturn = (g.onturn + 1) % len(g.players)
	g.turnnum++
	return playedTiles
}

func (g *XWordGame) PlayGameToTurn(repr *GameRepr, turnnum int) error {
	if turnnum < 0 || turnnum > len(repr.Turns) {
		return fmt.Errorf("game has %v turns, you have chosen a turn outside the range",
			len(repr.Turns))
	}

	g.board.Clear()
	g.bag.Refill()
	g.players.resetScore()
	g.turnnum = 0
	g.onturn = 0
	g.playing = true
	playedTiles := []alphabet.MachineLetter(nil)
	var t int
	for t = 0; t < turnnum; t++ {
		addlTiles := g.playTurn(repr, t)
		playedTiles = append(playedTiles, addlTiles...)
		// log.Debug().Msgf("played turn %v (%v) and added tiles %v", t, repr.Turns[t],
		// 	alphabet.MachineWord(addlTiles).UserVisible(g.alph))
	}

	// Now update the bag.
	err := g.reconcileTiles(repr, playedTiles, t)
	if err != nil {
		return err
	}

	// Check if the game is over.
	log.Debug().Msg("Checking if game is over...")
	// Note that tiles remaining are also unseen tiles.
	if g.bag.TilesRemaining() <= 7 {
		// All players must have empty racks. This normally doesn't make sense
		// in real Scrabble, but the way a GCG works, it only tracks one person's
		// tiles at a time. For simplicity, we assume tiles are always in the bag
		// if unseen!
		allEmpty := true

		log.Debug().Msg("Bag is empty")
		for _, p := range g.players {
			log.Debug().Msgf("Player %v has %v tiles", p, p.rack.NumTiles())
			if p.rack.NumTiles() != 0 {
				allEmpty = false
				break
			}
		}
		if allEmpty {
			log.Debug().Msg("Game is over")
			g.playing = false
		}
	}
	return nil
}

func (g *XWordGame) reconcileTiles(repr *GameRepr, playedTiles []alphabet.MachineLetter, turn int) error {
	// Reconcile tiles in the bag.

	var err error
	log.Debug().Msgf("Removing tiles played: %v", playedTiles)
	err = g.bag.RemoveTiles(playedTiles)
	log.Debug().Msgf("Error was %v", err)
	if err != nil {
		return err
	}
	// Determine the latest rack available. Sometimes a game representation
	// ends before the game itself is over, so we need to make sure
	// we don't mistakenly give the other player all the remaining tiles, etc
	var rack alphabet.MachineWord
	notOnTurn := (g.onturn + 1) % 2

	if turn < len(repr.Turns) {
		// Find the rack of the player currently on turn; after playing
		// turns up to this point.
		rack, err = alphabet.ToMachineWord(repr.Turns[turn][0].GetRack(), g.alph)

		if err != nil {
			return err
		}
	}
	// Always empty the opponent's rack, so it doesn't display.
	g.players[notOnTurn].SetRack([]alphabet.MachineLetter{}, g.alph)
	log.Debug().Msgf("My rack is %v, removing it from bag", rack.UserVisible(g.alph))
	g.players[g.onturn].SetRack(rack, g.alph)

	return g.bag.RemoveTiles(rack)
}

func addText(lines []string, row int, hpad int, text string) {
	str := lines[row]
	str += strings.Repeat(" ", hpad)
	str += text
	lines[row] = str
}

// ToDisplayText turns the current state of the game into a displayable
// string. It takes in an additional game representation, which is used
// to display more in-depth turn information.
func (g *XWordGame) ToDisplayText(repr *GameRepr) string {
	bt := g.Board().ToDisplayText(g.alph)
	// We need to insert rack, player, bag strings into the above string.
	bts := strings.Split(bt, "\n")
	hpadding := 3
	vpadding := 1
	bagColCount := 20
	for p := 0; p < len(g.players); p++ {
		addText(bts, p+vpadding, hpadding,
			g.players[p].stateString(g.playing && g.onturn == p))
	}
	bag := g.bag.Peek()
	addText(bts, vpadding+3, hpadding, fmt.Sprintf("Bag + unseen: (%d)", len(bag)))

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
		if cCtr == bagColCount {
			bagDisp = append(bagDisp, bagStr)
			bagStr = ""
			cCtr = 0
		}
	}
	if bagStr != "" {
		bagDisp = append(bagDisp, bagStr)
	}

	for p := vpadding; p < vpadding+len(bagDisp); p++ {
		addText(bts, p, hpadding, bagDisp[p-vpadding])
	}

	addText(bts, 12, hpadding, fmt.Sprintf("Turn %d:", g.turnnum))

	vpadding = 13

	if g.turnnum-1 >= 0 {
		addText(bts, vpadding, hpadding, repr.Turns[g.turnnum-1].summary())
	}

	vpadding = 15

	if !g.playing {
		addText(bts, vpadding, hpadding, "Game is over.")
	}

	return strings.Join(bts, "\n")

}

// Calculate the leave from the rack and the made play.
func leave(rack alphabet.MachineWord, play alphabet.MachineWord) (alphabet.MachineWord, error) {
	rackmls := map[alphabet.MachineLetter]int{}
	for _, t := range rack {
		rackmls[t]++
	}
	for _, t := range play {
		if t == alphabet.PlayedThroughMarker {
			continue
		}
		if t.IsBlanked() {
			t = alphabet.BlankMachineLetter
		}
		if rackmls[t] != 0 {
			// It should never be 0 unless the GCG is malformed somehow.
			rackmls[t]--
		} else {
			return nil, fmt.Errorf("Tile in play but not in rack: %v %v",
				string(t.UserVisible(alphabet.EnglishAlphabet())), rackmls[t])
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
	return leave, nil
}

// Generate a move from an event
func genMove(e Event, alph *alphabet.Alphabet) *move.Move {
	// Have to use type assertions here, but that's OK...
	if e.GetMove() != nil {
		return e.GetMove()
	}
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

		leaveMW, err := leave(rack, tiles)
		if err != nil {
			log.Error().Err(err).Msg("")
			return nil
		}
		// log.Debug().Msgf("calculated leave %v from rack %v, tiles %v",
		// 	leaveMW.UserVisible(alph), rack.UserVisible(alph),
		// 	tiles.UserVisible(alph))
		m = move.NewScoringMove(v.Score, tiles, leaveMW, v.Direction == "v",
			len(rack)-len(leaveMW), alph, int(v.Row), int(v.Column), v.Position)

	case *PassingEvent:
		if len(v.Exchanged) > 0 {
			tiles, err := alphabet.ToMachineWord(v.Exchanged, alph)
			if err != nil {
				log.Error().Err(err).Msg("")
				return nil
			}
			leaveMW, err := leave(rack, tiles)
			if err != nil {
				log.Error().Err(err).Msg("")
				return nil
			}
			m = move.NewExchangeMove(tiles, leaveMW, alph)
		} else {
			m = move.NewPassMove(rack, alph)
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

		if v.Type == LostChallenge {
			mt = move.MoveTypePhonyTilesReturned
		} else if v.Type == EndRackPenalty {
			mt = move.MoveTypeLostTileScore
		} else if v.Type == TimePenalty {
			mt = move.MoveTypeLostScoreOnTime
		}
		m = move.NewLostScoreMove(mt, rack, v.LostScore)
	default:
		log.Error().Msgf("Unhandled event %v", e)

	}
	e.SetMove(m)
	return m
}

// AddTurnFromPlay creates a new Turn, and adds it at the turn ID. It
// additionally truncates all moves after this one.
func (r *GameRepr) AddTurnFromPlay(turnnum int, m *move.Move, nick string, cumul int, appendPlay bool) error {
	turnToAdd := []Event{}
	switch m.Action() {
	case move.MoveTypePlay:
		evt := &TilePlacementEvent{}
		evt.Nickname = nick
		evt.Rack = m.FullRack()
		evt.Position = m.BoardCoords()
		evt.Play = m.Tiles().UserVisible(m.Alphabet())
		evt.Score = m.Score()
		evt.Cumulative = cumul
		evt.Type = RegMove
		evt.CalculateCoordsFromPosition()
		evt.SetMove(m)
		turnToAdd = append(turnToAdd, evt)

	case move.MoveTypePass:

	case move.MoveTypeExchange:

	}
	if appendPlay {
		r.Turns = append(r.Turns, turnToAdd)
	} else {
		r.Turns[turnnum] = turnToAdd
		r.Turns = r.Turns[:turnnum+1]
	}
	log.Debug().Msgf("turns are now: len: %v, turns: %v", len(r.Turns), r.Turns)
	return nil
}

// PlayAndAppendScoringMove is a utility function to append a scoring
// move to the game history, as well as play it on the board.
// The caller can provide coordinates and the move; this function
// scores and keeps track of the board state properly, modifying the
// state appropriately.
func PlayAndAppendScoringMove(state *XWordGame, repr *GameRepr,
	coords, word string) error {

	playerid := state.PlayerOnTurn()
	nick := state.players[playerid].info.Nickname

	m, err := PlayScoringMove(state, coords, word)
	if err != nil {
		log.Error().Msgf("Trying to play move, err was %v", err)
		return err
	}

	cumul := state.PointsFor(playerid) + m.Score()
	// Pass in 0 as the turnnum; it gets ignored as we are appending the play
	// at the end.``
	return repr.AddTurnFromPlay(0, m, nick, cumul, true)
}

// PlayScoringMove plays a move on a board that is described by the
// coordinates and word only. It returns the move.
func PlayScoringMove(state *XWordGame, coords, word string) (*move.Move, error) {
	playerid := state.PlayerOnTurn()
	rack := state.RackFor(playerid).String()

	m, err := state.CreateAndScorePlacementMove(coords, word, rack)
	if err != nil {
		log.Error().Msgf("Trying to create and score move, err was %v", err)
		return nil, err
	}
	// Actually make the play on the board:
	state.PlayMove(m, false)
	return m, nil
}
