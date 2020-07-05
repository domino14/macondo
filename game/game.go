// Package game encapsulates the main mechanics for a Crossword Game. It
// interacts heavily with the protobuf data structures.
package game

import (
	crypto_rand "crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"math/rand"
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/lithammer/shortuuid"
	"github.com/rs/zerolog/log"
)

const (
	//IdentificationAuthority is the authority that gives out game IDs
	IdentificationAuthority = "org.aerolith"

	MacondoCreation = "Created with Macondo"

	ExchangeLimit = 7
	RackTileLimit = 7
)

// RuleDefiner is an interface that is used for passing a set of rules
// to a game.
type RuleDefiner interface {
	Gaddag() *gaddag.SimpleGaddag
	Board() *board.GameBoard
	LetterDistribution() *alphabet.LetterDistribution

	LoadRule(lexiconName, letterDistributionName string) error
}

func seededRandSource() (int64, *rand.Rand) {
	var b [8]byte
	_, err := crypto_rand.Read(b[:])
	if err != nil {
		panic("cannot seed math/rand package with cryptographically secure random number generator")
	}

	randSeed := int64(binary.LittleEndian.Uint64(b[:]))
	randSource := rand.New(rand.NewSource(randSeed))

	return randSeed, randSource
}

// Game is the actual internal game structure that controls the entire
// business logic of the game; drawing, making moves, etc. The two
// structures above are basically data entities.
// Note: a Game doesn't care how it is played. It is just rules for gameplay.
// AI players, human players, etc will play a game outside of the scope of
// this module.
type Game struct {
	gaddag *gaddag.SimpleGaddag
	alph   *alphabet.Alphabet
	// board and bag will contain the latest (current) versions of these.
	board              *board.GameBoard
	letterDistribution *alphabet.LetterDistribution
	bag                *alphabet.Bag

	playing pb.PlayState

	randSeed   int64
	randSource *rand.Rand

	wentfirst      int
	scorelessTurns int
	onturn         int
	turnnum        int
	players        playerStates
	// history has a history of all the moves in this game. Note that
	// history only gets written to when someone plays a move that is NOT
	// backed up.
	history *pb.GameHistory
	// lastWordsFormed also does not need to be backed up, it only gets written
	// to when the history is written to. See comment above.
	lastWordsFormed []alphabet.MachineWord
	backupMode      BackupMode

	stateStack []*stateBackup
	stackPtr   int
}

// CalculateCoordsFromStringPosition turns a "position" on the board such as
// H7 and turns it into a numeric row, col, and direction.
func CalculateCoordsFromStringPosition(evt *pb.GameEvent) {
	// Note that this evt.Position has nothing to do with the type Position
	// we are defining in this package.
	row, col, vertical := move.FromBoardGameCoords(evt.Position)
	if vertical {
		evt.Direction = pb.GameEvent_VERTICAL
	} else {
		evt.Direction = pb.GameEvent_HORIZONTAL
	}
	evt.Row = int32(row)
	evt.Column = int32(col)
}

func newHistory(players playerStates, flipfirst bool) *pb.GameHistory {
	his := &pb.GameHistory{}

	playerInfo := make([]*pb.PlayerInfo, len(players))

	for i := 0; i < len(players); i++ {
		playerInfo[i] = &pb.PlayerInfo{Nickname: players[i].Nickname,
			RealName: players[i].RealName, UserId: players[i].UserId}
	}
	his.Players = playerInfo
	his.IdAuth = IdentificationAuthority
	his.Uid = shortuuid.New()
	his.Description = MacondoCreation
	his.Events = []*pb.GameEvent{}
	his.SecondWentFirst = flipfirst
	his.LastKnownRacks = []string{"", ""}
	return his
}

// NewGame is how one instantiates a brand new game.
func NewGame(rules RuleDefiner, playerinfo []*pb.PlayerInfo) (*Game, error) {
	game := &Game{}
	game.gaddag = rules.Gaddag()
	game.alph = game.gaddag.GetAlphabet()
	game.letterDistribution = rules.LetterDistribution()
	game.backupMode = NoBackup

	game.board = rules.Board().Copy()

	game.players = make([]*playerState, len(playerinfo))
	for idx, p := range playerinfo {
		game.players[idx] = &playerState{
			PlayerInfo: pb.PlayerInfo{
				Nickname: p.Nickname,
				UserId:   p.UserId,
				RealName: p.RealName},
		}
	}

	return game, nil
}

// NewFromHistory instantiates a Game from a history, and sets the current
// turn to the passed in turnnum. It assumes the rules contains the current
// lexicon in history, if any!
func NewFromHistory(history *pb.GameHistory, rules RuleDefiner, turnnum int) (*Game, error) {
	game, err := NewGame(rules, history.Players)
	if err != nil {
		return nil, err
	}
	game.history = history
	if history.Uid == "" {
		history.Uid = shortuuid.New()
		history.IdAuth = IdentificationAuthority
	}
	if history.Description == "" {
		history.Description = MacondoCreation
	}
	if history.LastKnownRacks == nil {
		history.LastKnownRacks = []string{"", ""}
	}

	// Initialize the bag and player rack structures to avoid panics.
	game.randSeed, game.randSource = seededRandSource()
	log.Debug().Msgf("History - Random seed for this game was %v", game.randSeed)
	game.bag = game.letterDistribution.MakeBag(game.randSource)
	for i := 0; i < game.NumPlayers(); i++ {
		game.players[i].rack = alphabet.NewRack(game.alph)
	}
	// Then play to the passed-in turn.
	err = game.PlayToTurn(turnnum)
	if err != nil {
		return nil, err
	}
	return game, nil
}

func (g *Game) SetNewRules(rules RuleDefiner) error {
	g.gaddag = rules.Gaddag()
	g.alph = g.gaddag.GetAlphabet()
	g.letterDistribution = rules.LetterDistribution()
	return nil
}

// StartGame seeds the random source anew, and starts a game, dealing out tiles
// to both players.
func (g *Game) StartGame() {
	g.Board().Clear()
	g.randSeed, g.randSource = seededRandSource()
	log.Debug().Msgf("Random seed for this game was %v", g.randSeed)
	g.bag = g.letterDistribution.MakeBag(g.randSource)

	goesfirst := g.randSource.Intn(2)
	g.history = newHistory(g.players, goesfirst == 1)
	// Deal out tiles
	for i := 0; i < g.NumPlayers(); i++ {
		tiles, err := g.bag.Draw(7)
		if err != nil {
			panic(err)
		}
		g.players[i].rack = alphabet.NewRack(g.alph)
		g.players[i].setRackTiles(tiles, g.alph)
		g.players[i].resetScore()
	}
	g.history.LastKnownRacks = []string{
		g.RackLettersFor(0), g.RackLettersFor(1),
	}
	g.playing = pb.PlayState_PLAYING
	g.history.PlayState = g.playing
	g.turnnum = 0
	g.onturn = goesfirst
	g.wentfirst = goesfirst
}

// ValidateMove validates the given move. It is meant to be used to validate
// user input games (perhaps from live play or GCGs). It does not check the
// validity of the words formed (unless the challenge rule is VOID),
// but it validates that the rules of the game are followed.
// It returns an array of `alphabet.MachineWord`s formed, or an error if
// the play is not game legal.
func (g *Game) ValidateMove(m *move.Move) ([]alphabet.MachineWord, error) {
	if g.playing == pb.PlayState_GAME_OVER {
		return nil, errors.New("cannot play a move on a game that is over")
	}
	if m.Action() == move.MoveTypeExchange {
		if g.playing == pb.PlayState_WAITING_FOR_FINAL_PASS {
			return nil, errors.New("you can only pass or challenge")
		}
		if g.bag.TilesRemaining() < ExchangeLimit {
			return nil, fmt.Errorf("not allowed to exchange with fewer than %d tiles in the bag",
				ExchangeLimit)
		}
		// Make sure we have the tiles we are trying to exchange.
		for _, t := range m.Tiles() {
			// Leave implicitly checks the tiles here.
			_, err := Leave(g.players[g.onturn].rack.TilesOn(), m.Tiles())
			if err != nil {
				return nil, err
			}

			if !g.players[g.onturn].rack.Has(t) {
				return nil, fmt.Errorf("your play contained a tile not in your rack: %v",
					t.UserVisible(g.alph))
			}
		}
		// no error, all tiles are here.
		return nil, nil
	} else if m.Action() == move.MoveTypePass {
		// This is always valid.
		return nil, nil
	} else if m.Action() == move.MoveTypeUnsuccessfulChallengePass {
		return nil, nil
	} else if m.Action() == move.MoveTypePlay {
		if g.playing == pb.PlayState_WAITING_FOR_FINAL_PASS {
			return nil, errors.New("you can only pass or challenge")
		}
		return g.validateTilePlayMove(m)
	} else {
		return nil, fmt.Errorf("move type %v is not user-inputtable", m.Action())
	}
}

func (g *Game) validateTilePlayMove(m *move.Move) ([]alphabet.MachineWord, error) {
	if m.TilesPlayed() > RackTileLimit {
		return nil, errors.New("your play contained too many tiles")
	}
	// Check that our move actually uses the tiles on our rack.
	_, err := Leave(g.players[g.onturn].rack.TilesOn(), m.Tiles())
	if err != nil {
		return nil, err
	}

	row, col, vert := m.CoordsAndVertical()
	err = g.Board().ErrorIfIllegalPlay(row, col, vert, m.Tiles())
	if err != nil {
		return nil, err
	}

	// The play is legal. What words does it form?
	formedWords, err := g.Board().FormedWords(m)
	if err != nil {
		return nil, err
	}
	if g.history.ChallengeRule == pb.ChallengeRule_VOID {
		// Actually check the validity of the words.
		illegalWords := validateWords(g.gaddag, formedWords)

		if len(illegalWords) > 0 {
			return nil, fmt.Errorf("the play contained illegal words: %v",
				strings.Join(illegalWords, ", "))
		}
	}
	return formedWords, nil
}

func (g *Game) endOfGameCalcs(onturn int, addToHistory bool) {
	unplayedPts := g.calculateRackPts(otherPlayer(onturn)) * 2

	g.players[onturn].points += unplayedPts
	if addToHistory {
		g.history.Events = append(g.history.Events, g.endRackEvt(onturn, unplayedPts))
	}
	log.Debug().Int("onturn", onturn).Int("unplayedpts", unplayedPts).Interface("players", g.players).
		Msg("endOfGameCalcs")
}

// PlayMove plays a move on the board. This function is meant to be used
// by simulators as it implements a subset of possible moves, and by remote
// gameplay engines as much as possible.
// If the millis argument is passed in, it adds this value to the history
// as the time remaining for the user (when they played the move).
func (g *Game) PlayMove(m *move.Move, addToHistory bool, millis int) error {

	if g.backupMode != NoBackup {
		g.backupState()
	}
	if addToHistory {
		// Also, validate that the move follows the rules.
		wordsFormed, err := g.ValidateMove(m)
		if err != nil {
			return err
		}
		g.lastWordsFormed = wordsFormed
	}

	switch m.Action() {
	case move.MoveTypePlay:
		g.board.PlayMove(m, g.gaddag, g.bag.LetterDistribution())
		score := m.Score()
		if score != 0 {
			g.scorelessTurns = 0
		}
		g.players[g.onturn].points += score
		if m.TilesPlayed() == 7 {
			g.players[g.onturn].bingos++
		}
		drew := g.bag.DrawAtMost(m.TilesPlayed())
		tiles := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].setRackTiles(tiles, g.alph)

		if addToHistory {
			evt := g.EventFromMove(m)
			evt.MillisRemaining = int32(millis)
			g.history.LastKnownRacks[g.onturn] = g.RackLettersFor(g.onturn)
			g.history.Events = append(g.history.Events, evt)
		}

		if g.players[g.onturn].rack.NumTiles() == 0 {
			if g.history.ChallengeRule != pb.ChallengeRule_VOID {
				// Basically, if the challenge rule is not void,
				// wait for the final pass (or challenge).
				g.playing = pb.PlayState_WAITING_FOR_FINAL_PASS
				g.history.PlayState = g.playing
				log.Info().Msg("waiting for final pass... (commit pass)")
			} else {
				log.Debug().Msg("game is over")
				g.playing = pb.PlayState_GAME_OVER
				g.history.PlayState = g.playing
				g.endOfGameCalcs(g.onturn, addToHistory)
			}
		}

	case move.MoveTypePass, move.MoveTypeUnsuccessfulChallengePass:
		// XXX: It would be ideal to log an unsuccessful challenge pass at
		// the end of the game, at least as a statistic (in DOUBLE challenge),
		// but that's not compatible with Quackle.
		if g.playing == pb.PlayState_WAITING_FOR_FINAL_PASS {
			g.playing = pb.PlayState_GAME_OVER
			g.history.PlayState = g.playing

			// Note that the player "on turn" changes here, as we created
			// a fake virtual turn on the pass. We need to calculate
			// the final score correctly.
			g.endOfGameCalcs((g.onturn+1)%2, addToHistory)
		} else {
			// If this is a regular pass (and not an end-of-game-pass) let's
			// log it in the history.
			g.scorelessTurns++
			if addToHistory {
				evt := g.EventFromMove(m)
				evt.MillisRemaining = int32(millis)
				g.history.Events = append(g.history.Events, evt)
			}
		}

	case move.MoveTypeExchange:
		drew, err := g.bag.Exchange([]alphabet.MachineLetter(m.Tiles()))
		if err != nil {
			return err
		}
		tiles := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].setRackTiles(tiles, g.alph)
		log.Debug().Str("newrack", g.players[g.onturn].rackLetters).Msg("new-rack")
		g.scorelessTurns++
		if addToHistory {
			evt := g.EventFromMove(m)
			evt.MillisRemaining = int32(millis)
			g.history.LastKnownRacks[g.onturn] = g.RackLettersFor(g.onturn)
			g.history.Events = append(g.history.Events, evt)
		}
	}

	gameEnded, err := g.handleConsecutiveScorelessTurns(addToHistory)
	if err != nil {
		return err
	}
	if !gameEnded {
		g.onturn = (g.onturn + 1) % len(g.players)
	}
	g.turnnum++

	// log.Debug().Interface("history", g.history).Int("onturn", g.onturn).Int("turnnum", g.turnnum).
	// 	Msg("newhist")
	return nil
}

func (g *Game) handleConsecutiveScorelessTurns(addToHistory bool) (bool, error) {
	var ended bool
	if g.scorelessTurns == 6 {
		ended = true
		log.Debug().Msg("game ended with 6 scoreless turns")
		g.playing = pb.PlayState_GAME_OVER
		g.history.PlayState = g.playing

		pts := g.calculateRackPts(g.onturn)
		g.players[g.onturn].points -= pts
		if addToHistory {
			penaltyEvt := g.endRackPenaltyEvt(pts)
			g.history.Events = append(g.history.Events, penaltyEvt)
		}
		g.onturn = (g.onturn + 1) % len(g.players)
		pts = g.calculateRackPts(g.onturn)
		g.players[g.onturn].points -= pts
		if addToHistory {
			penaltyEvt := g.endRackPenaltyEvt(pts)
			g.history.Events = append(g.history.Events, penaltyEvt)
		}
		log.Debug().Interface("players", g.players).Msg("player-states")
	}
	return ended, nil
}

// PlayScoringMove plays a move on a board that is described by the
// coordinates and word only. It returns the move.
func (g *Game) PlayScoringMove(coords, word string, addToHistory bool) (*move.Move, error) {
	playerid := g.onturn
	rack := g.RackFor(playerid).String()

	m, err := g.CreateAndScorePlacementMove(coords, word, rack)
	if err != nil {
		log.Error().Msgf("Trying to create and score move, err was %v", err)
		return nil, err
	}
	// Actually make the play on the board:
	g.PlayMove(m, addToHistory, 0)
	return m, nil
}

// CreateAndScorePlacementMove creates a *move.Move from the coords and
// given tiles. It scores the move, calculates the leave, etc. This should
// be used when a person is interacting with the interface.
func (g *Game) CreateAndScorePlacementMove(coords string, tiles string, rack string) (*move.Move, error) {

	row, col, vertical := move.FromBoardGameCoords(coords)

	// convert tiles to MachineWord
	mw, err := alphabet.ToMachineWord(tiles, g.alph)
	if err != nil {
		return nil, err
	}
	rackmw, err := alphabet.ToMachineWord(rack, g.alph)
	if err != nil {
		return nil, err
	}

	err = modifyForPlaythrough(mw, g.board, vertical, row, col)
	if err != nil {
		return nil, err
	}

	leavemw, err := Leave(rackmw, mw)
	if err != nil {
		return nil, err
	}
	err = g.Board().ErrorIfIllegalPlay(row, col, vertical, mw)
	if err != nil {
		return nil, err
	}
	// Notes: the cross direction is in the opposite direction that the
	// play is actually in. Additionally, we transpose the board if
	// the play is vertical, due to how the scoring routine works.
	// We transpose it back at the end.
	crossDir := board.VerticalDirection
	if vertical {
		crossDir = board.HorizontalDirection
		row, col = col, row
		g.Board().Transpose()
	}
	tilesPlayed := len(rackmw) - len(leavemw)

	// ScoreWord assumes the play is always horizontal, so we have to
	// do the transpositions beforehand.
	score := g.Board().ScoreWord(mw, row, col, tilesPlayed,
		crossDir, g.bag.LetterDistribution())
	// reset row, col back for the actual creation of the play.
	if vertical {
		row, col = col, row
		g.Board().Transpose()
	}
	m := move.NewScoringMove(score, mw, leavemw, vertical, tilesPlayed,
		g.alph, row, col, coords)
	return m, nil

}

func (g *Game) calculateRackPts(onturn int) int {
	rack := g.players[onturn].rack
	return rack.ScoreOn(g.bag.LetterDistribution())
}

func otherPlayer(idx int) int {
	return (idx + 1) % 2
}

func (g *Game) PlayToTurn(turnnum int) error {
	log.Debug().Int("turnnum", turnnum).Msg("playing to turn")
	if turnnum < 0 || turnnum > len(g.history.Events) {
		return fmt.Errorf("game has %v turns, you have chosen a turn outside the range",
			len(g.history.Events))
	}
	if g.board == nil {
		return fmt.Errorf("board does not exist")
	}
	if g.bag == nil {
		return fmt.Errorf("bag has not been initialized; need to start game")
	}

	g.board.Clear()
	g.bag.Refill()
	g.players.resetScore()
	g.players.resetRacks()
	g.turnnum = 0
	g.onturn = 0
	if g.history.SecondWentFirst {
		g.onturn = 1
	}
	g.playing = pb.PlayState_PLAYING
	g.history.PlayState = g.playing
	var t int
	for t = 0; t < turnnum; t++ {
		err := g.playTurn(t)
		if err != nil {
			return err
		}
		// g.onturn will get rewritten in the next iteration
		g.onturn = (g.onturn + 1) % 2
		log.Debug().Int("turn", t).Msg("played turn")
	}
	if t >= len(g.history.Events) {
		if len(g.history.LastKnownRacks[0]) > 0 && len(g.history.LastKnownRacks[1]) > 0 {
			g.SetRacksForBoth([]*alphabet.Rack{
				alphabet.RackFromString(g.history.LastKnownRacks[0], g.alph),
				alphabet.RackFromString(g.history.LastKnownRacks[1], g.alph),
			})
		} else if len(g.history.LastKnownRacks[0]) > 0 {
			// Rack1 but not rack2
			g.SetRackFor(0, alphabet.RackFromString(g.history.LastKnownRacks[0], g.alph))
		} else if len(g.history.LastKnownRacks[1]) > 0 {
			// Rack2 but not rack1
			g.SetRackFor(1, alphabet.RackFromString(g.history.LastKnownRacks[1], g.alph))
		} else {
			// They're both blank.
			// We don't have a recorded rack, so set it to a random one.
			g.SetRandomRack(g.onturn)
		}
	} else {
		// playTurn should have refilled the rack of the relevant player,
		// who was on turn.
		// So set the currently on turn's rack to whatever is in the history.
		log.Debug().Int("turn", t).Msg("setting rack from turn")
		err := g.SetRackFor(g.onturn, alphabet.RackFromString(
			g.history.Events[t].Rack, g.alph))
		if err != nil {
			return err
		}
	}

	for _, p := range g.players {
		if p.rack.NumTiles() == 0 {
			log.Debug().Msgf("Player %v has no tiles, game is over.", p)
			g.playing = pb.PlayState_GAME_OVER
			g.history.PlayState = g.playing

			break
		}
	}
	return nil
}

func (g *Game) playTurn(t int) error {
	// XXX: This function is pretty similar to PlayMove above. It has a
	// subset of the functionality as it's designed to replay an already
	// recorded turn on the board.
	evt := g.history.Events[t]

	// onturn should be based on the event nickname
	found := false
	for idx, p := range g.players {
		if p.Nickname == evt.Nickname {
			g.onturn = idx
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("player not found: %v", evt.Nickname)
	}

	// Set the rack for the user on turn to the rack in the history.
	g.SetRackFor(g.onturn, alphabet.RackFromString(evt.Rack, g.alph))
	m := MoveFromEvent(evt, g.alph, g.board)

	switch m.Action() {
	case move.MoveTypePlay:
		// We back up the board and bag since there's a possibility
		// this play will have to be taken back, if it's a challenged phony.
		g.board.SaveCopy()
		g.board.PlayMove(m, g.gaddag, g.bag.LetterDistribution())
		g.players[g.onturn].points += m.Score()
		if m.TilesPlayed() == 7 {
			g.players[g.onturn].bingos++
		}

		// Note that what we draw here (and in exchange, below) may not
		// be what was recorded. That's ok -- we always set the rack
		// at the beginning to whatever was recorded. Drawing like
		// normal, though, ensures we don't have to reconcile any
		// tiles with the bag.
		drew := g.bag.DrawAtMost(m.TilesPlayed())
		tiles := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].setRackTiles(tiles, g.alph)

		// Don't check game end logic here, as we assume we have the
		// right event for that (move.MoveTypeEndgameTiles for example).
	case move.MoveTypePhonyTilesReturned:
		// Score should have the proper sign at creation time
		g.players[g.onturn].points += m.Score()
		if m.TilesPlayed() == 7 {
			g.players[g.onturn].bingos--
		}
		g.board.RestoreFromCopy()
		// Throw the rack we drew after the phony back in the bag:
		g.players[g.onturn].throwRackIn(g.bag)
		// Also throw the tiles in the event back in the bag, so that
		// we can then re-draw them
		// NOTE: the event must have the PlayedTiles attribute set! This
		// must be done in the gcg parser, or whatever is generating these
		// events. (See challenge module as well)
		playedTiles := strings.ReplaceAll(evt.PlayedTiles, string(alphabet.ASCIIPlayedThrough), "")
		mw, err := alphabet.ToMachineWord(playedTiles, g.alph)
		if err != nil {
			return err
		}
		g.bag.PutBack(mw)
		// Set the tiles to be the tiles in the event.
		r := alphabet.NewRack(g.alph)
		r.Set(m.Tiles())
		g.SetRackFor(g.onturn, r)

	case move.MoveTypeChallengeBonus, move.MoveTypeEndgameTiles,
		move.MoveTypeLostTileScore, move.MoveTypeLostScoreOnTime:

		// The score should already have the proper sign at creation time.
		g.players[g.onturn].points += m.Score()

	case move.MoveTypeExchange:

		drew, err := g.bag.Exchange([]alphabet.MachineLetter(m.Tiles()))
		if err != nil {
			panic(err)
		}
		tiles := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].setRackTiles(tiles, g.alph)
	default:
		// Nothing

	}

	g.turnnum++
	return nil
}

// SetRackFor sets the player's current rack. It throws an error if
// the rack is impossible to set from the current unseen tiles. It
// puts tiles back from opponent racks and our own racks, then sets the rack,
// and finally redraws for opponent.
func (g *Game) SetRackFor(playerIdx int, rack *alphabet.Rack) error {
	// Put our tiles back in the bag, as well as our opponent's tiles.
	g.ThrowRacksIn()

	// Check if we can actually set our rack now that these tiles are in the
	// bag.
	log.Debug().Str("rack", rack.TilesOn().UserVisible(g.alph)).Msg("removing from bag")
	err := g.bag.RemoveTiles(rack.TilesOn())
	if err != nil {
		log.Error().Msgf("Unable to set rack: %v", err)
		return err
	}

	// success; set our rack
	g.players[playerIdx].rack = rack
	g.players[playerIdx].rackLetters = rack.String()
	log.Debug().Str("rack", g.players[playerIdx].rackLetters).
		Int("player", playerIdx).Msg("set rack")
	// And redraw a random rack for opponent.
	g.SetRandomRack(otherPlayer(playerIdx))

	return nil
}

// SetRacksForBoth sets both racks at the same time.
func (g *Game) SetRacksForBoth(racks []*alphabet.Rack) error {
	g.ThrowRacksIn()
	for _, rack := range racks {
		err := g.bag.RemoveTiles(rack.TilesOn())
		if err != nil {
			log.Error().Msgf("Unable to set rack: %v", err)
			return err
		}
	}
	for idx, player := range g.players {
		player.rack = racks[idx]
		player.rackLetters = racks[idx].String()
	}
	return nil
}

// ThrowRacksIn throws both players' racks back in the bag.
func (g *Game) ThrowRacksIn() {
	g.players[0].throwRackIn(g.bag)
	g.players[1].throwRackIn(g.bag)
}

// SetRandomRack sets the player's rack to a random rack drawn from the bag.
// It tosses the current rack back in first. This is used for simulations.
func (g *Game) SetRandomRack(playerIdx int) {
	// log.Debug().Int("player", playerIdx).Str("rack", g.RackFor(playerIdx).TilesOn().UserVisible(g.alph)).
	// 	Msg("setting random rack..")
	tiles := g.bag.Redraw(g.RackFor(playerIdx).TilesOn())
	g.players[playerIdx].setRackTiles(tiles, g.alph)
	// log.Debug().Int("player", playerIdx).Str("newrack", g.players[playerIdx].rackLetters).
	// 	Msg("set random rack")
}

// RackFor returns the rack for the player with the passed-in index
func (g *Game) RackFor(playerIdx int) *alphabet.Rack {
	return g.players[playerIdx].rack
}

// RackLettersFor returns a user-visible representation of the player's rack letters
func (g *Game) RackLettersFor(playerIdx int) string {
	return g.RackFor(playerIdx).String()
}

// PointsFor returns the number of points for the given player
func (g *Game) PointsFor(playerIdx int) int {
	return g.players[playerIdx].points
}

func (g *Game) PointsForNick(nick string) int {
	for i := range g.players {
		if g.players[i].Nickname == nick {
			return g.players[i].points
		}
	}
	return 0
}

func (g *Game) BingosForNick(nick string) int {
	for i := range g.players {
		if g.players[i].Nickname == nick {
			return g.players[i].bingos
		}
	}
	return 0
}

func (g *Game) SpreadFor(playerIdx int) int {
	return g.PointsFor(playerIdx) - g.PointsFor(otherPlayer(playerIdx))
}

// NumPlayers is always 2.
func (g *Game) NumPlayers() int {
	return 2
}

// Bag returns the current bag
func (g *Game) Bag() *alphabet.Bag {
	return g.bag
}

// Board returns the current board state.
func (g *Game) Board() *board.GameBoard {
	return g.board
}

// Gaddag returns this game's gaddag data structure.
func (g *Game) Gaddag() *gaddag.SimpleGaddag {
	return g.gaddag
}

func (g *Game) Turn() int {
	return g.turnnum
}

func (g *Game) Uid() string {
	return g.history.Uid
}

func (g *Game) Playing() pb.PlayState {
	return g.playing
}

func (g *Game) SetPlaying(s pb.PlayState) {
	g.playing = s
}

func (g *Game) PlayerOnTurn() int {
	return g.onturn
}

func (g *Game) NickOnTurn() string {
	return g.players[g.onturn].Nickname
}

func (g *Game) PlayerIDOnTurn() string {
	return g.players[g.onturn].UserId
}

func (g *Game) SetPlayerOnTurn(onTurn int) {
	g.onturn = onTurn
}

func (g *Game) SetPointsFor(player, pts int) {
	g.players[player].points = pts
}

func (g *Game) Alphabet() *alphabet.Alphabet {
	return g.alph
}

func (g *Game) CurrentSpread() int {
	return g.PointsFor(g.onturn) - g.PointsFor((g.onturn+1)%2)
}

func (g *Game) History() *pb.GameHistory {
	return g.history
}

func (g *Game) FirstPlayer() *pb.PlayerInfo {
	return &g.players[g.wentfirst].PlayerInfo
}
