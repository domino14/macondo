// Package game encapsulates the main mechanics for a Crossword Game. It
// interacts heavily with the protobuf data structures.
package game

import (
	"errors"
	"fmt"
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/lexicon"
	"github.com/domino14/macondo/move"
	"github.com/lithammer/shortuuid"
	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"
)

const (
	//IdentificationAuthority is the authority that gives out game IDs
	IdentificationAuthority = "io.woogles"

	MacondoCreation = "Created with Macondo"

	ExchangeLimit = 7
	RackTileLimit = 7

	DefaultMaxScorelessTurns = 6
)

// Game is the actual internal game structure that controls the entire
// business logic of the game; drawing, making moves, etc. The two
// structures above are basically data entities.
// Note: a Game doesn't care how it is played. It is just rules for gameplay.
// AI players, human players, etc will play a game outside of the scope of
// this module.
type Game struct {
	config      *config.Config
	crossSetGen cross_set.Generator
	lexicon     lexicon.Lexicon
	alph        *alphabet.Alphabet
	// board and bag will contain the latest (current) versions of these.
	board              *board.GameBoard
	letterDistribution *alphabet.LetterDistribution
	bag                *alphabet.Bag

	playing pb.PlayState

	wentfirst         int
	scorelessTurns    int
	maxScorelessTurns int
	onturn            int
	turnnum           int
	players           playerStates
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
	// if nextFirst is -1, first is determined randomly. Otherwise, first is
	// set to nextFirst.
	nextFirst int
	// rules contains the original game rules passed in to create this game.
	rules *GameRules
}

func (g *Game) Config() *config.Config {
	return g.config
}

func (g *Game) Lexicon() lexicon.Lexicon {
	return g.lexicon
}

func (g *Game) LexiconName() string {
	return g.lexicon.Name()
}

func (g *Game) LastWordsFormed() []alphabet.MachineWord {
	return g.lastWordsFormed
}

func (g *Game) Rules() *GameRules {
	return g.rules
}

func (g *Game) LastEvent() *pb.GameEvent {
	last := len(g.history.Events) - 1
	if last < 0 {
		return nil
	}
	return g.history.Events[last]
}

func (g *Game) addEventToHistory(evt *pb.GameEvent) {
	log.Debug().Msgf("Adding event to history: %v", evt)
	g.history.Events = append(g.history.Events, evt)
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
	his.Uid = shortuuid.New()[2:10] // It is up to the caller to check for duplication.
	his.Description = MacondoCreation
	his.Events = []*pb.GameEvent{}
	his.SecondWentFirst = flipfirst
	his.LastKnownRacks = []string{"", ""}
	return his
}

// NewGame is how one instantiates a brand new game.
func NewGame(rules *GameRules, playerinfo []*pb.PlayerInfo) (*Game, error) {
	game := &Game{}
	game.letterDistribution = rules.LetterDistribution()
	game.alph = game.letterDistribution.Alphabet()
	game.backupMode = NoBackup
	game.nextFirst = -1
	game.board = rules.Board().Copy()
	game.crossSetGen = rules.CrossSetGen()
	game.lexicon = rules.Lexicon()
	game.config = rules.Config()
	game.rules = rules
	game.maxScorelessTurns = DefaultMaxScorelessTurns

	game.players = make([]*playerState, len(playerinfo))
	ids := map[string]bool{}
	for idx, p := range playerinfo {
		game.players[idx] = &playerState{
			PlayerInfo: pb.PlayerInfo{
				Nickname: p.Nickname,
				UserId:   p.UserId,
				RealName: p.RealName},
		}
		ids[p.Nickname] = true
	}
	if len(ids) < len(playerinfo) {
		return nil, errors.New("all player nicknames must be unique")
	}
	return game, nil
}

// NewFromHistory instantiates a Game from a history, and sets the current
// turn to the passed in turnnum. It assumes the rules contains the current
// lexicon in history, if any!
func NewFromHistory(history *pb.GameHistory, rules *GameRules, turnnum int) (*Game, error) {
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
	game.bag = game.letterDistribution.MakeBag()
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

func NewFromSnapshot(rules *GameRules, players []*pb.PlayerInfo, lastKnownRacks []string,
	scores []int, boardRows []string) (*Game, error) {

	// This NewGame function copies the board from rules as well.
	game, err := NewGame(rules, players)
	if err != nil {
		return nil, err
	}

	game.history = newHistory(game.players, false)

	game.bag = game.letterDistribution.MakeBag()
	for i := 0; i < game.NumPlayers(); i++ {
		game.players[i].rack = alphabet.NewRack(game.alph)
	}

	playedLetters := []alphabet.MachineLetter{}
	for i, row := range boardRows {
		playedLetters = append(playedLetters,
			game.board.SetRow(i, row, rules.LetterDistribution().Alphabet())...)
	}

	err = game.bag.RemoveTiles(playedLetters)
	if err != nil {
		return nil, err
	}
	game.history.LastKnownRacks = lastKnownRacks
	// Set racks and tiles
	racks := []*alphabet.Rack{
		alphabet.RackFromString(game.history.LastKnownRacks[0], game.Alphabet()),
		alphabet.RackFromString(game.history.LastKnownRacks[1], game.Alphabet()),
	}
	game.history.Lexicon = game.Lexicon().Name()
	game.history.Variant = string(game.rules.Variant())
	game.history.LetterDistribution = game.rules.LetterDistributionName()
	game.history.BoardLayout = game.rules.BoardName()

	// set racks for both players; this removes the relevant letters from the bag.
	err = game.SetRacksForBoth(racks)
	if err != nil {
		return nil, err
	}

	// set scores
	for i, s := range scores {
		game.players[i].resetScore()
		game.players[i].points = s
	}
	// onturn is 0 by default, which is always correct in this function, as
	// the player to go next is listed first in the players/racks/scores.
	game.playing = pb.PlayState_PLAYING
	game.history.PlayState = game.playing

	if game.bag.TilesRemaining() == 0 && (game.RackFor(0).NumTiles() == 0 || game.RackFor(1).NumTiles() == 0) {
		game.playing = pb.PlayState_GAME_OVER
		game.history.PlayState = game.playing
		log.Info().Msg("this game is already over")
	}

	return game, nil
}

// SetNextFirst sets the player going first to the passed-in value. This
// will take effect the next time StartGame is called.
func (g *Game) SetNextFirst(first int) {
	g.nextFirst = first
}

// StartGame starts a game anew, dealing out tiles to both players.
func (g *Game) StartGame() {
	g.Board().Clear()
	g.bag = g.letterDistribution.MakeBag()
	var goesfirst int
	if g.nextFirst == -1 {
		goesfirst = frand.Intn(2)
		log.Debug().Msgf("randomly determined %v to go first", goesfirst)
	} else {
		goesfirst = g.nextFirst
		log.Debug().Msgf("forcing first to %v", g.nextFirst)
	}
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
	g.history.Lexicon = g.Lexicon().Name()
	g.history.Variant = string(g.rules.Variant())
	g.history.LetterDistribution = g.rules.LetterDistributionName()
	g.history.BoardLayout = g.rules.BoardName()
	g.playing = pb.PlayState_PLAYING
	g.history.PlayState = g.playing
	g.turnnum = 0
	g.scorelessTurns = 0
	g.onturn = goesfirst
	g.wentfirst = goesfirst
	g.lastWordsFormed = nil
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
	} else if m.Action() == move.MoveTypeChallenge {
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
		illegalWords := validateWords(g.lexicon, formedWords, g.rules.Variant())

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
		g.addEventToHistory(g.endRackEvt(onturn, unplayedPts))
	}
	log.Debug().Int("onturn", onturn).Int("unplayedpts", unplayedPts).Interface("players", g.players).
		Msg("endOfGameCalcs")
}

func (g *Game) SetMaxScorelessTurns(m int) {
	g.maxScorelessTurns = m
}

func (g *Game) SetScorelessTurns(n int) {
	g.scorelessTurns = n
}

// Convert the slice of MachineWord to user-visible, using the game's lexicon.
func convertToVisible(words []alphabet.MachineWord,
	alph *alphabet.Alphabet) []string {

	uvstrs := make([]string, len(words))
	for idx, w := range words {
		uvstrs[idx] = w.UserVisible(alph)
	}
	return uvstrs
}

// PlayMove plays a move on the board. This function is meant to be used
// by simulators as it implements a subset of possible moves, and by remote
// gameplay engines as much as possible.
// If the millis argument is passed in, it adds this value to the history
// as the time remaining for the user (when they played the move).
func (g *Game) PlayMove(m *move.Move, addToHistory bool, millis int) error {

	// We need to handle challenges separately.
	if m.Action() == move.MoveTypeChallenge {
		_, err := g.ChallengeEvent(0, 0)
		return err
	}

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
		ld := g.bag.LetterDistribution()
		g.board.PlayMove(m, ld)
		// Calculate cross-sets.
		g.crossSetGen.UpdateForMove(g.board, m)
		score := m.Score()
		if score != 0 {
			g.scorelessTurns = 0
		} // XXX: else we should increment the scoreless turns here!
		g.players[g.onturn].points += score
		g.players[g.onturn].turns += 1
		if m.TilesPlayed() == 7 {
			g.players[g.onturn].bingos++
		}
		drew := g.bag.DrawAtMost(m.TilesPlayed())
		tiles := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].setRackTiles(tiles, g.alph)

		if addToHistory {
			evt := g.EventFromMove(m)
			evt.MillisRemaining = int32(millis)
			evt.WordsFormed = convertToVisible(g.lastWordsFormed, g.alph)
			g.history.LastKnownRacks[g.onturn] = g.RackLettersFor(g.onturn)
			g.addEventToHistory(evt)
		}

		if g.players[g.onturn].rack.NumTiles() == 0 {
			// make sure not in sim mode. sim mode (montecarlo, endgame, etc) does not
			// generate end-of-game passes.
			if g.backupMode != SimulationMode && g.history != nil && g.history.ChallengeRule != pb.ChallengeRule_VOID {
				// Basically, if the challenge rule is not void,
				// wait for the final pass (or challenge).
				g.playing = pb.PlayState_WAITING_FOR_FINAL_PASS
				g.history.PlayState = g.playing
				log.Trace().Msg("waiting for final pass... (commit pass)")
			} else {
				log.Trace().Msg("game is over")
				g.playing = pb.PlayState_GAME_OVER
				if addToHistory {
					g.history.PlayState = g.playing
				}
				g.endOfGameCalcs(g.onturn, addToHistory)
				if addToHistory {
					g.AddFinalScoresToHistory()
				}
			}
		}

	case move.MoveTypePass, move.MoveTypeUnsuccessfulChallengePass:
		if g.playing == pb.PlayState_GAME_OVER {
			log.Warn().Msg("adding a pass when game is already over")
		}
		// Add the pass first so it comes before the end rack bonus
		if addToHistory {
			evt := g.EventFromMove(m)
			evt.MillisRemaining = int32(millis)
			g.addEventToHistory(evt)
		}
		if g.playing == pb.PlayState_WAITING_FOR_FINAL_PASS {
			g.playing = pb.PlayState_GAME_OVER
			g.history.PlayState = g.playing
			log.Trace().Msg("waiting -> gameover transition")
			// Note that the player "on turn" changes here, as we created
			// a fake virtual turn on the pass. We need to calculate
			// the final score correctly.
			g.endOfGameCalcs((g.onturn+1)%2, addToHistory)
			if addToHistory {
				g.AddFinalScoresToHistory()
			}
		} else {
			// If this is a regular pass (and not an end-of-game-pass) let's
			// log it in the history.
			g.scorelessTurns++
			g.players[g.onturn].turns += 1
		}

	case move.MoveTypeExchange:
		drew, err := g.bag.Exchange([]alphabet.MachineLetter(m.Tiles()))
		if err != nil {
			return err
		}
		tiles := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].setRackTiles(tiles, g.alph)
		log.Trace().Str("newrack", g.players[g.onturn].rackLetters).Msg("new-rack")
		g.scorelessTurns++
		g.players[g.onturn].turns += 1
		if addToHistory {
			evt := g.EventFromMove(m)
			evt.MillisRemaining = int32(millis)
			g.history.LastKnownRacks[g.onturn] = g.RackLettersFor(g.onturn)
			g.addEventToHistory(evt)
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

// AddFinalScoresToHistory adds the final scores and winner to the history.
func (g *Game) AddFinalScoresToHistory() {
	g.history.FinalScores = make([]int32, len(g.players))
	for pidx, p := range g.players {
		g.history.FinalScores[pidx] = int32(p.points)
	}
	if g.history.FinalScores[0] > g.history.FinalScores[1] {
		g.history.Winner = 0
	} else if g.history.FinalScores[0] < g.history.FinalScores[1] {
		g.history.Winner = 1
	} else {
		g.history.Winner = -1
	}
	log.Debug().Interface("finalscores", g.history.FinalScores).Msg("added-final-scores")
}

func (g *Game) handleConsecutiveScorelessTurns(addToHistory bool) (bool, error) {
	var ended bool
	if g.scorelessTurns == g.maxScorelessTurns {
		ended = true
		g.playing = pb.PlayState_GAME_OVER
		g.history.PlayState = g.playing

		pts := g.calculateRackPts(g.onturn)
		g.players[g.onturn].points -= pts
		if addToHistory {
			penaltyEvt := g.endRackPenaltyEvt(pts)
			g.addEventToHistory(penaltyEvt)
		}
		g.onturn = (g.onturn + 1) % len(g.players)
		pts = g.calculateRackPts(g.onturn)
		g.players[g.onturn].points -= pts
		if addToHistory {
			penaltyEvt := g.endRackPenaltyEvt(pts)
			g.addEventToHistory(penaltyEvt)
			g.AddFinalScoresToHistory()
		}
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
	err = g.PlayMove(m, addToHistory, 0)
	return m, err
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

	// Set backup mode to interactive gameplay mode so that we always have
	// a backup available while playing through the events.
	oldbackupMode := g.backupMode
	g.SetBackupMode(InteractiveGameplayMode)
	g.SetStateStackLength(1)
	for t = 0; t < turnnum; t++ {
		err := g.playTurn(t)
		if err != nil {
			return err
		}
		// g.onturn will get rewritten in the next iteration
		g.onturn = (g.onturn + 1) % 2
		log.Trace().Int("turn", t).Msg("played turn")
	}
	g.SetBackupMode(oldbackupMode)

	if t >= len(g.history.Events) {
		if len(g.history.LastKnownRacks[0]) > 0 && len(g.history.LastKnownRacks[1]) > 0 {
			g.SetRacksForBoth([]*alphabet.Rack{
				alphabet.RackFromString(g.history.LastKnownRacks[0], g.alph),
				alphabet.RackFromString(g.history.LastKnownRacks[1], g.alph),
			})
		} else if len(g.history.LastKnownRacks[0]) > 0 {
			// Rack1 but not rack2
			err := g.SetRackFor(0, alphabet.RackFromString(g.history.LastKnownRacks[0], g.alph))
			if err != nil {
				return err
			}
		} else if len(g.history.LastKnownRacks[1]) > 0 {
			// Rack2 but not rack1
			err := g.SetRackFor(1, alphabet.RackFromString(g.history.LastKnownRacks[1], g.alph))
			if err != nil {
				return err
			}
		} else {
			// They're both blank.
			// We don't have a recorded rack, so set it to a random one.
			g.SetRandomRack(g.onturn)
		}

		log.Debug().Str("r0", g.players[0].rackLetters).Str("r1", g.players[1].rackLetters).Msg("PlayToTurn-set-racks")

	} else {
		// playTurn should have refilled the rack of the relevant player,
		// who was on turn.
		// So set the currently on turn's rack to whatever is in the history.
		log.Trace().Int("turn", t).Msg("setting rack from turn")
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

// PlayLatestEvent "plays" the latest event on the board. This is used for
// replaying a game from a GCG.
func (g *Game) PlayLatestEvent() error {
	return g.playTurn(len(g.history.Events) - 1)
}

func (g *Game) playTurn(t int) error {
	// XXX: This function is pretty similar to PlayMove above. It has a
	// subset of the functionality as it's designed to replay an already
	// recorded turn on the board.
	evt := g.history.Events[t]
	log.Trace().Int("event-type", int(evt.Type)).Int("turn", t).Msg("playTurn")
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

	m, err := MoveFromEvent(evt, g.alph, g.board)
	if err != nil {
		return err
	}
	log.Trace().Int("movetype", int(m.Action())).Msg("move-action")
	switch m.Action() {
	case move.MoveTypePlay:
		// Set the rack for the user on turn to the rack in the history.
		err := g.SetRackFor(g.onturn, alphabet.RackFromString(evt.Rack, g.alph))
		if err != nil {
			return err
		}
		// We validate tile play moves only.
		wordsFormed, err := g.ValidateMove(m)
		if err != nil {
			return err
		}

		g.lastWordsFormed = wordsFormed
		// We back up the board and bag since there's a possibility
		// this play will have to be taken back, if it's a challenged phony.
		g.backupState()

		ld := g.bag.LetterDistribution()
		g.board.PlayMove(m, ld)
		g.crossSetGen.UpdateForMove(g.board, m)
		g.players[g.onturn].points += m.Score()
		if m.TilesPlayed() == 7 {
			g.players[g.onturn].bingos++
		}
		evt.WordsFormed = convertToVisible(g.lastWordsFormed, g.alph)
		// Note that what we draw here (and in exchange, below) may not
		// be what was recorded. That's ok -- we always set the rack
		// at the beginning to whatever was recorded. Drawing like
		// normal, though, ensures we don't have to reconcile any
		// tiles with the bag.
		drew := g.bag.DrawAtMost(m.TilesPlayed())
		tiles := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].setRackTiles(tiles, g.alph)
		g.scorelessTurns = 0
		// Don't check game end logic here, as we assume we have the
		// right event for that (move.MoveTypeEndgameTiles for example).
	case move.MoveTypePhonyTilesReturned:
		// Unplaying the last move restores the state as it was prior
		// to the phony being played.
		g.UnplayLastMove()
		g.lastWordsFormed = nil
		g.scorelessTurns++

	case move.MoveTypeChallengeBonus, move.MoveTypeEndgameTiles,
		move.MoveTypeLostTileScore, move.MoveTypeLostScoreOnTime:

		// The score should already have the proper sign at creation time.
		g.players[g.onturn].points += m.Score()
		g.lastWordsFormed = nil // prevent another +5 on app restart

	case move.MoveTypeExchange:
		// Set the rack for the user on turn to the rack in the history.
		err := g.SetRackFor(g.onturn, alphabet.RackFromString(evt.Rack, g.alph))
		if err != nil {
			return err
		}
		drew, err := g.bag.Exchange([]alphabet.MachineLetter(m.Tiles()))
		if err != nil {
			panic(err)
		}
		tiles := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].setRackTiles(tiles, g.alph)
		g.players[g.onturn].turns += 1
		g.scorelessTurns++

	case move.MoveTypePass, move.MoveTypeUnsuccessfulChallengePass:
		g.scorelessTurns++

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
	log.Trace().Str("rack", rack.TilesOn().UserVisible(g.alph)).Msg("removing from bag")
	err := g.bag.RemoveTiles(rack.TilesOn())
	if err != nil {
		log.Error().Msgf("Unable to set rack: %v", err)
		return err
	}

	// success; set our rack
	g.players[playerIdx].rack = rack
	g.players[playerIdx].rackLetters = rack.String()
	log.Trace().Str("rack", g.players[playerIdx].rackLetters).
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
			log.Error().Msgf("both: Unable to set rack: %v", err)
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

func (g *Game) TurnsForNick(nick string) int {
	for i := range g.players {
		if g.players[i].Nickname == nick {
			return g.players[i].turns
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

func (g *Game) SetBoard(b *board.GameBoard) {
	g.board = b
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

func (g *Game) NextPlayer() int {
	return (g.onturn + 1) % g.NumPlayers()
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

func (g *Game) SetHistory(h *pb.GameHistory) {
	g.history = h
}

func (g *Game) FirstPlayer() *pb.PlayerInfo {
	return &g.players[g.wentfirst].PlayerInfo
}

func (g *Game) RecalculateBoard() {
	// Recalculate cross-sets and anchors for move generator
	g.crossSetGen.GenerateAll(g.board)
	g.board.UpdateAllAnchors()
}

func (g *Game) ScorelessTurns() int {
	return g.scorelessTurns
}
