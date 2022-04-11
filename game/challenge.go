package game

import (
	"errors"

	"github.com/domino14/macondo/alphabet"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/lexicon"
	"github.com/domino14/macondo/move"
	"github.com/rs/zerolog/log"
)

// SetChallengeRule sets the challenge rule for a game. The game
// must already be started with StartGame above (call immediately afterwards).
// It would default to the 0 state (VOID) otherwise.
func (g *Game) SetChallengeRule(rule pb.ChallengeRule) {
	g.history.ChallengeRule = rule
}

// ChallengeEvent should only be called if there is a history of events.
// It has the logic for appending challenge events and calculating scores
// properly.
// Note that this event can change the history of the game, including
// things like reset gameEnded back to false (for example if someone plays
// out with a phony).
// Return playLegal, error
func (g *Game) ChallengeEvent(addlBonus int, millis int) (bool, error) {
	if len(g.history.Events) == 0 {
		return false, errors.New("this game has no history")
	}
	if g.history.ChallengeRule == pb.ChallengeRule_VOID {
		return false, errors.New("challenges are not valid in void")
	}
	if len(g.lastWordsFormed) == 0 {
		return false, errors.New("there are no words to challenge")
	}
	// Note that the player on turn right now needs to be the player
	// who is making the challenge.
	illegalWords := validateWords(g.lexicon, g.lastWordsFormed, g.rules.Variant())
	playLegal := len(illegalWords) == 0

	lastEvent := g.history.Events[len(g.history.Events)-1]
	cumeScoreBeforeChallenge := lastEvent.Cumulative

	challengee := otherPlayer(g.onturn)

	offBoardEvent := &pb.GameEvent{
		Nickname:    lastEvent.Nickname,
		Type:        pb.GameEvent_PHONY_TILES_RETURNED,
		LostScore:   lastEvent.Score,
		Cumulative:  cumeScoreBeforeChallenge - lastEvent.Score,
		Rack:        lastEvent.Rack,
		PlayedTiles: lastEvent.PlayedTiles,
		// Note: these millis remaining would be the challenger's
		MillisRemaining: int32(millis),
	}

	var err error
	// This ideal system makes it so someone always loses
	// the game.
	if g.history.ChallengeRule == pb.ChallengeRule_TRIPLE {
		// Set the winner and loser before calling PlayMove, as
		// that changes who is on turn
		var winner int32
		if playLegal {
			// The challenge was wrong, they lose the game
			winner = int32(challengee)
		} else {
			// The challenger was right, they win the game
			winner = int32(g.onturn)
			// Take the play off the board.
			g.addEventToHistory(offBoardEvent)
			g.UnplayLastMove()
			g.history.LastKnownRacks[challengee] = lastEvent.Rack
		}
		g.history.Winner = winner

		// Don't call AddFinalScoresToHistory, this will
		// overwrite the correct winner
		g.playing = pb.PlayState_GAME_OVER
		g.history.PlayState = g.playing

		// This is the only case where the winner needs to be determined
		// independently from the score, so we copy just these lines from
		// AddFinalScoresToHistory.
		g.history.FinalScores = make([]int32, len(g.players))
		for pidx, p := range g.players {
			g.history.FinalScores[pidx] = int32(p.points)
		}

	} else if !playLegal {
		log.Debug().Msg("Successful challenge")

		// the play comes off the board. Add the offBoardEvent.
		g.addEventToHistory(offBoardEvent)

		// Unplay the last move to restore everything as it was board-wise
		// (and un-end the game if it had ended)
		g.UnplayLastMove()

		// We must also set the last known rack of the challengee back to
		// their rack before they played the phony.
		g.history.LastKnownRacks[challengee] = lastEvent.Rack
		// Explicitly set racks for both players. This prevents a bug where
		// part of the game may have been loaded from a GameHistory (through the
		// PlayGameToTurn flow) and the racks continually get reset.
		g.SetRacksForBoth([]*alphabet.Rack{
			alphabet.RackFromString(g.history.LastKnownRacks[0], g.alph),
			alphabet.RackFromString(g.history.LastKnownRacks[1], g.alph),
		})

		// Note that if backup mode is InteractiveGameplayMode, which it should be,
		// we do not back up the turn number. So restoring it doesn't change
		// the turn number or whose turn it is; unplay only copies the
		// needed variables.
		// and we must add one to scoreless turns:
		g.scorelessTurns++
		_, err = g.handleConsecutiveScorelessTurns(true)
		if err != nil {
			return playLegal, err
		}

		// Finally, let's re-shuffle the bag. This is so we don't give the
		// player who played the phony knowledge about the next few tiles in the bag.
		g.bag.Shuffle()
	} else {
		log.Debug().Msg("Unsuccessful challenge")

		addPts := int32(0)
		shouldAddPts := false

		bonusScoreEvent := func(bonus int32) *pb.GameEvent {
			return &pb.GameEvent{
				Nickname:   lastEvent.Nickname,
				Type:       pb.GameEvent_CHALLENGE_BONUS,
				Rack:       g.players[challengee].rackLetters,
				Bonus:      bonus + int32(addlBonus),
				Cumulative: cumeScoreBeforeChallenge + bonus + int32(addlBonus),
				// Note: these millis remaining would be the challenger's
				MillisRemaining: int32(millis),
			}
		}

		switch g.history.ChallengeRule {
		case pb.ChallengeRule_DOUBLE:
			// This "draconian" American system makes it so someone always loses
			// their turn.
			// challenger was wrong. They lose their turn.
			// XXX: Note -- we're not handling the six consecutive zero rule here.
			// This is an extreme edge case -- it would have to be a zero-point tile placement
			// move after a number of zero point moves.
			g.PlayMove(move.NewUnsuccessfulChallengePassMove(
				g.players[g.onturn].rack.TilesOn(), g.alph), true, millis)

		case pb.ChallengeRule_FIVE_POINT:
			// Append a bonus to the event.
			shouldAddPts = true
			addPts = 5

		case pb.ChallengeRule_TEN_POINT:
			shouldAddPts = true
			addPts = 10

		case pb.ChallengeRule_SINGLE:
			shouldAddPts = true
			addPts = 0
		}

		if shouldAddPts {
			evt := bonusScoreEvent(addPts)
			log.Debug().Interface("evt", evt).Msg("adding bonus score evt")
			g.addEventToHistory(evt)
			g.players[challengee].points += int(addPts) + addlBonus
		}

		if g.playing == pb.PlayState_WAITING_FOR_FINAL_PASS {
			g.playing = pb.PlayState_GAME_OVER
			g.history.PlayState = g.playing
			// Game is actually over now, after the failed challenge.
			// do calculations with the player on turn being the player who
			// didn't challenge, as this is a special event where the turn
			// did not _actually_ change.
			g.endOfGameCalcs(otherPlayer(g.onturn), true)
			g.AddFinalScoresToHistory()
		}

	}

	// Finally set the last words formed to nil.
	g.lastWordsFormed = nil
	g.turnnum = len(g.history.Events)
	return playLegal, err
}

func validateWords(lex lexicon.Lexicon, words []alphabet.MachineWord, variant Variant) []string {
	var illegalWords []string
	alph := lex.GetAlphabet()
	log.Debug().Interface("words", words).Msg("challenge-evt")
	for _, word := range words {
		var valid bool
		if variant == VarWordSmog {
			valid = lex.HasAnagram(word)
		} else {
			valid = lex.HasWord(word)
		}
		if !valid {
			illegalWords = append(illegalWords, word.UserVisible(alph))
		}
	}
	return illegalWords
}
