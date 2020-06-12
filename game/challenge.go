package game

import (
	"errors"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
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
func (g *Game) ChallengeEvent(addlBonus int) (bool, error) {
	if len(g.history.Turns) == 0 {
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
	illegalWords := validateWords(g.gaddag, g.lastWordsFormed)
	playLegal := len(illegalWords) == 0

	lastTurn := g.history.Turns[len(g.history.Turns)-1]
	cumeScoreBeforeChallenge := lastTurn.Events[0].Cumulative

	challengee := otherPlayer(g.onturn)

	if !playLegal {
		offBoardEvent := &pb.GameEvent{
			Nickname:   lastTurn.Events[0].Nickname,
			Type:       pb.GameEvent_PHONY_TILES_RETURNED,
			LostScore:  -lastTurn.Events[0].Score,
			Cumulative: cumeScoreBeforeChallenge - lastTurn.Events[0].Score,
			Rack:       lastTurn.Events[0].Rack,
		}

		// the play comes off the board.
		// make it so the events are ONLY tile placement move
		// and phony tiles returned (so remove any out-play bonus if it exists)
		lastTurn.Events = []*pb.GameEvent{lastTurn.Events[0], offBoardEvent}
		// Unplay the last move to restore everything as it was board-wise
		// (and un-end the game if it had ended)
		g.UnplayLastMove()
		// We must also set the last known rack of the challengee back to
		// their rack before they played the phony.
		g.history.LastKnownRacks[challengee] = lastTurn.Events[0].Rack

		// Note that if backup mode is InteractiveGameplayMode, which it should be,
		// we do not back up the turn number. So restoring it doesn't change
		// the turn number or whose turn it is; unplay only copies the
		// needed variables.
		// and we must add one to scoreless turns:
		g.scorelessTurns++
		g.handleConsecutiveScorelessTurns(true, lastTurn)

		// Finally, let's re-shuffle the bag. This is so we don't give the
		// player who played the phony knowledge about the next few tiles in the bag.
		g.bag.Shuffle()
	} else {
		addPts := int32(0)
		shouldAddPts := false

		bonusScoreEvent := func(bonus int32) *pb.GameEvent {
			return &pb.GameEvent{
				Nickname:   lastTurn.Events[0].Nickname,
				Type:       pb.GameEvent_CHALLENGE_BONUS,
				Bonus:      bonus + int32(addlBonus),
				Cumulative: cumeScoreBeforeChallenge + bonus,
			}
		}

		switch g.history.ChallengeRule {
		case pb.ChallengeRule_DOUBLE:
			// This "draconian" American system makes it so someone always loses
			// their turn.
			// challenger was wrong. They lose their turn.
			g.PlayMove(move.NewUnsuccessfulChallengePassMove(
				g.players[g.onturn].rack.TilesOn(), g.alph), true)

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
			lastTurn.Events = append(lastTurn.Events, bonusScoreEvent(addPts))
			g.players[challengee].points += int(addPts)
		}

		if g.playing == pb.PlayState_WAITING_FOR_FINAL_PASS {
			g.playing = pb.PlayState_GAME_OVER
			g.history.PlayState = g.playing
			// Game is actually over now, after the failed challenge.
			// do calculations with the player on turn being the player who
			// didn't challenge, as this is a special event where the turn
			// did not _actually_ change.
			g.endOfGameCalcs(otherPlayer(g.onturn), lastTurn, true)
		}

	}

	// Finally set the last words formed to nil.
	g.lastWordsFormed = nil
	return playLegal, nil
}
