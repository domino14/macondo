// game_assembler.go - a feature-vector generator.

package main

import (
	"strings"

	"github.com/cespare/xxhash"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/turnplayer"
	"github.com/domino14/word-golib/tilemapping"
)

var DefaultConfig = config.DefaultConfig()

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

// GameAssembler emits training vectors once it has look-ahead data for a ply.
type GameAssembler struct {
	horizon int                    // #plies to look ahead
	games   map[string]*gameWindow // live games by GameID
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

// Sliding window of recent positions for one game.
type gameWindow struct {
	turns  []Turn      // length ≤ horizon+1
	states [][]float32 // feature vectors after each ply (same len)
	game   turnplayer.BaseTurnPlayer
}

// ──────────────────────────────────────────────────────────────────────────────
// Constructor
// ──────────────────────────────────────────────────────────────────────────────

func NewGameAssembler(horizon int) *GameAssembler {
	return &GameAssembler{
		horizon: horizon,
		games:   make(map[string]*gameWindow),
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Core API
// ──────────────────────────────────────────────────────────────────────────────

func otherPlayer(playerID string) string {
	if playerID == "p1" {
		return "p2"
	}
	return "p1"
}

func shouldTranspose(id string) bool {
	hash := xxhash.Sum64String(id)
	return hash%2 == 0
}

// FeedTurn ingests one ply, updates state, and maybe produces vectors.
func (ga *GameAssembler) FeedTurn(t Turn) [][]float32 {
	outVecs := make([][]float32, 0)
	gw := ga.games[t.GameID]
	if gw == nil {
		gw = &gameWindow{}
		// The lexicon doesn't matter below; just choose any random one.
		rules, err := game.NewBasicGameRules(DefaultConfig, "NWL23",
			board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
		if err != nil {
			panic(err)
		}
		tp, err := turnplayer.BaseTurnPlayerFromRules(
			&turnplayer.GameOptions{
				Variant:         game.VarClassic,
				BoardLayoutName: board.CrosswordGameLayout},
			[]*pb.PlayerInfo{
				{Nickname: t.PlayerID, RealName: t.PlayerID},
				{Nickname: otherPlayer(t.PlayerID), RealName: otherPlayer(t.PlayerID)},
			}, rules)
		if err != nil {
			panic(err)
		}
		gw.game = *tp
		ga.games[t.GameID] = gw
	}

	// 1) Apply move, update board/racks, compute after-move features.
	stateVec := updateBoardAndExtractFeatures(gw, t)

	// 2) Push into sliding window.
	gw.turns = append(gw.turns, t)
	gw.states = append(gw.states, stateVec)

	// 3) Emit when window deep enough.
	if len(gw.states) > ga.horizon {
		vec := makeTrainingVector(gw.states[0], gw.states[ga.horizon])
		outVecs = append(outVecs, vec)

		// Slide window forward by dropping the oldest ply.
		gw.turns = gw.turns[1:]
		gw.states = gw.states[1:]
	}
	// log.Info().Msgf("Game %s: fed turn %s, now have %d turns in window",
	// 	t.GameID, t.Play, len(gw.turns))
	// 4) Detect end-of-game; flush leftovers then delete.
	if gw.game.Playing() == pb.PlayState_GAME_OVER ||
		gw.game.Playing() == pb.PlayState_WAITING_FOR_FINAL_PASS {

		if gw.game.Playing() == pb.PlayState_WAITING_FOR_FINAL_PASS {
			passMove := move.NewPassMove(gw.game.RackFor(gw.game.PlayerOnTurn()).TilesOn(),
				gw.game.Alphabet())
			err := gw.game.PlayMove(passMove, false, 0)
			if err != nil {
				log.Fatal().Msgf("Failed to play final pass move: %v, error was %v", passMove, err)
			}
		}
		if gw.game.Playing() != pb.PlayState_GAME_OVER {
			log.Fatal().Msgf("Game %s is not over, but we got a game end signal. Playing state: %s",
				t.GameID, gw.game.Playing())
		}

		outVecs = append(outVecs, ga.flushRemainder(gw)...)
		delete(ga.games, t.GameID)
	}
	return outVecs
}

// ──────────────────────────────────────────────────────────────────────────────
// Flush any remaining positions when a game ends.
// ──────────────────────────────────────────────────────────────────────────────
func (ga *GameAssembler) flushRemainder(gw *gameWindow) [][]float32 {
	if len(gw.states) == 0 {
		return nil
	}
	outVecs := make([][]float32, 0)
	lastIdx := len(gw.states) - 1

	// Emit vectors for every leftover ply i where i < lastIdx
	for i := 0; i < lastIdx; i++ {
		future := i + ga.horizon
		if future > lastIdx {
			future = lastIdx // clamp to final
		}
		vec := makeTrainingVector(gw.states[i], gw.states[future])
		outVecs = append(outVecs, vec)
	}
	return outVecs
}

// Given current game window + turn, mutate board state and return features.
// Must return the *after-move* tensor for that ply.
func updateBoardAndExtractFeatures(gw *gameWindow, t Turn) []float32 {
	transpose := shouldTranspose(t.GameID)
	tp := stats.Normalize(t.Play) // normalize play string
	gw.game.ThrowRacksIn()
	err := gw.game.SetRackFor(gw.game.PlayerOnTurn(), tilemapping.RackFromString(t.Rack, gw.game.Alphabet()))
	if err != nil {
		log.Fatal().Msgf("Failed to set rack for player: %d, error was %v", gw.game.PlayerOnTurn(), err)
	}

	m, err := gw.game.ParseMove(gw.game.PlayerOnTurn(), false, strings.Fields(tp), transpose)
	if err != nil {
		log.Fatal().Msgf("Failed to parse move: %v, error was %v", t, err)
	}
	// PlayMove plays the move, updates board, cross-checks, player on turn, scores, etc.
	// It also draws replenishment tiles from the bag. We don't want that for
	// training purposes.

	err = gw.game.PlayMove(m, false, 0)
	if err != nil {
		log.Fatal().Msgf("Failed to play move: %s, error was %v", m.ShortDescription(), err)
	}
	// Undo rack replenishment.
	// Throw racks in and assign rack to player who just went; they should
	// only have their leave.
	gw.game.ThrowRacksIn()

	// switch player on turn back to the one who just played the move.
	gw.game.SetPlayerOnTurn(1 - gw.game.PlayerOnTurn())

	if t.Leave != "" {
		rack := tilemapping.RackFromString(t.Leave, gw.game.Alphabet())
		err = gw.game.SetRackForOnly(gw.game.PlayerOnTurn(), rack)
		if err != nil {
			log.Fatal().Msgf("Failed to set rack for player: %d, error was %v", gw.game.PlayerOnTurn(), err)
		}
	}
	vecPtr, err := gw.game.BuildMLVector(t.Score, t.Equity)
	if err != nil {
		log.Fatal().Msgf("Failed to build ML vector: %v", err)
	}
	vec := *vecPtr
	// We temporarily encode the player whose spread this is for since we don't
	// have that data anywhere else.
	spreadFor := gw.game.PlayerOnTurn()
	unNormalizedSpread := gw.game.SpreadFor(spreadFor)
	vec[len(vec)-1] = float32(unNormalizedSpread) // update spread for player on turn
	vec = append(vec, float32(spreadFor))         // append player index for spread

	// Undo the switch of the player on turn.
	gw.game.SetPlayerOnTurn(1 - gw.game.PlayerOnTurn())
	game.MLVectorPool.Put(vecPtr)
	return vec
}

// Build final training vector from state at ply t and ply t+horizon.
func makeTrainingVector(stateNow, stateFuture []float32) []float32 {
	if len(stateNow) != len(stateFuture) {
		log.Fatal().Msgf("State vectors must be of the same length, got %d and %d", len(stateNow), len(stateFuture))
	}
	// We want our final model to predict spread change after horizon plies.
	// So we take the difference between the two states.
	spreadForNow := int(stateNow[len(stateNow)-1])          // last element is the player index this spread is for.
	spreadForFuture := int(stateFuture[len(stateFuture)-1]) // same for future state

	vec := make([]float32, len(stateNow)-1)
	copy(vec, stateNow[:len(stateNow)-1]) // copy current state minus the last element

	// Get the actual spread values for the relevant player.
	futureSpread := stateFuture[len(stateFuture)-2]
	// nowSpread := stateNow[len(stateNow)-2]

	if spreadForFuture != spreadForNow {
		// The two players are different. We want the future spread to be
		// for the same player as the current spread.

		futureSpread = -futureSpread // flip the sign of the future spread
	}

	// spreadDiff := futureSpread - nowSpread
	// Let's try to just get a win or loss signal. Note we are looking ahead
	// N Plies and it's not necessarily who won the entire game. We can
	// train that way later.
	win := 0.0
	if futureSpread > 0 {
		win = 1.0
	} else if futureSpread < 0 {
		win = -1.0
	}

	// log.Info().Msgf("future state spread %f, now spread %f, spreadForNow: %d, spreadForFuture: %d, Spread diff: %f, normalized: %f",
	// 	stateFuture[len(stateFuture)-2], stateNow[len(stateNow)-2],
	// 	spreadForNow, spreadForFuture,
	// 	spreadDiff, normalizeSpread(spreadDiff))
	// replace previous element with normalized spread of this move only
	vec[len(vec)-1] = game.NormalizeSpreadForML(vec[len(vec)-1])
	vec = append(vec, float32(win))

	return vec
}
