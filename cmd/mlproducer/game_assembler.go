// game_assembler.go - a feature-vector generator.

package main

import (
	"math"
	"strings"

	"github.com/cespare/xxhash"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/turnplayer"
)

var DefaultConfig = config.DefaultConfig()

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

// GameAssembler emits training vectors once it has look-ahead data for a ply.
type GameAssembler struct {
	horizon int                               // #plies to look ahead
	games   map[string]*gameWindow            // live games by GameID
	eqCalc  *equity.ExhaustiveLeaveCalculator // equity calculator for leave values
	winpcts [][]float32
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

// Sliding window of recent positions for one game.
type gameWindow struct {
	turns      []Turn // length ≤ horizon+1
	moves      []*move.Move
	states     []*[]float32 // feature vectors after each ply (same len)
	spreadsFor []int
	game       turnplayer.BaseTurnPlayer
}

// ──────────────────────────────────────────────────────────────────────────────
// Constructor
// ──────────────────────────────────────────────────────────────────────────────

func NewGameAssembler(horizon int) *GameAssembler {
	els, err := equity.NewExhaustiveLeaveCalculator("NWL23", DefaultConfig, "")
	if err != nil {
		log.Fatal().Msgf("Failed to create exhaustive leave calculator: %v", err)
	}
	// some hardcoded stuff here:
	winpct, err := cache.Load(DefaultConfig.WGLConfig(), "winpctfile:NWL20:winpct.csv", equity.WinPCTLoadFunc)
	if err != nil {
		panic(err)
	}
	var ok bool
	winPcts, ok := winpct.([][]float32)
	if !ok {
		panic("win percentages not correct type")
	}
	return &GameAssembler{
		horizon: horizon,
		games:   make(map[string]*gameWindow),
		eqCalc:  els,
		winpcts: winPcts,
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

type outputVector struct {
	features    *[]float32
	predictions []float32
}

// FeedTurn ingests one ply, updates state, and maybe produces vectors.
func (ga *GameAssembler) FeedTurn(t Turn) []outputVector {
	outVecs := []outputVector{}
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
	var lastMove *move.Move
	if len(gw.moves) > 0 {
		lastMove = gw.moves[len(gw.moves)-1]
	}
	// 1) Apply move, update board/racks, compute after-move features.
	m, stateVec, spreadFor := updateBoardAndExtractFeatures(gw, t, ga.eqCalc, lastMove)

	// 2) Push into sliding window.
	gw.turns = append(gw.turns, t)
	gw.moves = append(gw.moves, m)
	gw.states = append(gw.states, stateVec)
	gw.spreadsFor = append(gw.spreadsFor, spreadFor)

	// 3) Emit when window deep enough.
	if len(gw.states) > ga.horizon {
		vec := makeTrainingVector(ga, gw, gw.states[0], gw.states[ga.horizon], gw.spreadsFor[0], gw.spreadsFor[ga.horizon])
		outVecs = append(outVecs, vec)

		// Slide window forward by dropping the oldest ply.
		gw.turns = gw.turns[1:]
		gw.spreadsFor = gw.spreadsFor[1:]
		// game.MLVectorPool.Put(gw.states[0])
		gw.states = gw.states[1:]
		gw.moves = gw.moves[1:]
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
func (ga *GameAssembler) flushRemainder(gw *gameWindow) []outputVector {
	if len(gw.states) == 0 {
		return nil
	}
	outVecs := make([]outputVector, 0)
	lastIdx := len(gw.states) - 1

	// Emit vectors for every leftover ply i where i < lastIdx
	for i := 0; i < lastIdx; i++ {
		future := i + ga.horizon
		if future > lastIdx {
			future = lastIdx // clamp to final
		}
		vec := makeTrainingVector(ga, gw, gw.states[i], gw.states[future], gw.spreadsFor[i], gw.spreadsFor[future])
		outVecs = append(outVecs, vec)
	}
	game.MLVectorPool.Put(gw.states[lastIdx]) // return last state to pool
	return outVecs
}

// Given current game window + turn, mutate board state and return features.
// Must return the *after-move* tensor for that ply.
func updateBoardAndExtractFeatures(gw *gameWindow, t Turn, eqCalc *equity.ExhaustiveLeaveCalculator,
	lastMove *move.Move) (*move.Move, *[]float32, int) {
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
	leaveVal := 0.0
	err = gw.game.PlayMove(m, false, 0)
	if err != nil {
		log.Fatal().Msgf("Failed to play move: %s, error was %v", m.ShortDescription(), err)
	}
	// Undo rack replenishment.
	// Throw racks in and assign rack to player who just went; they should
	// only have their leave.
	gw.game.ThrowRacksIn()
	if m.Action() == move.MoveTypeExchange {
		// If it's an exchange, we want to treat this a bit differently.
		// We don't actually want to throw the player's exchanged tiles into
		// the bag, as we wish to evaluate what the bag would look like
		// at the moment they decide to exchange.
		// So we need to remove the exchanged tiles from the bag, but
		// re-add them after building the vector so we don't corrupt the game.
		err = gw.game.Bag().RemoveTiles(m.Tiles())
		if err != nil {
			log.Fatal().Msgf("Failed to remove exchanged tiles from bag: %v, error was %v", m.Tiles(), err)
		}
	}

	// switch player on turn back to the one who just played the move.
	gw.game.SetPlayerOnTurn(1 - gw.game.PlayerOnTurn())
	rack := tilemapping.RackFromString(t.Leave, gw.game.Alphabet())
	err = gw.game.SetRackForOnly(gw.game.PlayerOnTurn(), rack)
	if err != nil {
		log.Fatal().Msgf("Failed to set rack for player: %d, error was %v", gw.game.PlayerOnTurn(), err)
	}
	leaveVal = eqCalc.LeaveValue(rack.TilesOn())

	vecPtr, err := gw.game.BuildMLVector(m, leaveVal, lastMove)
	if err != nil {
		log.Fatal().Msgf("Failed to build ML vector: %v", err)
	}

	if m.Action() == move.MoveTypeExchange {
		// If it's an exchange, we need to re-add the exchanged tiles to the bag.
		// This is so that the game state is not corrupted for the next player.
		gw.game.Bag().PutBack(m.Tiles())
	}

	// vec := *vecPtr
	spreadFor := gw.game.PlayerOnTurn()
	unNormalizedSpread := gw.game.SpreadFor(spreadFor)
	(*vecPtr)[len(*vecPtr)-1] = float32(unNormalizedSpread) // update spread for player on turn

	// Undo the switch of the player on turn.
	gw.game.SetPlayerOnTurn(1 - gw.game.PlayerOnTurn())

	// 	game.MLVectorPool.Put(vecPtr) XXX put it back elsewhere.
	return m, vecPtr, spreadFor
}

// Build final training vector from state at ply t and ply t+horizon.
func makeTrainingVector(ga *GameAssembler, gw *gameWindow, stateNow, stateFuture *[]float32,
	spreadForNow, spreadForFuture int) outputVector {
	if len(*stateNow) != len(*stateFuture) {
		log.Fatal().Msgf("State vectors must be of the same length, got %d and %d", len(*stateNow), len(*stateFuture))
	}

	// Get the actual spread values for the relevant player.
	futureSpread := (*stateFuture)[len(*stateFuture)-1]
	// nowSpread := stateNow[len(stateNow)-2]
	bagRemaining := int(math.Round(float64((*stateFuture)[len(*stateFuture)-2] * 100.0))) // second to last element is bag remaining

	if spreadForFuture != spreadForNow {
		// The two players are different. We want the future spread to be
		// for the same player as the current spread.

		futureSpread = -futureSpread // flip the sign of the future spread
	}

	// spreadDiff := futureSpread - nowSpread
	// Let's try to just get a win or loss signal. Note we are looking ahead
	// N Plies and it's not necessarily who won the entire game. We can
	// train that way later.
	win := float32(0.0)
	bogowin := float32(0.0)
	if gw.game.Playing() == pb.PlayState_GAME_OVER {
		switch {
		case futureSpread > 0:
			win = 1.0
			bogowin = 1.0 // we won, so BOGO win is also 100%
		case futureSpread < 0:
			win = -1.0
			bogowin = 0.0 // we lost, so BOGO win is 0%
		case futureSpread == 0:
			win = 0.0
			bogowin = 0.5 // we can consider this a draw, so 50% chance of winning
		}
	} else {
		// We are not at the end of the game, so calculate bogowin (winpct lookup table)
		// percentage based on the future spread.
		if futureSpread > equity.MaxRepresentedWinSpread {
			futureSpread = equity.MaxRepresentedWinSpread
		} else if futureSpread < -equity.MaxRepresentedWinSpread {
			futureSpread = -equity.MaxRepresentedWinSpread
		}
		if bagRemaining >= len(ga.winpcts) || bagRemaining < 0 {
			log.Fatal().Msgf("Bag remaining %d is out of bounds for winpcts", bagRemaining)
		}
		bogowin = ga.winpcts[int(equity.MaxRepresentedWinSpread-futureSpread)][bagRemaining]
		if futureSpread > 0 {
			win = 1.0 // win is after N plies
		} else if futureSpread < 0 {
			win = -1.0
		}
	}

	// log.Info().Msgf("future state spread %f, now spread %f, spreadForNow: %d, spreadForFuture: %d, Spread diff: %f, normalized: %f",
	// 	stateFuture[len(stateFuture)-2], stateNow[len(stateNow)-2],
	// 	spreadForNow, spreadForFuture,
	// 	spreadDiff, normalizeSpread(spreadDiff))
	// replace previous element with normalized spread of this move only
	(*stateNow)[len(*stateNow)-1] = game.NormalizeSpreadForML((*stateNow)[len((*stateNow))-1])
	ov := outputVector{
		features:    stateNow,
		predictions: make([]float32, 1),
	}
	ov.predictions[0] = win // 1 if we win, -1 if we lose, 0 if draw
	_ = bogowin             // ignore bogowin for now, it does badly.

	return ov
}
