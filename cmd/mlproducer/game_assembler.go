// game_assembler.go - a feature-vector generator.

package main

import (
	"container/list"
	"log"
	"strings"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo/stats"
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
	queue   *list.List             // FIFO of [][]float32 ready to pop
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
		queue:   list.New(),
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

// FeedTurn ingests one ply, updates state, and maybe produces vectors.
func (ga *GameAssembler) FeedTurn(t Turn) {
	gw := ga.games[t.GameID]
	if gw == nil {
		gw = &gameWindow{}
		// The lexicon doesn't matter below; just choose any random one.
		rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18",
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
	stateVec := updateBoardAndExtractFeatures(gw, t) // TODO

	// 2) Push into sliding window.
	gw.turns = append(gw.turns, t)
	gw.states = append(gw.states, stateVec)

	// 3) Emit when window deep enough.
	if len(gw.states) > ga.horizon {
		vec := makeTrainingVector(gw.states[0], gw.states[ga.horizon]) // TODO
		ga.queue.PushBack(vec)

		// Slide window forward by dropping the oldest ply.
		gw.turns = gw.turns[1:]
		gw.states = gw.states[1:]
	}

	// 4) Detect end-of-game; flush leftovers then delete.
	if gw.game.Playing() == pb.PlayState_GAME_OVER {
		ga.flushRemainder(gw)
		delete(ga.games, t.GameID)
	}
}

// Ready reports whether at least one vector is waiting.
func (ga *GameAssembler) Ready() bool { return ga.queue.Len() > 0 }

// PopVector returns the next []float32 ready for output.
func (ga *GameAssembler) PopVector() []float32 {
	front := ga.queue.Front()
	vec := front.Value.([]float32)
	ga.queue.Remove(front)
	return vec
}

// ──────────────────────────────────────────────────────────────────────────────
// Flush any remaining positions when a game ends.
// ──────────────────────────────────────────────────────────────────────────────
func (ga *GameAssembler) flushRemainder(gw *gameWindow) {
	if len(gw.states) == 0 {
		return
	}

	lastIdx := len(gw.states) - 1

	// Emit vectors for every leftover ply i where i < lastIdx
	for i := 0; i < lastIdx; i++ {
		future := i + ga.horizon
		if future > lastIdx {
			future = lastIdx // clamp to final
		}
		vec := makeTrainingVector(gw.states[i], gw.states[future])
		ga.queue.PushBack(vec)
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Stubs you must fill in
// ──────────────────────────────────────────────────────────────────────────────

// Given current game window + turn, mutate board state and return features.
// Must return the *after-move* tensor for that ply.
func updateBoardAndExtractFeatures(gw *gameWindow, t Turn) []float32 {
	tp := stats.Normalize(t.Play) // normalize play string
	m, err := gw.game.ParseMove(gw.game.PlayerOnTurn(), false, strings.Fields(tp))
	if err != nil {
		log.Fatal("Failed to parse move: ", t, "error was", err)
	}
	// PlayMove plays the move, updates board, cross-checks, player on turn, scores, etc.
	// It also draws replenishment tiles from the bag. We don't want that for
	// training purposes.
	err = gw.game.PlayMove(m, false, 0)
	if err != nil {
		log.Fatal("Failed to play move: ", m.ShortDescription(), "error was", err)
	}
	// Undo rack replenishment.
	// Throw racks in and assign rack to player who just went; they should
	// only have their leave.
	gw.game.ThrowRacksIn()
	if t.Leave != "" {
		rack := tilemapping.RackFromString(t.Leave, gw.game.Alphabet())
		err = gw.game.SetRackForOnly(1-gw.game.PlayerOnTurn(), rack)
		if err != nil {
			log.Fatal("Failed to set rack for player: ", 1-gw.game.PlayerOnTurn(), "error was", err)
		}
	}
	// build up vector of features.

	tilePlanes := make([]float32, 26*225) // 26 planes for letters A-Z
	isBlankPlane := make([]float32, 225)  // 15x15 board tiles

	// 26 planes for board tiles
	// Each plane is a 15x15 grid, flattened to 225 elements.
	plane := gw.game.Board()
	for j := range 15 {
		for k := range 15 {
			tile := plane.GetLetter(j, k)
			if tile == 0 {
				continue // skip empty squares
			}
			unblanked := tile.Unblank()
			if unblanked != tile {
				// The tile was blanked
				isBlankPlane[j*15+k] = 1.0 // mark as blank
			} else {
				// The tile is a letter. The unblanked tile will range from
				// 1 to 26 for A-Z. (Fix later for other alphabets)
				ll := unblanked - 1 // convert to 0-based index
				if ll >= 26 {
					log.Fatalf("Invalid tile index %d for tile %d at position (%d, %d)", ll, tile, j, k)
				}
				// Set the corresponding plane for this letter
				tilePlanes[int(ll)*225+j*15+k] = 1.0 // this tile is in the letter plane
			}
		}
	}

	horCCs := make([]float32, 26*225)
	verCCs := make([]float32, 26*225)

	// 26 planes for horizontal cross-checks
	// 26 planes for vertical cross-checks
	for i := range 15 {
		for j := range 15 {
			hc := gw.game.Board().GetCrossSet(i, j, board.HorizontalDirection)
			vc := gw.game.Board().GetCrossSet(i, j, board.VerticalDirection)
			if hc == board.TrivialCrossSet && vc == board.TrivialCrossSet {
				// We skip trivial cross-checks to make this structure nicer to
				// the neural net. Trivial cross-checks are for empty squares where
				// technically every tile is allowed. But we really care about
				// empty squares that are right next to a tile (anchors, basically);
				// those would always have non-trivial cross-checks.
				continue
			}
			for t := range 26 { // A-Z are 1-26; change for other alphabets in future.
				// For each letter A-Z, we check if it is in the cross-check set.
				letter := tilemapping.MachineLetter(t + 1) // convert to 1-based index
				if hc.Allowed(letter) {
					horCCs[int(t)*225+i*15+j] = 1.0 // this tile is in the horizontal cross-check
				}
				if vc.Allowed(letter) {
					verCCs[int(t)*225+i*15+j] = 1.0 // this tile is in the vertical cross-check
				}
			}
		}
	}
	// Consider a single plane that has all the "anchors" or empty squares that are
	// adjacent to tiles in the future.

	// uncovered bonus square planes
	bonus2LPlane := make([]float32, 225) // 2x letter bonus
	bonus3LPlane := make([]float32, 225) // 3x letter bonus
	bonus2WPlane := make([]float32, 225) // 2x word bonus
	bonus3WPlane := make([]float32, 225) // 3x word bonus
	for i := range 15 {
		for j := range 15 {
			letter := gw.game.Board().GetLetter(i, j)
			if letter != 0 {
				// This is a letter tile, not an empty square.
				continue
			}
			bonus := gw.game.Board().GetBonus(i, j)

			if bonus == board.NoBonus {
				continue // no bonus here
			}
			if bonus == board.Bonus2LS {
				bonus2LPlane[i*15+j] = 1.0 // mark as 2x letter bonus
			} else if bonus == board.Bonus3LS {
				bonus3LPlane[i*15+j] = 1.0 // mark as 3x letter bonus
			} else if bonus == board.Bonus2WS {
				bonus2WPlane[i*15+j] = 1.0 // mark as 2x word bonus
			} else if bonus == board.Bonus3WS {
				bonus3WPlane[i*15+j] = 1.0 // mark as 3x word bonus
			} else {
				log.Fatalf("Unknown bonus type %d at position (%d, %d)", bonus, i, j)
			}
		}
	}

	// 1 vector for our rack leave (size = 27 tiles)
	// 1 vector for unseen tiles in the bag (size = 27 tiles)
	// scalar for score diff after making move
	// scalar for num tiles left in bag
	rackVector := make([]float32, 27)   // 26 letters + 1 for unseen
	unseenVector := make([]float32, 27) // 26 letters + 1 for unseen
	if t.Leave != "" {
		rack := tilemapping.RackFromString(t.Leave, gw.game.Alphabet())
		bag := gw.game.Bag().PeekMap()
		for i := range 27 {
			rackVector[i] = float32(rack.LetArr[i]) / 7.0 // normalize to 0-1 range
			// technically we can even divide by 12 here i think. (number of Es in the bag)
			unseenVector[i] = float32(bag[i]) / 100.0 // normalize to 0-1 range
		}
	}
	// Note the spread is our spread after making this move. The player on turn
	// switched after playing the move, so that's why we do 1 - gw.game.PlayerOnTurn().
	normalizedSpread := float32(gw.game.SpreadFor(1 - gw.game.PlayerOnTurn())) // update spread for opponent
	if normalizedSpread < -300 {
		normalizedSpread = -300
	} else if normalizedSpread > 300 {
		normalizedSpread = 300
	}
	normalizedSpread /= 300.0                                         // normalize to -1 to 1 range
	tilesRemaining := float32(gw.game.Bag().TilesRemaining()) / 100.0 // normalize to 0-1 range
	// Concatenate all feature planes and vectors into a single flat []float32.
	features := []float32{}
	features = append(features, tilePlanes...)
	features = append(features, isBlankPlane...)
	features = append(features, horCCs...)
	features = append(features, verCCs...)
	features = append(features, bonus2LPlane...)
	features = append(features, bonus3LPlane...)
	features = append(features, bonus2WPlane...)
	features = append(features, bonus3WPlane...)
	features = append(features, rackVector...)
	features = append(features, unseenVector...)
	features = append(features, tilesRemaining, normalizedSpread)
	return features

	// XXX: figure out board transposition
}

// Build final training vector from state at ply t and ply t+horizon.
func makeTrainingVector(stateNow, stateFuture []float32) []float32 {

	// We want our final model to predict spread change after horizon plies.
	// So we take the difference between the two states.
	vec := make([]float32, len(stateNow))
	copy(vec, stateNow) // copy current state
	// The last element of the original vector is the spread after the move.
	// So we compute the difference between the future and current spread.
	vec = append(vec, stateFuture[len(stateFuture)-1]-stateNow[len(stateNow)-1])

	return vec
}
