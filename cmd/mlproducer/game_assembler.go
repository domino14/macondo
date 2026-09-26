// game_assembler.go - a feature-vector generator.

package main

import (
	"context"
	"errors"
	"math/rand"
	"strings"
	"time"

	"github.com/cespare/xxhash"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/turnplayer"
)

var DefaultConfig = config.DefaultConfig()

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

// GameAssembler emits training vectors once it has look-ahead data for a ply.
type GameAssembler struct {
	horizon        int                               // #plies to look ahead
	games          map[string]*gameWindow            // live games by GameID
	eqCalc         *equity.ExhaustiveLeaveCalculator // equity calculator for leave values
	winpcts        [][]float32
	gamesProcessed int64

	// Rollout labeling (nil = table lookup after `horizon` real plies).
	labeler *RolloutLabeler
	// valueFromResult makes the value target the mover's real result
	// (the same signal as TargetWDL) and the spread target the spread
	// change to the end of the game, instead of the horizon lookup.
	valueFromResult bool
	// endgamePlies > 0 labels an emitted position whose bag was already
	// empty by a quick endgame search (that many plies, greedy playout at
	// the leaves) from the opponent's reply, instead of by how the logged
	// game happened to play out.
	endgamePlies int
	kwg          *kwg.KWG
	solved       int64
	// A search that runs past this is abandoned and the position keeps
	// its logged label; 2-ply searches take milliseconds, so this only
	// guards against degenerate positions.
	endgameTimeout  time.Duration
	endgameTimeouts int64
	// Which positions to label and emit. With pickMax > 0, one turn per
	// game is drawn uniformly from 1..pickMax up front and only that
	// position gets a feature vector (and a rollout label); otherwise every
	// position is emitted, subject to `sample` when rollout-labeling.
	pickMax   int
	fixedPick int // tests: the turn to pick for every game instead of drawing one
	sample    float64
	labeled   int64
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

// ply is one turn of a game as replayed, with whatever the labels need.
type ply struct {
	turn   Turn
	move   *move.Move
	state  *[]float32 // feature vector after the move; nil when not built
	mover  int        // player index who moved
	spread float32    // raw spread after the move, from the mover's side
	bag    int        // tiles unseen after the move (bag + opponent's rack)
	label  *rolloutLabel
	solved *rolloutLabel // endgame-search label (value, spread change), mover's side
}

// Sliding window of recent positions for one game.
type gameWindow struct {
	plies []ply // length ≤ horizon+1
	// Every move of the game so far. The feature vector's
	// turns-since-opponent-bingo counts back over this, and the bot at play
	// time has the whole game, so the window alone would cap it at 3.
	history []*move.Move
	game    turnplayer.BaseTurnPlayer
	ai      *aiturnplayer.AIStaticTurnPlayer // static best play for rollouts
	pick    int                              // the one turn to emit, or 0 for all
	// Vectors whose horizon label is done but whose final-result label
	// (win/draw/loss for the mover) needs the game to end first.
	pending []outputVector
	es      *negamax.Solver // endgame search, created on first use
}

// ──────────────────────────────────────────────────────────────────────────────
// Constructor
// ──────────────────────────────────────────────────────────────────────────────

// NewGameAssembler builds an assembler. With a non-nil `shared`, positions
// are rollout-labeled (`plies` plies, `rollouts` samples) with probability
// `sample`; otherwise the label is the table lookup after `horizon` plies.
func NewGameAssembler(horizon int, shared *RolloutShared, plies, rollouts int, sample float64) *GameAssembler {
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
	ga := &GameAssembler{
		horizon: horizon,
		games:   make(map[string]*gameWindow),
		eqCalc:  els,
		winpcts: winPcts,
		sample:  sample,
	}
	if shared != nil {
		ga.labeler = NewRolloutLabeler(shared, plies, rollouts, els)
	}
	return ga
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

// Training targets, in the order they are written after the features.
const (
	TargetValue    = iota // bogowin after the horizon (or the real result), scaled to [-1, 1]
	TargetSpread          // spread change over the horizon (or to the end), tanh-scaled
	TargetWDL             // final game result for the mover: -1, 0, 1
	TargetOppBingo        // opponent bingos on their next turn: 0 or 1
	TargetOppScore        // opponent's next score / 300
	NumTargets
)

// Spatial targets: four 15x15 planes written after the scalar targets, one
// float per square, in this order. They are training-only signal for the
// trunk (the bot never reads them): where each player's next move lands, and
// the same conjoined with that player going on to win the game.
const (
	SpatialOppNext  = iota // squares the opponent's next move covers
	SpatialSelfNext        // squares the mover's own next move covers
	SpatialOppWin          // SpatialOppNext, all zeros unless the opponent won
	SpatialSelfWin         // SpatialSelfNext, all zeros unless the mover won
	NumSpatialTargets
)

const SpatialCells = 15 * 15

// NumPredictions is the length of outputVector.predictions: the scalar
// targets, then the spatial planes.
const NumPredictions = NumTargets + NumSpatialTargets*SpatialCells

// spatialPlane is plane k of a predictions slice.
func spatialPlane(preds []float32, k int) []float32 {
	off := NumTargets + k*SpatialCells
	return preds[off : off+SpatialCells]
}

// markPlacement sets to 1 every square that m places a tile on. Exchanges,
// passes, and played-through tiles mark nothing.
func markPlacement(plane []float32, m *move.Move) {
	if m == nil || m.Action() != move.MoveTypePlay {
		return
	}
	r, c, vertical := m.CoordsAndVertical()
	ri, ci := 0, 1
	if vertical {
		ri, ci = 1, 0
	}
	for i, t := range m.Tiles() {
		if t == 0 {
			continue // played through
		}
		curR, curC := r+i*ri, c+i*ci
		if curR < 0 || curR >= 15 || curC < 0 || curC >= 15 {
			log.Fatal().Msgf("placement out of bounds at (%d, %d) for %s", curR, curC, m.ShortDescription())
		}
		plane[curR*15+curC] = 1
	}
}

type outputVector struct {
	features    *[]float32
	predictions []float32 // indexed by the Target* constants
	gameID      string
	turn        int
	mover       int
	spreadNow   float32       // raw spread at the position, mover's side
	label       *rolloutLabel // set when rollout-labeled, for the labels file
	solved      *rolloutLabel // set when endgame-search-labeled
}

// FeedTurn ingests one ply, updates state, and maybe produces vectors.
func (ga *GameAssembler) FeedTurn(t Turn) []outputVector {
	gw := ga.games[t.GameID]
	if gw == nil {
		gw = ga.newGameWindow(t)
		ga.games[t.GameID] = gw
	}

	// 1) Apply move, update board/racks, compute after-move features.
	p := ga.updateBoardAndExtractFeatures(gw, t)

	// 2) Push into sliding window (and the full history).
	gw.plies = append(gw.plies, p)
	gw.history = append(gw.history, p.move)

	// 3) Emit when window deep enough.
	if len(gw.plies) > ga.horizon {
		if vec, ok := ga.makeTrainingVector(gw, 0, 1, ga.horizon); ok {
			gw.pending = append(gw.pending, vec)
		}
		// Slide window forward by dropping the oldest ply.
		gw.plies = gw.plies[1:]
	}

	// 4) Detect end-of-game; flush leftovers then delete.
	var out []outputVector
	if gw.game.Playing() == pb.PlayState_GAME_OVER ||
		gw.game.Playing() == pb.PlayState_WAITING_FOR_FINAL_PASS {

		if gw.game.Playing() == pb.PlayState_WAITING_FOR_FINAL_PASS {
			// The racks were thrown in after the play-out; the bag now holds
			// exactly the passing player's tiles. Give them back so the
			// end-of-game bonus is computed from a real rack.
			setRackFromBag(gw, gw.game.PlayerOnTurn())
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

		ga.flushRemainder(gw)
		out = ga.release(gw)
		delete(ga.games, t.GameID)
		ga.gamesProcessed++
	}
	return out
}

// newGameWindow starts the replay of a game whose first turn is t: a fresh
// game between t's player and the other one, plus whatever the labelers
// need per game.
func (ga *GameAssembler) newGameWindow(t Turn) *gameWindow {
	gw := &gameWindow{}
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
	if ga.labeler != nil {
		gw.ai, err = aiturnplayer.NewAIStaticTurnPlayerFromGame(gw.game.Game, DefaultConfig, ga.labeler.calcs)
		if err != nil {
			panic(err)
		}
		gw.game.SetStateStackLength(ga.labeler.StackLength())
	}
	if ga.fixedPick > 0 {
		gw.pick = ga.fixedPick
	} else if ga.pickMax > 0 {
		gw.pick = 1 + rand.Intn(ga.pickMax)
	}
	return gw
}

// release stamps every held vector with the final result from its mover's
// side (and, with valueFromResult, makes that the value target) and hands
// them all back. Only valid once the game is over.
func (ga *GameAssembler) release(gw *gameWindow) []outputVector {
	for i := range gw.pending {
		vec := &gw.pending[i]
		final := float32(gw.game.SpreadFor(vec.mover))
		var wdl float32
		switch {
		case final > 0:
			wdl = 1.0
		case final < 0:
			wdl = -1.0
		}
		vec.predictions[TargetWDL] = wdl
		// The win-conjunction planes: the next-move planes, kept only for
		// the player who went on to win (a draw leaves both empty).
		if wdl < 0 {
			copy(spatialPlane(vec.predictions, SpatialOppWin), spatialPlane(vec.predictions, SpatialOppNext))
		} else if wdl > 0 {
			copy(spatialPlane(vec.predictions, SpatialSelfWin), spatialPlane(vec.predictions, SpatialSelfNext))
		}
		if vec.solved != nil {
			vec.predictions[TargetValue] = vec.solved.value
			vec.predictions[TargetSpread] = game.NormalizeSpreadForML(vec.solved.spread)
		} else if ga.valueFromResult {
			vec.predictions[TargetValue] = wdl
			vec.predictions[TargetSpread] = game.NormalizeSpreadForML(final - vec.spreadNow)
		}
	}
	out := gw.pending
	gw.pending = nil
	return out
}

// setRackFromBag gives playerIdx every tile left in the bag. When the real
// bag is empty, the replay's bag (racks thrown in, the other player's rack
// set) holds exactly this player's tiles. Whatever rack they had goes back
// in first; SetRackFor deals the non-mover a random one.
func setRackFromBag(gw *gameWindow, playerIdx int) {
	gw.game.ThrowRacksInFor(playerIdx)
	tiles := gw.game.Bag().Peek()
	rack := tilemapping.NewRack(gw.game.Alphabet())
	rack.Set(tiles)
	if err := gw.game.SetRackForOnly(playerIdx, rack); err != nil {
		log.Fatal().Msgf("Failed to set rack from bag for player %d: %v", playerIdx, err)
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Flush any remaining positions when a game ends.
// ──────────────────────────────────────────────────────────────────────────────
func (ga *GameAssembler) flushRemainder(gw *gameWindow) {
	if len(gw.plies) == 0 {
		return
	}
	lastIdx := len(gw.plies) - 1

	// Emit vectors for every leftover ply i where i < lastIdx
	for i := 0; i < lastIdx; i++ {
		future := i + ga.horizon
		if future > lastIdx {
			future = lastIdx // clamp to final
		}
		next := i + 1
		if next > lastIdx {
			next = lastIdx // clamp to final
		}
		if vec, ok := ga.makeTrainingVector(gw, i, next, future); ok {
			gw.pending = append(gw.pending, vec)
		}
	}
	if gw.plies[lastIdx].state != nil {
		game.MLVectorPool.Put(gw.plies[lastIdx].state) // return last state to pool
	}
}

// moveHistory is every move of the game so far, oldest first.
func (gw *gameWindow) moveHistory() []*move.Move {
	return gw.history
}

// Given current game window + turn, mutate board state and return the ply.
// The feature vector is the *after-move* tensor for that ply; it is only
// built for positions that can be emitted.
func (ga *GameAssembler) updateBoardAndExtractFeatures(gw *gameWindow, t Turn) ply {
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
	if t.TilesRemaining == 0 {
		// The bag is really empty, so everything still in the replay's bag
		// is the opponent's rack. PlayMove needs it to score the end of the
		// game (2x the unplayed tiles) correctly if this move plays out.
		setRackFromBag(gw, 1-gw.game.PlayerOnTurn())
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
	mover := gw.game.PlayerOnTurn()
	rack := tilemapping.RackFromString(t.Leave, gw.game.Alphabet())
	err = gw.game.SetRackForOnly(mover, rack)
	if err != nil {
		log.Fatal().Msgf("Failed to set rack for player: %d, error was %v", mover, err)
	}

	p := ply{
		turn:   t,
		move:   m,
		mover:  mover,
		spread: float32(gw.game.SpreadFor(mover)),
		bag:    gw.game.Bag().TilesRemaining(),
	}
	// With one position per game, a drawn turn that lands in the endgame
	// (bag already empty) emits nothing: the bot hands the endgame to the
	// solver and never consults the net there. (fixedPick, the test hook,
	// bypasses this so endgame labeling can be tested.)
	wanted := gw.pick == 0 || (gw.pick == t.TurnNumber && (t.TilesRemaining > 0 || ga.fixedPick > 0))
	if wanted {
		leaveVal := ga.eqCalc.LeaveValue(rack.TilesOn())
		p.state, err = gw.game.BuildMLVector(m, leaveVal, gw.moveHistory())
		if err != nil {
			log.Fatal().Msgf("Failed to build ML vector: %v", err)
		}
		(*p.state)[len(*p.state)-1] = p.spread // raw spread, normalized when emitted
	}

	if m.Action() == move.MoveTypeExchange {
		// If it's an exchange, we need to re-add the exchanged tiles to the bag.
		// This is so that the game state is not corrupted for the next player.
		gw.game.Bag().PutBack(m.Tiles())
	}

	// Endgame label: the bag was empty before this move, so both racks are
	// known and a quick search from the opponent's reply beats whatever the
	// logged game did from here.
	if wanted && ga.endgamePlies > 0 && t.TilesRemaining == 0 && gw.game.Playing() == pb.PlayState_PLAYING {
		lbl, err := ga.solveEndgame(gw, mover, rack)
		switch {
		case errors.Is(err, context.DeadlineExceeded):
			ga.endgameTimeouts++
			log.Warn().Msgf("endgame search timed out on game %s turn %d; keeping the logged label", t.GameID, t.TurnNumber)
		case err != nil:
			log.Fatal().Msgf("Failed to endgame-label game %s turn %d: %v", t.GameID, t.TurnNumber, err)
		default:
			p.solved = &lbl
			ga.solved++
		}
	}

	// Rollout label, from exactly this state: mover on turn holding the
	// leave, opponent's rack in the bag.
	if wanted && ga.labeler != nil && (gw.pick > 0 || rand.Float64() < ga.sample) {
		lbl, ok, err := ga.labeler.Label(gw.game.Game, gw.ai, mover, m, gw.moveHistory())
		if err != nil {
			log.Fatal().Msgf("Failed to rollout-label game %s turn %d: %v", t.GameID, t.TurnNumber, err)
		}
		if ok {
			p.label = &lbl
			ga.labeled++
		}
	}

	// Undo the switch of the player on turn.
	gw.game.SetPlayerOnTurn(1 - gw.game.PlayerOnTurn())
	return p
}

// makeTrainingVector builds the training vector for plies[now] using
// plies[next] (the opponent's reply) and plies[future] (the horizon). ok is
// false when the position is not emitted: no feature vector was built, or
// a rollout labeler is on and the position was not labeled; the state
// vector, if any, goes back to the pool.
func (ga *GameAssembler) makeTrainingVector(gw *gameWindow, now, next, future int) (outputVector, bool) {
	pNow, pNext, pFuture := &gw.plies[now], &gw.plies[next], &gw.plies[future]
	if pNow.state == nil {
		return outputVector{}, false
	}
	if ga.labeler != nil && pNow.label == nil {
		game.MLVectorPool.Put(pNow.state)
		pNow.state = nil
		return outputVector{}, false
	}

	// Spreads from the mover's side; the future one flips if the other
	// player moved there.
	futureSpread := pFuture.spread
	if pFuture.mover != pNow.mover {
		futureSpread = -futureSpread
	}
	spreadDiff := futureSpread - pNow.spread

	// The value target is a win/loss-like signal after the horizon, not the
	// result of the entire game; that is TargetWDL, stamped at game end.
	bogowin := float32(0.0)
	if gw.game.Playing() == pb.PlayState_GAME_OVER {
		switch {
		case futureSpread > 0:
			bogowin = 1.0
		case futureSpread < 0:
			bogowin = -1.0
		}
	} else {
		// We are not at the end of the game, so calculate bogowin (winpct lookup table)
		// percentage based on the future spread.
		if futureSpread > equity.MaxRepresentedWinSpread {
			futureSpread = equity.MaxRepresentedWinSpread
		} else if futureSpread < -equity.MaxRepresentedWinSpread {
			futureSpread = -equity.MaxRepresentedWinSpread
		}
		bagRemaining := pFuture.bag
		if bagRemaining >= len(ga.winpcts) || bagRemaining < 0 {
			log.Fatal().Msgf("Bag remaining %d is out of bounds for winpcts", bagRemaining)
		}
		bogowin = ga.winpcts[int(equity.MaxRepresentedWinSpread-futureSpread)][bagRemaining]
		bogowin = bogowin*2 - 1 // scale to [-1, 1] range
	}

	// replace the raw spread with the normalized spread of this move only
	(*pNow.state)[len(*pNow.state)-1] = game.NormalizeSpreadForML(pNow.spread)
	ov := outputVector{
		features:    pNow.state,
		predictions: make([]float32, NumPredictions),
		gameID:      pNow.turn.GameID,
		turn:        pNow.turn.TurnNumber,
		mover:       pNow.mover,
		spreadNow:   pNow.spread,
		label:       pNow.label,
		solved:      pNow.solved,
	}
	pNow.state = nil // owned by the output now
	if pNow.label != nil {
		bogowin = pNow.label.value
		spreadDiff = pNow.label.spread
	}
	ov.predictions[TargetValue] = bogowin
	ov.predictions[TargetSpread] = game.NormalizeSpreadForML(spreadDiff)
	// TargetWDL is filled in by release once the game is over.

	// Did the opponent bingo on their next turn?
	if pNext.move.Action() == move.MoveTypePlay && pNext.move.TilesPlayed() == game.RackTileLimit {
		ov.predictions[TargetOppBingo] = 1.0
	}

	// Opponent's next score, capped.
	score := float32(pNext.move.Score())
	if score > 300 {
		score = 300
	}
	ov.predictions[TargetOppScore] = score / 300.0

	// Where the opponent's reply and the mover's own next move land. The
	// mover's next move is the ply after the reply; past the end of the game
	// (the opponent went out) it stays empty. The win conjunctions are
	// filled in by release.
	markPlacement(spatialPlane(ov.predictions, SpatialOppNext), pNext.move)
	if self := next + 1; next > now && self < len(gw.plies) {
		markPlacement(spatialPlane(ov.predictions, SpatialSelfNext), gw.plies[self].move)
	}

	return ov, true
}

// endgameLabel turns the mover's current spread and the search's value
// (the opponent's spread change to the end of the game) into the mover's
// label: their result and their spread change.
func endgameLabel(moverSpreadNow float32, oppChange int16) rolloutLabel {
	change := -float32(oppChange)
	lbl := rolloutLabel{spread: change}
	switch final := moverSpreadNow + change; {
	case final > 0:
		lbl.value = 1
	case final < 0:
		lbl.value = -1
	}
	return lbl
}

// solveEndgame labels the position the game is in (just after `mover`'s
// play with the bag empty; mover on turn holding `leave`, opponent's tiles
// in the bag) by a quick endgame search from the opponent's reply:
// endgamePlies plies of negamax with a greedy playout at the leaves. The
// search's value is the opponent's spread change to the end of the game.
// The game is left as it was found.
func (ga *GameAssembler) solveEndgame(gw *gameWindow, mover int, leave *tilemapping.Rack) (rolloutLabel, error) {
	g := gw.game.Game
	opp := 1 - mover
	spreadNow := float32(g.SpreadFor(mover))
	// ThrowRacksIn clears rack objects in place, so keep the leave's tiles,
	// not the rack, for the restore.
	leaveTiles := leave.TilesOn()
	setRackFromBag(gw, opp) // exactly the opponent's tiles
	g.SetPlayerOnTurn(opp)
	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(ga.endgamePlies + negamax.MaxGreedyPlayoutPlies + 5)
	g.SetEndgameMode(true)
	defer func() {
		g.SetEndgameMode(false)
		g.SetBackupMode(game.NoBackup)
		g.ThrowRacksIn()
		rack := tilemapping.NewRack(g.Alphabet())
		rack.Set(leaveTiles)
		if err := g.SetRackForOnly(mover, rack); err != nil {
			panic(err)
		}
		g.SetPlayerOnTurn(mover)
	}()

	if gw.es == nil {
		gen := movegen.NewGordonGenerator(ga.kwg, g.Board(), g.Bag().LetterDistribution())
		gw.es = new(negamax.Solver)
		if err := gw.es.Init(gen, g); err != nil {
			return rolloutLabel{}, err
		}
		gw.es.SetThreads(1)
		gw.es.SetFirstWinOptim(false) // we want the spread, not just the sign
		gw.es.SetSkipMaterialize(true)
		gw.es.SetNegascoutOptim(true)
	}
	ctx := context.Background()
	if ga.endgameTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, ga.endgameTimeout)
		defer cancel()
	}
	v, _, err := gw.es.QuickAndDirtySolve(ctx, ga.endgamePlies, 0)
	if err != nil {
		return rolloutLabel{}, err
	}
	return endgameLabel(spreadNow, v), nil
}
