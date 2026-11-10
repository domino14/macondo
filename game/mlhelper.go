package game

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/triton"
)

// NN_H x NN_W is the board the net sees. The number of planes and scalars
// depends on the alphabet (see MLLayout).
const NN_H, NN_W = 15, 15

// MLVectorPool holds feature vectors (BuildMLVector); a vector is grown to
// the layout's row length when taken.
var MLVectorPool = sync.Pool{
	New: func() interface{} {
		return new([]float32)
	},
}

// MLEvaluateMove evaluates a single move using the machine learning model.
// It's a wrapper around MLEvaluateMoves.
func (g *Game) MLEvaluateMove(m *move.Move, leaveCalc *equity.ExhaustiveLeaveCalculator,
	lastMoves []*move.Move) (*triton.ModelOutputs, error) {
	evals, err := g.MLEvaluateMoves([]*move.Move{m}, leaveCalc, lastMoves)
	if err != nil {
		return nil, err
	}
	return evals, nil
}

// MLEvaluateMoves evaluates a slice of moves in a single batch inference.
// Their equities must already be set.
func (g *Game) MLEvaluateMoves(moves []*move.Move, leaveCalc *equity.ExhaustiveLeaveCalculator,
	lastMoves []*move.Move) (*triton.ModelOutputs, error) {
	return g.MLEvaluateMovesWith(nil, moves, leaveCalc, lastMoves)
}

// MLEvaluateMovesWith is MLEvaluateMoves against the given Triton client
// (a bot's own model); nil means the game's client.
func (g *Game) MLEvaluateMovesWith(client *triton.TritonClient, moves []*move.Move,
	leaveCalc *equity.ExhaustiveLeaveCalculator, lastMoves []*move.Move) (*triton.ModelOutputs, error) {

	lay, err := g.mlLayout()
	if err != nil {
		return nil, err
	}

	start := time.Now()
	defer func() {
		elapsed := time.Since(start).Milliseconds()
		log.Debug().Int64("evaluate_moves_ms", elapsed).
			Int("num_moves", len(moves)).
			Msg("evaluated moves")
	}()
	if len(moves) == 0 {
		return nil, nil
	}
	if g.config.GetBool(config.ConfigTritonUseTriton) {
		c := client
		if c == nil {
			c = g.tritonClient
		}
		if c != nil {
			if err := checkMLModelShape(c.ModelName(), c.InputShape, lay, g.letterDistribution.Name); err != nil {
				return nil, err
			}
		}
		if c != nil && c.BoardUint8() {
			boards, scalars, err := g.mlByteVectorsInto(g.mlBoardBytesBuf[:0], g.mlScalarsBuf[:0], moves, leaveCalc, lastMoves)
			if err != nil {
				return nil, err
			}
			g.mlBoardBytesBuf, g.mlScalarsBuf = boards, scalars
			return mlInferBatches(len(moves), func(from, to int) (*triton.ModelOutputs, error) {
				return c.InferUint8(boards[from*lay.PlaneLen:to*lay.PlaneLen],
					scalars[from*lay.NumScalars:to*lay.NumScalars], to-from)
			})
		}
	}
	allPlaneVectors, allScalarVectors, err := g.mlVectorsInto(g.mlPlanesBuf[:0], g.mlScalarsBuf[:0], moves, leaveCalc, lastMoves)
	if err != nil {
		return nil, err
	}
	// Keep the (possibly grown) buffers for the next request. The vectors
	// are consumed before this returns: the inference call is synchronous.
	g.mlPlanesBuf, g.mlScalarsBuf = allPlaneVectors, allScalarVectors
	if g.config.GetBool(config.ConfigTritonUseTriton) {
		return g.mlevaluateMovesTriton(client, lay, len(moves), allPlaneVectors, allScalarVectors)
	}
	return g.mlevaluateMovesLocal(lay, len(moves), allPlaneVectors, allScalarVectors)
}

// MLVectorsForMoves builds the feature vectors the net sees for each
// candidate move, concatenated: the position right after the move, from
// the mover's side, holding only the leave, with the opponent's rack put
// back into the unseen pool and no tiles drawn. This is the inference-side
// counterpart of the producer's training vectors (cmd/mlproducer), and
// the two must agree; TestInferenceVectorsMatchTraining checks that.
func (g *Game) MLVectorsForMoves(moves []*move.Move, leaveCalc *equity.ExhaustiveLeaveCalculator,
	lastMoves []*move.Move) (planes, scalars []float32, err error) {
	return g.mlVectorsInto(nil, nil, moves, leaveCalc, lastMoves)
}

// mlVectorsInto is MLVectorsForMoves appending to the given buffers (pass
// them with length 0 to reuse their capacity); it returns the filled slices.
func (g *Game) mlVectorsInto(allPlaneVectors, allScalarVectors []float32, moves []*move.Move,
	leaveCalc *equity.ExhaustiveLeaveCalculator, lastMoves []*move.Move) (planes, scalars []float32, err error) {

	lay, err := g.mlLayout()
	if err != nil {
		return nil, nil, err
	}
	numMoves := len(moves)
	allPlaneVectors = slices.Grow(allPlaneVectors[:0], numMoves*lay.PlaneLen)
	allScalarVectors = slices.Grow(allScalarVectors[:0], numMoves*lay.NumScalars)
	err = g.forEachMLMove(moves, func(m *move.Move) error {
		vec, err := g.BuildMLVector(m, leaveCalc.LeaveValue(m.Leave()), lastMoves)
		if err != nil {
			return fmt.Errorf("failed to build ML vector for move: %w", err)
		}
		allPlaneVectors = append(allPlaneVectors, (*vec)[:lay.PlaneLen]...)
		allScalarVectors = append(allScalarVectors, (*vec)[lay.PlaneLen:]...)
		MLVectorPool.Put(vec) // Return the vector to the pool
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return allPlaneVectors, allScalarVectors, nil
}

// forEachMLMove calls f with the game in the state the net sees each
// candidate in: the move played (score, rack and board updated, cross-sets
// regenerated), the mover still on turn, and the opponent's rack put back
// into the bag. The game is restored after each call.
func (g *Game) forEachMLMove(moves []*move.Move, f func(m *move.Move) error) error {
	backupMode := g.backupMode
	g.SetBackupMode(SimulationMode)
	defer g.SetBackupMode(backupMode)

	for _, m := range moves {
		g.backupState()
		switch m.Action() {
		case move.MoveTypePlay:
			g.board.PlayMove(m)
			g.crossSetGen.UpdateForMove(g.board, m)
			score := m.Score()
			g.lastScorelessTurns = g.scorelessTurns
			g.scorelessTurns = 0
			g.players[g.onturn].points += score
			g.players[g.onturn].turns += 1
			if m.TilesPlayed() == RackTileLimit {
				g.players[g.onturn].bingos++
			}
			// Take the tiles played from the rack.
			for _, t := range m.Tiles() {
				if t != 0 {
					g.players[g.onturn].rack.Take(t.IntrinsicTileIdx())
				}
			}

		case move.MoveTypePass:
			g.lastScorelessTurns = g.scorelessTurns
			g.scorelessTurns++
			g.players[g.onturn].turns += 1
		case move.MoveTypeExchange:
			// Don't throw the exchanged tiles in, since we want to evaluate
			// the move as if the player was about to draw new tiles.
			g.lastScorelessTurns = g.scorelessTurns
			g.scorelessTurns++
			g.players[g.onturn].turns += 1
			// Take the tiles exchanged from the rack.
			for _, t := range m.Tiles() {
				g.players[g.onturn].rack.Take(t.IntrinsicTileIdx())
			}

		}
		// Put the opponent's rack back into the bag for evaluation.
		// This is necessary to ensure the model sees the correct bag state.
		oppRack := g.RackFor(1 - g.PlayerOnTurn())
		g.bag.PutBack(oppRack.TilesOn())

		err := f(m)
		g.onturn = 1 - g.onturn // switch turn back to the original player
		g.UnplayLastMove()
		if err != nil {
			return err
		}
	}
	return nil
}

func (g *Game) mlevaluateMovesTriton(client *triton.TritonClient, lay *MLLayout, nmoves int, planeVectors, scalarVectors []float32) (*triton.ModelOutputs, error) {
	if client == nil {
		client = g.tritonClient
	}
	if client == nil {
		if _, err := g.MLModelName(); err != nil {
			return nil, err // no model for this lexicon
		}
		return nil, errors.New("triton client is not initialized")
	}
	log.Debug().Int("num_moves", nmoves).
		Msg("evaluating moves with Triton")
	return mlInferBatches(nmoves, func(from, to int) (*triton.ModelOutputs, error) {
		return client.Infer(planeVectors[from*lay.PlaneLen:to*lay.PlaneLen],
			scalarVectors[from*lay.NumScalars:to*lay.NumScalars], to-from)
	})
}

// mlInferBatches runs infer on [0, nmoves) in one request, or, with more
// candidates than the engine's largest batch, in several, outputs
// concatenated in order.
func mlInferBatches(nmoves int, infer func(from, to int) (*triton.ModelOutputs, error)) (*triton.ModelOutputs, error) {
	if nmoves <= MLMaxBatch {
		return infer(0, nmoves)
	}
	all := &triton.ModelOutputs{}
	for _, r := range mlBatchRanges(nmoves, MLMaxBatch) {
		out, err := infer(r[0], r[1])
		if err != nil {
			return nil, err
		}
		all.Value = append(all.Value, out.Value...)
		all.Spread = append(all.Spread, out.Spread...)
		all.Points = append(all.Points, out.Points...)
		all.BingoProb = append(all.BingoProb, out.BingoProb...)
		all.OppScore = append(all.OppScore, out.OppScore...)
	}
	return all, nil
}

// MLMaxBatch is the largest batch the served TensorRT engines accept
// (pytorch/onnx-to-tensorrt.py builds them for batch 1..128).
const MLMaxBatch = 128

// mlBatchRanges splits n items into consecutive [from, to) ranges of at
// most max items.
func mlBatchRanges(n, max int) [][2]int {
	var out [][2]int
	for from := 0; from < n; from += max {
		to := from + max
		if to > n {
			to = n
		}
		out = append(out, [2]int{from, to})
	}
	return out
}

// MLSpreadScale is the tanh scale of the net's spread input and spread
// output: NormalizeSpreadForML(x) = tanh(x / MLSpreadScale).
const MLSpreadScale = 130.0

func NormalizeSpreadForML(spread float32) float32 {
	return ScaleScoreWithTanh(spread, 0.0, MLSpreadScale)
}

func ScaleScoreWithTanh(score float32, center float32, scaleFactor float32) float32 {
	// Center the score around the desired value (40 or 50)
	centered := score - center

	// Apply tanh scaling
	return float32(math.Tanh(float64(centered / scaleFactor)))
}

func InverseScaleScoreWithTanh(y float32, center float32, scaleFactor float32) float32 {
	// Use the atanh function: atanh(y) = 0.5 * ln((1 + y) / (1 - y))
	atanh := 0.5 * math.Log((1+float64(y))/(1-float64(y)))
	return float32(float64(scaleFactor)*atanh + float64(center))
}

// BuildMLVector builds the feature vector for the current game state. It
// should not modify the game state!
// lastMoves contains the last few moves played, most recent last. (-1 is opponent)
func (g *Game) BuildMLVector(m *move.Move, evalMoveLeaveVal float64, lastMoves []*move.Move) (
	*[]float32, error) {

	lay, err := g.mlLayout()
	if err != nil {
		return nil, err
	}
	vecPtr := MLVectorPool.Get().(*[]float32)
	if cap(*vecPtr) < lay.RowLen {
		*vecPtr = make([]float32, lay.RowLen)
	}
	*vecPtr = (*vecPtr)[:lay.RowLen]
	vec := *vecPtr
	clear(vec)

	// Define slices for each feature plane, pointing into the main vector `vec`.
	const planeSize = mlPlaneSize
	tilePlanes := vec[0:lay.blankPlane]
	isBlankPlane := vec[lay.blankPlane:lay.horCCPlanes]
	horCCs := vec[lay.horCCPlanes:lay.verCCPlanes]
	verCCs := vec[lay.verCCPlanes:lay.bonusPlanes]
	bonus2LPlane := vec[lay.bonusPlanes : lay.bonusPlanes+planeSize]
	bonus3LPlane := vec[lay.bonusPlanes+planeSize : lay.bonusPlanes+2*planeSize]
	bonus2WPlane := vec[lay.bonusPlanes+2*planeSize : lay.bonusPlanes+3*planeSize]
	bonus3WPlane := vec[lay.bonusPlanes+3*planeSize : lay.historyPlanes]
	// historyPlane indices: 0 is used for the immediate move that is being evaluated, on our
	// side, 1 is the last move played by the opponent
	historyPlanes := vec[lay.historyPlanes:lay.PlaneLen]

	// Board features
	b := g.Board()
	for r := range NN_H {
		for c := range NN_W {
			idx := r*NN_W + c
			tile := b.GetLetter(r, c)
			if tile != 0 {
				unblanked := tile.Unblank()
				if unblanked != tile {
					isBlankPlane[idx] = 1.0
				}
				ll := int(unblanked) - 1
				if ll >= lay.Letters {
					return nil, fmt.Errorf("invalid tile index %d for tile %d at position (%d, %d)", ll, tile, r, c)
				}
				tilePlanes[ll*planeSize+idx] = 1.0

			} else { // Empty square, check for bonuses
				bonus := b.GetBonus(r, c)
				switch bonus {
				case board.Bonus2LS:
					bonus2LPlane[idx] = 1.0
				case board.Bonus3LS:
					bonus3LPlane[idx] = 1.0
				case board.Bonus2WS:
					bonus2WPlane[idx] = 1.0
				case board.Bonus3WS:
					bonus3WPlane[idx] = 1.0
				}
			}

			// Cross-checks
			hc := b.GetCrossSet(r, c, board.HorizontalDirection)
			vc := b.GetCrossSet(r, c, board.VerticalDirection)
			for t := range lay.Letters {
				letter := tilemapping.MachineLetter(t + 1)
				if hc.Allowed(letter) {
					horCCs[t*planeSize+idx] = 1.0
				}
				if vc.Allowed(letter) {
					verCCs[t*planeSize+idx] = 1.0
				}
			}
		}
	}

	relevantMoves := []*move.Move{}
	relevantMoves = append(relevantMoves, m) // Current move being evaluated
	if len(lastMoves) > 0 {
		relevantMoves = append(relevantMoves, lastMoves[len(lastMoves)-1]) // Last OPPONENT move
	}
	for lmIdx, lastMove := range relevantMoves {
		if lastMove.Action() == move.MoveTypePlay {
			r, c, vertical := lastMove.CoordsAndVertical()
			ri, ci := 0, 1
			if vertical {
				ri, ci = 1, 0 // Vertical means row changes, column stays
			}
			for i := range lastMove.Tiles() {
				curR := r + i*ri
				curC := c + i*ci
				if curR < 0 || curR >= NN_H || curC < 0 || curC >= NN_W {
					return nil, fmt.Errorf("last move out of bounds at (%d, %d)", curR, curC)
				}
				if lastMove.Tiles()[i] != 0 {
					idx := curR*NN_W + curC
					historyPlanes[lmIdx*planeSize+idx] = 1.0 // Mark the square where the last move was played
				}
			}
		}
	}

	g.mlScalarsInto(lay, vec[lay.PlaneLen:], m, evalMoveLeaveVal, lastMoves)
	return vecPtr, nil
}

// mlScalarsInto writes the layout's scalar features into sc, which must
// be zeroed.
func (g *Game) mlScalarsInto(lay *MLLayout, sc []float32, m *move.Move, evalMoveLeaveVal float64, lastMoves []*move.Move) {
	rackVector := sc[0 : lay.Letters+1]
	unseenVector := sc[lay.Letters+1 : lay.historyScal]

	// History vectors have three elements each, in order:
	// score, n tiles played, n tiles exchanged.
	historyVectors := sc[lay.historyScal:lay.powerScal]
	powerTilesVector := sc[lay.powerScal:lay.vcBagScal]
	vcRatioBag := sc[lay.vcBagScal:lay.vcRackScal]
	vcRatioRack := sc[lay.vcRackScal:lay.addlScal]
	addlFeatures := sc[lay.addlScal:lay.NumScalars]
	var lastOpp *move.Move
	if len(lastMoves) > 0 {
		lastOpp = lastMoves[len(lastMoves)-1]
	}
	turnsSinceLastOppBingo := mlTurnsSinceLastOppBingo(lastMoves)

	rack := g.RackFor(g.PlayerOnTurn())
	bag := g.bag.PeekMap()
	tr := g.bag.TilesRemaining()
	for i := 0; i <= lay.Letters; i++ {
		rackVector[i] = float32(rack.LetArr[i]) / 7 // Rack tiles
		if tr > 0 {
			unseenVector[i] = float32(bag[i]) / float32(tr) // rough prob of drawing this tile
		}
	}
	// power tiles seem very redundant since this info is already in the bag
	// vector. However, it's not scaled there the same. Let's try it anyway.
	// Each is the fraction of that tile still unseen (English: blank/2,
	// J, Q, S/4, X, Z).
	for k, ml := range lay.power {
		powerTilesVector[k] = float32(bag[ml]) / lay.powerCount[k]
	}
	// The last opponent move, if any:
	if lastMove := lastOpp; lastMove != nil {
		lmIdx := 0
		score := ScaleScoreWithTanh(float32(lastMove.Score()), 45.0, 40.0)
		var tilesPlayed, tilesExchanged float32
		if lastMove.Action() == move.MoveTypePlay {
			tilesPlayed = float32(lastMove.TilesPlayed()) / 7.0
		} else if lastMove.Action() == move.MoveTypeExchange {
			tilesExchanged = float32(lastMove.TilesPlayed()) / 7.0
		}
		historyVectors[lmIdx*3] = score            // last move score
		historyVectors[lmIdx*3+1] = tilesPlayed    // last move tiles
		historyVectors[lmIdx*3+2] = tilesExchanged // last move tiles exchanged
	}

	var vowelsBag uint8 // English: AEIOU
	vowelsRack := 0
	for _, v := range lay.vowels {
		vowelsBag += bag[v]
		vowelsRack += rack.LetArr[v]
	}
	if tr > 0 {
		vcRatioBag[0] = float32(vowelsBag) / float32(tr)         // Vowel ratio in bag
		vcRatioBag[1] = float32(tr-int(vowelsBag)) / float32(tr) // Consonant ratio in bag
	}
	if rack.NumTiles() > 0 {
		vcRatioRack[0] = float32(vowelsRack) / float32(rack.NumTiles())                      // Vowel ratio in rack
		vcRatioRack[1] = float32(int(rack.NumTiles())-vowelsRack) / float32(rack.NumTiles()) // Consonant ratio in rack
	}

	addlFeatures[0] = float32(turnsSinceLastOppBingo) / 25.0                // turns since last opponent bingo
	addlFeatures[1] = ScaleScoreWithTanh(float32(m.Score()), 45.0, 40.0)    // evaluated move score
	addlFeatures[2] = ScaleScoreWithTanh(float32(evalMoveLeaveVal), 10, 20) // evaluated move leave value
	addlFeatures[3] = float32(g.Bag().TilesRemaining()) / 100.0
	addlFeatures[4] = NormalizeSpreadForML(float32(g.SpreadFor(g.PlayerOnTurn())))
}

// mlTurnsSinceLastOppBingo counts the opponent's turns (every other entry
// of lastMoves, most recent last) back to their last bingo.
func mlTurnsSinceLastOppBingo(lastMoves []*move.Move) int {
	n := 0
	for i := len(lastMoves) - 1; i >= 0; i -= 2 {
		if lastMoves[i].Action() == move.MoveTypePlay && lastMoves[i].TilesPlayed() == RackTileLimit {
			// Bingo played, stop looking for more moves
			break
		}
		n++
	}
	return n
}
