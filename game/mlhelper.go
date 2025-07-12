package game

import (
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"github.com/rs/zerolog/log"
	"gorgonia.org/tensor"

	"github.com/domino14/word-golib/cache"
	wglconfig "github.com/domino14/word-golib/config"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/dataloaders"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/triton"
)

const (
	NN_C        = 85
	NN_H, NN_W  = 15, 15
	NN_N_PLANES = NN_C * NN_H * NN_W
	NN_N_SCAL   = 72
	NN_RowLen   = NN_N_PLANES + NN_N_SCAL
)

const (
	HistoryStartIdx      = NN_N_PLANES + 54
	HistoryEndIdx        = HistoryStartIdx + 3
	PowerTilesStartIdx   = HistoryEndIdx
	PowerTilesEndIdx     = PowerTilesStartIdx + 6
	VCRatioBagStartIdx   = PowerTilesEndIdx
	VCRatioBagEndIdx     = VCRatioBagStartIdx + 2
	VCRatioRackStartIdx  = VCRatioBagEndIdx
	VCRatioRackEndIdx    = VCRatioRackStartIdx + 2
	AddlFeaturesStartIdx = VCRatioRackEndIdx
	AddlFeaturesEndIdx   = AddlFeaturesStartIdx + 5
)

var MLVectorPool = sync.Pool{
	New: func() interface{} {
		// The Pool's New function should return a pointer to a slice of float32.
		v := make([]float32, NN_RowLen)
		return &v
	},
}

type MLModel struct {
	backend *gorgonnx.Graph
	model   *onnx.Model
}

// MLModelTemplate holds the raw ONNX model data.
type MLModelTemplate struct {
	data []byte
}

// NewInstance creates a new MLModel from the template.
func (t *MLModelTemplate) NewInstance() (*MLModel, error) {
	start := time.Now()
	defer func() {
		elapsed := time.Since(start).Milliseconds()
		log.Debug().Int64("onnx_model_init_ms", elapsed).Msg("onnx model instance created")
	}()
	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)
	err := model.UnmarshalBinary(t.data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX model: %w", err)
	}
	return &MLModel{
		backend: backend,
		model:   model,
	}, nil
}

func MLLoadFunc(cfg *wglconfig.Config, key string) (interface{}, error) {
	fields := strings.Split(key, ":")
	if fields[0] != "onnx" {
		return nil, errors.New("mlloadfunc - bad cache key: " + key)
	}
	if len(fields) != 2 {
		return nil, errors.New("cache key missing fields")
	}
	reader, err := dataloaders.StratFileForLexicon(
		dataloaders.StrategyParamsPath(cfg), "models/macondo-nn/1/model.onnx", fields[1])
	if err != nil {
		return nil, fmt.Errorf("failed to load ONNX model: %w", err)
	}
	bytes, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read ONNX model file: %w", err)
	}
	err = reader.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to close ONNX model file: %w", err)
	}

	log.Debug().Str("lexiconName", fields[1]).
		Int("model-size", len(bytes)).
		Msg("loaded-onnx-model")

	// Return the model template.
	modelTemplate := &MLModelTemplate{
		data: bytes,
	}

	return modelTemplate, nil
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

	if strings.ToLower(g.letterDistribution.Name) != "english" {
		return nil, fmt.Errorf("machine learning evaluation is only supported for English lexica at this time, got %s", g.letterDistribution.Name)
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
	backupMode := g.backupMode
	g.SetBackupMode(SimulationMode)
	defer g.SetBackupMode(backupMode)

	numMoves := len(moves)
	allPlaneVectors := make([]float32, 0, numMoves*NN_N_PLANES)
	allScalarVectors := make([]float32, 0, numMoves*NN_N_SCAL)
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

		vec, err := g.BuildMLVector(m, leaveCalc.LeaveValue(m.Leave()), lastMoves)
		if err != nil {
			g.UnplayLastMove() // Unplay before returning the error
			return nil, fmt.Errorf("failed to build ML vector for move: %w", err)
		}

		// Compute SHA256 hash of the vector for debugging or deduplication
		// Import "crypto/sha256" at the top if not already imported
		// hash := sha256.Sum256(unsafe.Slice((*byte)(unsafe.Pointer(&(*vec)[0])), len(*vec)*4))
		// fmt.Printf("ML vector SHA256: %x, move %s\n", hash, m.ShortDescription())

		allPlaneVectors = append(allPlaneVectors, (*vec)[:NN_N_PLANES]...)
		allScalarVectors = append(allScalarVectors, (*vec)[NN_N_PLANES:]...)
		MLVectorPool.Put(vec)   // Return the vector to the pool
		g.onturn = 1 - g.onturn // switch turn back to the original player
		g.UnplayLastMove()
	}

	// write the vector to a test file for debugging. I think this only works
	// for one single position.
	// testFile, err := os.Create("/tmp/test-vec-infer.bin")
	// if err != nil {
	// 	log.Fatal().Err(err).Msg("Failed to create test file")
	// }

	// testOut := bufio.NewWriterSize(testFile, 50000)
	// if err := BinaryWriteMLVector(testOut, append(allPlaneVectors, allScalarVectors...)); err != nil {
	// 	log.Fatal().Err(err).Msg("Failed to write test vector to file")
	// }
	// if err := testOut.Flush(); err != nil {
	// 	log.Fatal().Err(err).Msg("Failed to flush test vector to file")
	// }
	// testFile.Close()

	if g.config.GetBool(config.ConfigTritonUseTriton) {
		return g.mlevaluateMovesTriton(len(moves), allPlaneVectors, allScalarVectors)
	}
	return g.mlevaluateMovesLocal(len(moves), allPlaneVectors, allScalarVectors)
}

func (g *Game) mlevaluateMovesTriton(nmoves int, planeVectors, scalarVectors []float32) (*triton.ModelOutputs, error) {

	if g.tritonClient == nil {
		return nil, errors.New("triton client is not initialized")
	}
	log.Debug().Int("num_moves", nmoves).
		Msg("evaluating moves with Triton")
	// Ensure the input vectors are of the correct size
	return g.tritonClient.Infer(planeVectors, scalarVectors, nmoves)
}

func (g *Game) mlevaluateMovesLocal(nmoves int, planeVectors, scalarVectors []float32) (*triton.ModelOutputs, error) {

	net, err := cache.Load(g.config.WGLConfig(), "onnx:"+g.LexiconName(), MLLoadFunc)
	if err != nil {
		log.Err(err).Msg("loading-ml-model")
		return nil, err
	}
	modelTemplate, ok := net.(*MLModelTemplate)
	if !ok {
		return nil, errors.New("failed to type-assert ONNX model template")
	}
	model, err := modelTemplate.NewInstance()
	if err != nil {
		return nil, fmt.Errorf("failed to create new ONNX model instance: %w", err)
	}

	boardTensor := tensor.New(tensor.WithShape(nmoves, NN_C, NN_H, NN_W),
		tensor.WithBacking(planeVectors))
	scalTensor := tensor.New(tensor.WithShape(nmoves, NN_N_SCAL),
		tensor.WithBacking(scalarVectors))

	model.model.SetInput(0, boardTensor)
	model.model.SetInput(1, scalTensor)

	log.Debug().Int("num_moves", nmoves).
		Msg("evaluating moves with local ONNX model")

	if err := model.backend.Run(); err != nil {
		return nil, fmt.Errorf("failed to run ONNX model: %w", err)
	}

	// XXX: FIX ME

	output, err := model.model.GetOutputTensors()
	if err != nil {
		return nil, fmt.Errorf("failed to get output tensors: %w", err)
	}

	var evals []float32

	// fmt.Println("output tensors:", len(output), output[0].Shape(), output[0].Data())
	switch v := output[0].Data().(type) {
	case []float32:
		evals = v
	case float32:
		evals = []float32{v}
	default:
		return nil, fmt.Errorf("unexpected output type: %T", v)
	}
	mo := &triton.ModelOutputs{
		Value: evals,
	}

	return mo, nil
}

func NormalizeSpreadForML(spread float32) float32 {
	return ScaleScoreWithTanh(spread, 0.0, 130.0)
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

	vecPtr := MLVectorPool.Get().(*[]float32)
	vec := *vecPtr
	// Clear the vector
	for i := range vec {
		vec[i] = 0
	}

	// Define slices for each feature plane, pointing into the main vector `vec`.
	const planeSize = 15 * 15
	tilePlanes := vec[0 : 26*planeSize]
	isBlankPlane := vec[26*planeSize : 27*planeSize]
	horCCs := vec[27*planeSize : 53*planeSize]
	verCCs := vec[53*planeSize : 79*planeSize]
	bonus2LPlane := vec[79*planeSize : 80*planeSize]
	bonus3LPlane := vec[80*planeSize : 81*planeSize]
	bonus2WPlane := vec[81*planeSize : 82*planeSize]
	bonus3WPlane := vec[82*planeSize : 83*planeSize]
	// historyPlane indices: 0 is used for the immediate move that is being evaluated, on our
	// side, 1 is the last move played by the opponent
	historyPlanes := vec[83*planeSize : 85*planeSize]

	// Board features
	b := g.Board()
	for r := range 15 {
		for c := range 15 {
			idx := r*15 + c
			tile := b.GetLetter(r, c)
			if tile != 0 {
				unblanked := tile.Unblank()
				if unblanked != tile {
					isBlankPlane[idx] = 1.0
				}
				ll := unblanked - 1
				if ll >= 26 {
					return nil, fmt.Errorf("invalid tile index %d for tile %d at position (%d, %d)", ll, tile, r, c)
				}
				tilePlanes[int(ll)*planeSize+idx] = 1.0

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
			for t := range 26 {
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
	turnsSinceLastOppBingo := 0
	for i := len(lastMoves) - 1; i >= 0; i -= 2 {
		if lastMoves[i].Action() == move.MoveTypePlay && lastMoves[i].TilesPlayed() == RackTileLimit {
			// Bingo played, stop looking for more moves
			break
		}
		turnsSinceLastOppBingo++
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
				if curR < 0 || curR >= 15 || curC < 0 || curC >= 15 {
					return nil, fmt.Errorf("last move out of bounds at (%d, %d)", curR, curC)
				}
				if lastMove.Tiles()[i] != 0 {
					idx := curR*15 + curC
					historyPlanes[lmIdx*planeSize+idx] = 1.0 // Mark the square where the last move was played
				}
			}
		}
	}

	// Scalar features
	scalarsStart := NN_N_PLANES
	rackVector := vec[scalarsStart : scalarsStart+27]
	unseenVector := vec[scalarsStart+27 : scalarsStart+54]

	// History vectors have three elements each, in order:
	// score, n tiles played, n tiles exchanged.
	historyVectors := vec[HistoryStartIdx:HistoryEndIdx]
	powerTilesVector := vec[PowerTilesStartIdx:PowerTilesEndIdx]
	vcRatioBag := vec[VCRatioBagStartIdx:VCRatioBagEndIdx]
	vcRatioRack := vec[VCRatioRackStartIdx:VCRatioRackEndIdx]
	addlFeatures := vec[AddlFeaturesStartIdx:AddlFeaturesEndIdx]

	rack := g.RackFor(g.PlayerOnTurn())
	bag := g.bag.PeekMap()
	tr := g.bag.TilesRemaining()
	for i := 0; i < 27; i++ {
		rackVector[i] = float32(rack.LetArr[i]) / 7     // Rack tiles
		unseenVector[i] = float32(bag[i]) / float32(tr) // rough prob of drawing this tile
		// power tiles seem very redundant since this info is already in the bag
		// vector. However, it's not scaled there the same. Let's try it anyway.
		switch i {
		case 0: // Blank tile
			powerTilesVector[0] = float32(bag[i]) / 2.0
		case 10: // J tile
			powerTilesVector[1] = float32(bag[i])
		case 17: // Q tile
			powerTilesVector[2] = float32(bag[i])
		case 19: // S tile
			powerTilesVector[3] = float32(bag[i]) / 4.0
		case 24: // X tile
			powerTilesVector[4] = float32(bag[i])
		case 26: // Z tile
			powerTilesVector[5] = float32(bag[i])
		}
	}
	// Skip our own move, just go through last few opp moves:
	for lmIdx, lastMove := range relevantMoves[1:] {
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

	vowelsBag := bag[1] + bag[5] + bag[9] + bag[15] + bag[21] // AEIOU
	vowelsRack := rack.LetArr[1] + rack.LetArr[5] + rack.LetArr[9] + rack.LetArr[15] + rack.LetArr[21]
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
	return &vec, nil
}
