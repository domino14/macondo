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
)

const (
	NN_C        = 85
	NN_H, NN_W  = 15, 15
	NN_N_PLANES = NN_C * NN_H * NN_W
	NN_N_SCAL   = 71
	NN_RowLen   = NN_N_PLANES + NN_N_SCAL
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
	lastMove *move.Move) (float32, error) {
	evals, err := g.MLEvaluateMoves([]*move.Move{m}, leaveCalc, lastMove)
	if err != nil {
		return 0, err
	}
	return evals[0], nil
}

// MLEvaluateMoves evaluates a slice of moves in a single batch inference.
// Their equities must already be set.
func (g *Game) MLEvaluateMoves(moves []*move.Move, leaveCalc *equity.ExhaustiveLeaveCalculator,
	lastMove *move.Move) ([]float32, error) {

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
		return []float32{}, nil
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

		vec, err := g.BuildMLVector(m, leaveCalc.LeaveValue(m.Leave()), lastMove)
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

func (g *Game) mlevaluateMovesTriton(nmoves int, planeVectors, scalarVectors []float32) ([]float32, error) {

	if g.tritonClient == nil {
		return nil, errors.New("triton client is not initialized")
	}
	log.Debug().Int("num_moves", nmoves).
		Msg("evaluating moves with Triton")
	// Ensure the input vectors are of the correct size
	return g.tritonClient.Infer(planeVectors, scalarVectors, nmoves)
}

func (g *Game) mlevaluateMovesLocal(nmoves int, planeVectors, scalarVectors []float32) ([]float32, error) {

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

	return evals, nil
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

// BuildMLVector builds the feature vector for the current game state. It
// should not modify the game state!
func (g *Game) BuildMLVector(m *move.Move, evalMoveLeaveVal float64, lastMove *move.Move) (
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
	lastMovePlane := vec[83*planeSize : 84*planeSize]
	ourMovePlane := vec[84*planeSize : 85*planeSize]

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

	if lastMove != nil {
		// Encode the last move's tiles played.
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
					lastMovePlane[curR*15+curC] = 1.0 // Mark the square where the last move was played
				}
			}
		}

	}
	if m.Action() == move.MoveTypePlay {
		// Encode the current move's tiles played.
		r, c, vertical := m.CoordsAndVertical()
		ri, ci := 0, 1
		if vertical {
			ri, ci = 1, 0 // Vertical means row changes, column stays
		}
		for i := range m.Tiles() {
			curR := r + i*ri
			curC := c + i*ci
			if curR < 0 || curR >= 15 || curC < 0 || curC >= 15 {
				return nil, fmt.Errorf("current move out of bounds at (%d, %d)", curR, curC)

			}
			if m.Tiles()[i] != 0 {
				ourMovePlane[curR*15+curC] = 1.0 // Mark the square where the current move is played
			}
		}
	}

	// Scalar features
	scalarsStart := NN_N_PLANES
	rackVector := vec[scalarsStart : scalarsStart+27]
	unseenVector := vec[scalarsStart+27 : scalarsStart+54]
	lastOppScoreVector := vec[scalarsStart+54 : scalarsStart+55]
	lastOppNTilesPlayedVector := vec[scalarsStart+55 : scalarsStart+56]
	lastOppNTilesExchangedVector := vec[scalarsStart+56 : scalarsStart+57]
	powerTilesVector := vec[scalarsStart+57 : scalarsStart+63]
	vcRatioBag := vec[scalarsStart+63 : scalarsStart+65]
	vcRatioRack := vec[scalarsStart+65 : scalarsStart+67]
	scoreAndBagFeatures := vec[scalarsStart+67 : scalarsStart+71]

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
	if lastMove != nil {
		lastOppScoreVector[0] = ScaleScoreWithTanh(float32(lastMove.Score()), 45.0, 40.0) // last opponent's move score
		if lastMove.Action() == move.MoveTypePlay {
			lastOppNTilesPlayedVector[0] = float32(lastMove.TilesPlayed()) / 7.0 // last opponent's move tiles played
		} else if lastMove.Action() == move.MoveTypeExchange {
			lastOppNTilesExchangedVector[0] = float32(lastMove.TilesPlayed()) / 7.0 // last opponent's move tiles exchanged
		}
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

	scoreAndBagFeatures[0] = ScaleScoreWithTanh(float32(m.Score()), 45.0, 40.0)    // last move score
	scoreAndBagFeatures[1] = ScaleScoreWithTanh(float32(evalMoveLeaveVal), 10, 20) // last move leave value
	scoreAndBagFeatures[2] = float32(g.Bag().TilesRemaining()) / 100.0
	scoreAndBagFeatures[3] = NormalizeSpreadForML(float32(g.SpreadFor(g.PlayerOnTurn())))

	return &vec, nil
}
