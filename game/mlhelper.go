package game

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"github.com/rs/zerolog/log"
	"gorgonia.org/tensor"

	"github.com/domino14/word-golib/cache"
	wglconfig "github.com/domino14/word-golib/config"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/dataloaders"
	"github.com/domino14/macondo/move"
)

const (
	NN_C        = 83
	NN_H, NN_W  = 15, 15
	NN_N_PLANES = NN_C * NN_H * NN_W // 18 675
	NN_N_SCAL   = 56
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
		dataloaders.StrategyParamsPath(cfg), "macondo-nn.onnx", fields[1])
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
func (g *Game) MLEvaluateMove(m *move.Move) (float32, error) {
	evals, err := g.MLEvaluateMoves([]*move.Move{m})
	if err != nil {
		return 0, err
	}
	return evals[0], nil
}

// MLEvaluateMoves evaluates a slice of moves in a single batch inference.
func (g *Game) MLEvaluateMoves(moves []*move.Move) ([]float32, error) {
	start := time.Now()
	defer func() {
		elapsed := time.Since(start).Milliseconds()
		log.Debug().Int64("evaluate_moves_ms", elapsed).
			Int("num_moves", len(moves)).
			Msg("evaluated moves with ML")
	}()

	if len(moves) == 0 {
		return []float32{}, nil
	}

	backupMode := g.backupMode
	g.SetBackupMode(SimulationMode)
	defer g.SetBackupMode(backupMode)

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
		case move.MoveTypePass:
			g.lastScorelessTurns = g.scorelessTurns
			g.scorelessTurns++
			g.players[g.onturn].turns += 1
		case move.MoveTypeExchange:
			g.bag.PutBack(m.Tiles())
			g.lastScorelessTurns = g.scorelessTurns
			g.scorelessTurns++
			g.players[g.onturn].turns += 1
		}

		vec, err := g.BuildMLVector()
		if err != nil {
			g.UnplayLastMove() // Unplay before returning the error
			return nil, fmt.Errorf("failed to build ML vector for move: %w", err)
		}

		h := sha256.New()
		for _, f := range *vec {
			b := make([]byte, 4)
			binary.LittleEndian.PutUint32(b, math.Float32bits(f))
			h.Write(b)
		}
		log.Debug().Str("vec_sha256", fmt.Sprintf("%x", h.Sum(nil))).Msg("ml vector hash")

		allPlaneVectors = append(allPlaneVectors, (*vec)[:NN_N_PLANES]...)
		allScalarVectors = append(allScalarVectors, (*vec)[NN_N_PLANES:]...)
		MLVectorPool.Put(vec)   // Return the vector to the pool
		g.onturn = 1 - g.onturn // switch turn back to the original player

		g.UnplayLastMove()
	}

	boardTensor := tensor.New(tensor.WithShape(numMoves, NN_C, NN_H, NN_W),
		tensor.WithBacking(allPlaneVectors))
	scalTensor := tensor.New(tensor.WithShape(numMoves, NN_N_SCAL),
		tensor.WithBacking(allScalarVectors))

	model.model.SetInput(0, boardTensor)
	model.model.SetInput(1, scalTensor)

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

	for i := range evals {
		evals[i] *= 300.0
	}

	return evals, nil
}

func NormalizeSpreadForML(spread float32) float32 {
	// Normalize spread to -1 to 1 range.
	// First we clamp it to -300 or 300.
	if spread < -300 {
		spread = -300.0
	} else if spread > 300 {
		spread = 300.0
	}
	return float32(spread) / 300.0
}

// BuildMLVector builds the feature vector for the current game state.
func (g *Game) BuildMLVector() (*[]float32, error) {
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

	// Board features
	b := g.Board()
	for r := 0; r < 15; r++ {
		for c := 0; c < 15; c++ {
			idx := r*15 + c
			tile := b.GetLetter(r, c)
			if tile != 0 {
				unblanked := tile.Unblank()
				if unblanked != tile {
					isBlankPlane[idx] = 1.0
				} else {
					ll := unblanked - 1
					if ll >= 26 {
						return nil, fmt.Errorf("invalid tile index %d for tile %d at position (%d, %d)", ll, tile, r, c)
					}
					tilePlanes[int(ll)*planeSize+idx] = 1.0
				}
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
			if hc != board.TrivialCrossSet || vc != board.TrivialCrossSet {
				for t := 0; t < 26; t++ {
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
	}

	// Scalar features
	scalarsStart := NN_N_PLANES
	rackVector := vec[scalarsStart : scalarsStart+27]
	unseenVector := vec[scalarsStart+27 : scalarsStart+54]

	rack := g.RackFor(g.PlayerOnTurn())
	oppRack := g.RackFor(1 - g.PlayerOnTurn())
	g.bag.PutBack(oppRack.TilesOn())
	bag := g.Bag().PeekMap()

	for i := 0; i < 27; i++ {
		rackVector[i] = float32(rack.LetArr[i]) / 7.0
		unseenVector[i] = float32(bag[i]) / 20.0
	}
	g.bag.PutBack(oppRack.TilesOn()) // Restore opponent's rack

	vec[NN_RowLen-2] = float32(g.Bag().TilesRemaining()) / 100.0
	vec[NN_RowLen-1] = NormalizeSpreadForML(float32(g.SpreadFor(g.PlayerOnTurn())))

	return &vec, nil
}

func (g *Game) BinaryWriteMLVector(w *bufio.Writer, vec []float32) error {
	// Re-interpret the []float32 backing array as []byte
	byteSlice := unsafe.Slice(
		(*byte)(unsafe.Pointer(&vec[0])),
		len(vec)*4,
	)

	// 1) length prefix (little-endian uint32)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(byteSlice))); err != nil {
		return err
	}
	fmt.Println("wrote little endian length:", len(byteSlice))
	// 2) payload
	_, err := w.Write(byteSlice)
	return err
}
