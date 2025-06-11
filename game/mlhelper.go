package game

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"
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

// MLEvaluateMove evaluates a move using the machine learning model.
// Assume that the board and racks etc are already properly assigned.
func (g *Game) MLEvaluateMove(m *move.Move) (float32, error) {
	start := time.Now()
	defer func() {
		elapsed := time.Since(start).Milliseconds()
		log.Debug().Int64("evaluate_move_ms", elapsed).Msg("evaluated move with ML")
	}()
	backupMode := g.backupMode
	g.SetBackupMode(SimulationMode)
	defer g.SetBackupMode(backupMode)

	net, err := cache.Load(g.config.WGLConfig(), "onnx:"+g.LexiconName(), MLLoadFunc)
	if err != nil {
		log.Err(err).Msg("loading-ml-model")
		return 0, err
	}
	modelTemplate, ok := net.(*MLModelTemplate)
	if !ok {
		return 0, errors.New("failed to type-assert ONNX model template")
	}
	model, err := modelTemplate.NewInstance()
	if err != nil {
		return 0, fmt.Errorf("failed to create new ONNX model instance: %w", err)
	}

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
		// Don't draw replacement tiles.
		// don't deal with end of game at this moment.
	case move.MoveTypePass:
		g.lastScorelessTurns = g.scorelessTurns
		g.scorelessTurns++
		g.players[g.onturn].turns += 1

	case move.MoveTypeExchange:
		// don't actually exchange, but we want to track our leave.
		g.bag.PutBack(m.Tiles())
		g.lastScorelessTurns = g.scorelessTurns
		g.scorelessTurns++
		g.players[g.onturn].turns += 1
	}

	vec, err := g.BuildMLVector()
	if err != nil {
		return 0, fmt.Errorf("failed to build ML vector: %w", err)
	}

	// Generate a SHA256 checksum for the ML vector.
	// hasher := sha256.New()
	// for _, v := range vec {
	// 	b := make([]byte, 4)
	// 	binary.LittleEndian.PutUint32(b, math.Float32bits(v))
	// 	hasher.Write(b)
	// }
	// checksum := hex.EncodeToString(hasher.Sum(nil))
	// log.Info().Str("ml_vector_checksum", checksum).Msg("computed ML vector checksum")

	board := tensor.New(tensor.WithShape(1, NN_C, NN_H, NN_W),
		tensor.WithBacking(vec[:NN_N_PLANES]))
	scal := tensor.New(tensor.WithShape(1, NN_N_SCAL), tensor.WithBacking(vec[NN_N_PLANES:]))
	model.model.SetInput(0, board)
	model.model.SetInput(1, scal)
	err = model.backend.Run()
	if err != nil {
		return 0, fmt.Errorf("failed to run ONNX model: %w", err)
	}
	output, err := model.model.GetOutputTensors()
	if err != nil {
		return 0, fmt.Errorf("failed to get output tensors: %w", err)
	}
	eval := output[0].Data().(float32) * 300.0 // scale back to -300 to 300 range
	g.onturn = 1 - g.onturn                    // switch turn back to the original player

	g.UnplayLastMove() // this undoes the g.onturn change above.

	return eval, nil
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

// BuildMLVector builds the feature vector for the current game state. Assumes
// the player on turn has just put tiles on a board and tallied up their new
// score but not drawn replacement tiles.
func (g *Game) BuildMLVector() ([]float32, error) {

	tilePlanes := make([]float32, 26*225) // 26 planes for letters A-Z
	isBlankPlane := make([]float32, 225)  // 15x15 board tiles

	// 26 planes for board tiles
	// Each plane is a 15x15 grid, flattened to 225 elements.
	plane := g.Board()
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
					return nil, fmt.Errorf("Invalid tile index %d for tile %d at position (%d, %d)", ll, tile, j, k)
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
			hc := g.Board().GetCrossSet(i, j, board.HorizontalDirection)
			vc := g.Board().GetCrossSet(i, j, board.VerticalDirection)
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
			letter := g.Board().GetLetter(i, j)
			if letter != 0 {
				// This is a letter tile, not an empty square.
				continue
			}
			bonus := g.Board().GetBonus(i, j)

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
				return nil, fmt.Errorf("Unknown bonus type %d at position (%d, %d)", bonus, i, j)
			}
		}
	}

	// 1 vector for our rack leave (size = 27 tiles)
	// 1 vector for unseen tiles in the bag (size = 27 tiles)
	// scalar for score diff after making move
	// scalar for num tiles left in bag
	rackVector := make([]float32, 27)   // 26 letters + 1 for unseen
	unseenVector := make([]float32, 27) // 26 letters + 1 for unseen

	rack := g.RackFor(g.PlayerOnTurn())
	oppRack := g.RackFor(1 - g.PlayerOnTurn())
	g.bag.PutBack(oppRack.TilesOn()) // put back opponent's tiles to calculate all unseen tiles
	bag := g.Bag().PeekMap()
	for i := range 27 {
		rackVector[i] = float32(rack.LetArr[i]) / 7.0 // normalize to 0-1 range
		// technically we can even divide by 12 here i think. (number of Es in the bag)
		// divide by 20 for now. make the vectors "cared about" a little bit sooner.
		unseenVector[i] = float32(bag[i]) / 20.0 // normalize to 0-1 range

	}

	// Note the spread is our spread after making this move. The player on turn
	// switched after playing the move, so that's why we do 1 - gw.game.PlayerOnTurn().
	// We temporarily encode the player whose spread this is for since we don't
	// have that data anywhere else.
	// spreadFor := 1 - g.PlayerOnTurn()
	// spread := float32(g.SpreadFor(spreadFor)) // update spread for opponent

	normalizedSpread := NormalizeSpreadForML(float32(g.SpreadFor(g.PlayerOnTurn())))
	// normalize to -1 to 1 range
	tilesRemaining := float32(g.Bag().TilesRemaining()) / 100.0 // normalize to 0-1 range
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

	return features, nil
}

func BinaryWriteMLVector(w *bufio.Writer, vec []float32) error {
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
