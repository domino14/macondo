//go:build cgo

package game

import (
	"fmt"
	"io"
	"runtime"
	"sync"

	"github.com/rs/zerolog/log"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/dataloaders"
	"github.com/domino14/macondo/triton"
)

// Local (no Triton) inference runs the model's ONNX file on the CPU with
// ONNX Runtime. It needs the ONNX Runtime shared library (version 1.29,
// github.com/microsoft/onnxruntime/releases): onnxruntime-lib gives its
// path, otherwise the system's library search path is used. The model is
// the one Triton would serve: models/<triton-model-name>/<version>/model.onnx
// under the strategy directory (lexicon directory first, then default),
// downloaded from ml-models-url when it is not there (see mlfetch.go).

var (
	ortInitOnce sync.Once
	ortInitErr  error
	ortMu       sync.Mutex
	ortModels   = map[string]*ortModel{}
)

type ortModel struct {
	name            string
	sess            *ort.DynamicAdvancedSession
	outputs         []string
	planes, scalars int // the inputs' per-position sizes, 0 if unknown
}

func ortLibrary(cfg *config.Config) string {
	if p := cfg.GetString(config.ConfigOnnxRuntimeLib); p != "" {
		return p
	}
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.dylib"
	case "windows":
		return "onnxruntime.dll"
	}
	return "libonnxruntime.so"
}

func (g *Game) ortModelFor() (*ortModel, error) {
	ortInitOnce.Do(func() {
		ort.SetSharedLibraryPath(ortLibrary(g.config))
		ortInitErr = ort.InitializeEnvironment()
	})
	if ortInitErr != nil {
		return nil, fmt.Errorf("ONNX Runtime (set onnxruntime-lib to libonnxruntime's path): %w", ortInitErr)
	}
	name, err := g.MLModelName()
	if err != nil {
		return nil, err
	}
	ver := mlModelVersion(g.config)
	file := mlModelFile(name, ver)
	key := g.LexiconName() + ":" + file

	ortMu.Lock()
	defer ortMu.Unlock()
	if m := ortModels[key]; m != nil {
		return m, nil
	}
	if err := ensureMLModel(g.config, name, ver, g.LexiconName()); err != nil {
		return nil, fmt.Errorf("no model %s for %s: %w", name, g.LexiconName(), err)
	}
	r, err := dataloaders.StratFileForLexicon(dataloaders.StrategyParamsPath(g.config.WGLConfig()), file, g.LexiconName())
	if err != nil {
		return nil, fmt.Errorf("loading %s: %w", file, err)
	}
	data, err := io.ReadAll(r)
	r.Close()
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", file, err)
	}
	ins, outs, err := ort.GetInputOutputInfoWithONNXData(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", file, err)
	}
	if len(ins) != 2 {
		return nil, fmt.Errorf("%s: expected 2 inputs (board, scalars), got %d", file, len(ins))
	}
	m := &ortModel{name: name}
	if d := ins[0].Dimensions; len(d) == 4 {
		m.planes = int(d[1]) // (batch, planes, rows, cols)
	}
	if d := ins[1].Dimensions; len(d) == 2 {
		m.scalars = int(d[1])
	}
	inNames := []string{ins[0].Name, ins[1].Name}
	for _, o := range outs {
		switch o.Name {
		case "value", "spread", "opp_bingo_prob", "opp_score":
			m.outputs = append(m.outputs, o.Name)
		}
	}
	if len(m.outputs) == 0 {
		// Older models: a single unnamed-by-convention value output.
		m.outputs = []string{outs[0].Name}
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	if n := g.config.GetInt(config.ConfigOnnxThreads); n > 0 {
		if err := opts.SetIntraOpNumThreads(n); err != nil {
			return nil, err
		}
	}
	m.sess, err = ort.NewDynamicAdvancedSessionWithONNXData(data, inNames, m.outputs, opts)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", file, err)
	}
	log.Info().Str("model", file).Strs("outputs", m.outputs).Msg("loaded ONNX model for CPU inference")
	ortModels[key] = m
	return m, nil
}

func (g *Game) mlevaluateMovesLocal(lay *MLLayout, nmoves int, planeVectors, scalarVectors []float32) (*triton.ModelOutputs, error) {
	m, err := g.ortModelFor()
	if err != nil {
		return nil, err
	}
	if err := checkMLModelShape(m.name, func() (int, int) { return m.planes, m.scalars }, lay, g.letterDistribution.Name); err != nil {
		return nil, err
	}
	board, err := ort.NewTensor(ort.NewShape(int64(nmoves), int64(lay.NumPlanes), NN_H, NN_W), planeVectors[:nmoves*lay.PlaneLen])
	if err != nil {
		return nil, err
	}
	defer board.Destroy()
	scal, err := ort.NewTensor(ort.NewShape(int64(nmoves), int64(lay.NumScalars)), scalarVectors[:nmoves*lay.NumScalars])
	if err != nil {
		return nil, err
	}
	defer scal.Destroy()
	outs := make([]ort.Value, len(m.outputs)) // nil: allocated by the run
	if err := m.sess.Run([]ort.Value{board, scal}, outs); err != nil {
		return nil, fmt.Errorf("running the ONNX model: %w", err)
	}
	mo := &triton.ModelOutputs{}
	for i, name := range m.outputs {
		t, ok := outs[i].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("output %s: unexpected type %T", name, outs[i])
		}
		v := append([]float32(nil), t.GetData()...)
		t.Destroy()
		switch name {
		case "spread":
			mo.Spread = v
		case "opp_bingo_prob":
			mo.BingoProb = v
		case "opp_score":
			mo.OppScore = v
		default:
			mo.Value = v
		}
	}
	return mo, nil
}
