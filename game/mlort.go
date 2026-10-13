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
// under the strategy directory (lexicon directory first, then default).

var (
	ortInitOnce sync.Once
	ortInitErr  error
	ortMu       sync.Mutex
	ortModels   = map[string]*ortModel{}
)

type ortModel struct {
	sess    *ort.DynamicAdvancedSession
	outputs []string
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
	name := g.config.GetString(config.ConfigTritonModelName)
	ver := g.config.GetString(config.ConfigTritonModelVersion)
	if ver == "" {
		ver = "1"
	}
	file := fmt.Sprintf("models/%s/%s/model.onnx", name, ver)
	key := g.LexiconName() + ":" + file

	ortMu.Lock()
	defer ortMu.Unlock()
	if m := ortModels[key]; m != nil {
		return m, nil
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
	m := &ortModel{}
	inNames := []string{ins[0].Name, ins[1].Name}
	for _, o := range outs {
		if o.Name == "value" || o.Name == "spread" {
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

func (g *Game) mlevaluateMovesLocal(nmoves int, planeVectors, scalarVectors []float32) (*triton.ModelOutputs, error) {
	m, err := g.ortModelFor()
	if err != nil {
		return nil, err
	}
	board, err := ort.NewTensor(ort.NewShape(int64(nmoves), NN_C, NN_H, NN_W), planeVectors[:nmoves*NN_N_PLANES])
	if err != nil {
		return nil, err
	}
	defer board.Destroy()
	scal, err := ort.NewTensor(ort.NewShape(int64(nmoves), NN_N_SCAL), scalarVectors[:nmoves*NN_N_SCAL])
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
		if name == "spread" {
			mo.Spread = v
		} else {
			mo.Value = v
		}
	}
	return mo, nil
}
