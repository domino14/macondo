//go:build !cgo

package game

import (
	"errors"

	"github.com/domino14/macondo/triton"
)

// Without cgo (the Lambda build) there is no ONNX Runtime: the net is
// served by Triton only.
func (g *Game) mlevaluateMovesLocal(lay *MLLayout, nmoves int, planeVectors, scalarVectors []float32) (*triton.ModelOutputs, error) {
	return nil, errors.New("local ONNX inference needs a cgo build; use Triton (triton-use-triton)")
}
