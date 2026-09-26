package triton

import (
	"context"
	"fmt"
	"math"

	grpc_client "github.com/domino14/macondo/triton/grpc-client"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ModelOutputs contains the different outputs from the model
type ModelOutputs struct {
	Value     []float32 // win prediction value
	Spread    []float32 // predicted spread change, tanh-scaled (see game.NormalizeSpreadForML)
	Points    []float32 // predicted points
	BingoProb []float32 // bingo probability
	OppScore  []float32 // opponent score
}

// TritonClient is a client for the Triton Inference Server.
type TritonClient struct {
	client       *grpc_client.GRPCInferenceServiceClient
	modelName    string
	modelVersion string
	outputs      []string // output tensors to request; default just "value"
	debug        bool     // Enable debug logging
}

// NewTritonClient creates a new TritonClient.
func NewTritonClient(serverURL, modelName, modelVersion string) (*TritonClient, error) {
	log.Debug().Msgf("Connecting to Triton server at %s for model %s version %s", serverURL, modelName, modelVersion)
	conn, err := grpc.NewClient(serverURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("could not connect to triton server: %w", err)
	}

	client := grpc_client.NewGRPCInferenceServiceClient(conn)

	return &TritonClient{
		client:       &client,
		modelName:    modelName,
		modelVersion: modelVersion,
		outputs:      []string{"value"},
		debug:        false, // Set to true for detailed debugging
	}, nil
}

// SetOutputs chooses which output tensors to request. Every name must be an
// output of the served model; "value" is always available, "spread" only on
// multi-head exports.
func (c *TritonClient) SetOutputs(names []string) {
	c.outputs = names
}

// SetDebug enables or disables debug logging
func (c *TritonClient) SetDebug(debug bool) {
	c.debug = debug
}

// Infer performs inference using the Triton server.
func (c *TritonClient) Infer(boardTensorData []float32, scalarTensorData []float32, numMoves int) (*ModelOutputs, error) {
	if c.debug {
		log.Debug().Msgf("Inferring %d moves", numMoves)
	}

	boardInput := &grpc_client.ModelInferRequest_InferInputTensor{
		Name:     "board",
		Datatype: "FP32",
		Shape:    []int64{int64(numMoves), 85, 15, 15},
	}
	scalarInput := &grpc_client.ModelInferRequest_InferInputTensor{
		Name:     "scalars",
		Datatype: "FP32",
		Shape:    []int64{int64(numMoves), 72},
	}

	requested := make([]*grpc_client.ModelInferRequest_InferRequestedOutputTensor, len(c.outputs))
	for i, name := range c.outputs {
		requested[i] = &grpc_client.ModelInferRequest_InferRequestedOutputTensor{Name: name}
	}

	inferRequest := &grpc_client.ModelInferRequest{
		ModelName:    c.modelName,
		ModelVersion: c.modelVersion,
		Inputs:       []*grpc_client.ModelInferRequest_InferInputTensor{boardInput, scalarInput},
		Outputs:      requested,
	}
	inferRequest.RawInputContents = append(inferRequest.RawInputContents, float32ToByte(boardTensorData), float32ToByte(scalarTensorData))

	if c.debug {
		log.Debug().Msgf("Sending inference request for %d moves", numMoves)
	}

	inferResponse, err := (*c.client).ModelInfer(context.Background(), inferRequest)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Debug output shapes
	if c.debug {
		for i, output := range inferResponse.Outputs {
			log.Debug().Msgf("Output %d: %s, shape: %v", i, output.Name, output.Shape)
		}
	}

	// Outputs come back in the order requested, but map them by name anyway.
	outputs := &ModelOutputs{}
	for i, out := range inferResponse.Outputs {
		if i >= len(inferResponse.RawOutputContents) {
			return nil, fmt.Errorf("output %s has no raw contents", out.Name)
		}
		data := byteToFloat32(inferResponse.RawOutputContents[i])
		switch out.Name {
		case "value":
			outputs.Value = data
		case "spread":
			outputs.Spread = data
		}
	}
	if outputs.Value == nil {
		return nil, fmt.Errorf("model %s returned no value output", c.modelName)
	}

	return outputs, nil
}

// InferWithDiagnostics performs inference and includes detailed diagnostics about output values
func (c *TritonClient) InferWithDiagnostics(boardTensorData []float32, scalarTensorData []float32, numMoves int) (*ModelOutputs, map[string]interface{}, error) {
	// Enable debug temporarily
	originalDebug := c.debug
	c.debug = true

	outputs, err := c.Infer(boardTensorData, scalarTensorData, numMoves)
	if err != nil {
		return nil, nil, err
	}

	// Create diagnostics
	diagnostics := make(map[string]interface{})

	// Check value ranges
	if len(outputs.Value) > 0 {
		valueStats := analyzeFloatArray(outputs.Value)
		diagnostics["value_stats"] = valueStats

		// Count out-of-range values
		outOfRangeCount := 0
		for _, v := range outputs.Value {
			if v < -1 || v > 1 {
				outOfRangeCount++
			}
		}
		diagnostics["value_out_of_range_count"] = outOfRangeCount

		// Sample out-of-range values (up to 5)
		outOfRangeSamples := make([]float32, 0)
		count := 0
		for _, v := range outputs.Value {
			if (v < -1 || v > 1) && count < 5 {
				outOfRangeSamples = append(outOfRangeSamples, v)
				count++
			}
		}
		diagnostics["value_out_of_range_samples"] = outOfRangeSamples
	}

	// Check output shapes
	outputShapes := map[string]int{
		"value": len(outputs.Value),
	}

	// Add optional outputs if they exist
	if len(outputs.Points) > 0 {
		outputShapes["total_game_points"] = len(outputs.Points)
	}
	if len(outputs.BingoProb) > 0 {
		outputShapes["opp_bingo_prob"] = len(outputs.BingoProb)
	}
	if len(outputs.OppScore) > 0 {
		outputShapes["opp_score"] = len(outputs.OppScore)
	}

	diagnostics["output_shapes"] = outputShapes

	// Restore original debug setting
	c.debug = originalDebug

	return outputs, diagnostics, nil
}

// Helper function to analyze a float array
func analyzeFloatArray(data []float32) map[string]interface{} {
	if len(data) == 0 {
		return map[string]interface{}{
			"error": "empty array",
		}
	}

	min, max := data[0], data[0]
	sum := float64(0)

	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += float64(v)
	}

	avg := sum / float64(len(data))

	return map[string]interface{}{
		"min":   min,
		"max":   max,
		"avg":   avg,
		"count": len(data),
	}
}

func float32ToByte(f []float32) []byte {
	b := make([]byte, 4*len(f))
	for i, v := range f {
		u := math.Float32bits(v)
		b[4*i+0] = byte(u)
		b[4*i+1] = byte(u >> 8)
		b[4*i+2] = byte(u >> 16)
		b[4*i+3] = byte(u >> 24)
	}
	return b
}

func byteToFloat32(b []byte) []float32 {
	f := make([]float32, len(b)/4)
	for i := range f {
		u := uint32(b[4*i+0]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24
		f[i] = math.Float32frombits(u)
	}
	return f
}
