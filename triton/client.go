package triton

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
	"unsafe"

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
	BingoProb []float32 // P(opponent bingos on the next turn)
	OppScore  []float32 // opponent's predicted next-turn score, points
}

// TritonClient is a client for the Triton Inference Server.
type TritonClient struct {
	client       *grpc_client.GRPCInferenceServiceClient
	serverURL    string
	modelName    string
	modelVersion string
	outputs      []string // output tensors to request; default just "value"
	boardUint8   bool     // the model takes the board as uint8 (all planes are 0/1)
	boardBuf     []byte   // reused uint8 board buffer
	debug        bool     // Enable debug logging
}

// modelInfo is what DetectOutputs learns from a model's metadata.
type modelInfo struct {
	outputs    []string
	boardUint8 bool
}

// connsPerServer bounds the gRPC connections a process opens to one
// server. Games used to open a connection each and never close it, so a
// long replay (one game per position) ran Triton out of file descriptors;
// connections are now shared, a few per server so parallel bots are not
// all multiplexed onto one.
const connsPerServer = 16

var (
	connMu   sync.Mutex
	conns    = map[string][]*grpc.ClientConn{}
	connNext = map[string]int{}
	// Detected model outputs and board type by url/model/version, so each
	// game does not ask the server again.
	infoByModel = map[string]modelInfo{}
)

func sharedConn(serverURL string) (*grpc.ClientConn, error) {
	connMu.Lock()
	defer connMu.Unlock()
	if len(conns[serverURL]) < connsPerServer {
		conn, err := grpc.NewClient(serverURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			return nil, err
		}
		conns[serverURL] = append(conns[serverURL], conn)
		return conn, nil
	}
	i := connNext[serverURL] % connsPerServer
	connNext[serverURL] = i + 1
	return conns[serverURL][i], nil
}

// NewTritonClient creates a new TritonClient. Clients share a small pool
// of connections per server (see connsPerServer).
func NewTritonClient(serverURL, modelName, modelVersion string) (*TritonClient, error) {
	log.Debug().Msgf("Connecting to Triton server at %s for model %s version %s", serverURL, modelName, modelVersion)
	conn, err := sharedConn(serverURL)
	if err != nil {
		return nil, fmt.Errorf("could not connect to triton server: %w", err)
	}

	client := grpc_client.NewGRPCInferenceServiceClient(conn)

	return &TritonClient{
		client:       &client,
		serverURL:    serverURL,
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

// Outputs is the list of output tensors the client requests.
func (c *TritonClient) Outputs() []string { return c.outputs }

// DetectOutputs asks the server which outputs the served model has and
// requests "value" plus "spread" when the model exports it (the multi-head
// nets do; the old CNN does not). The bot's decided-game ranking needs the
// spread head, so this is called once when the game's client is created.
// It also notes whether the model takes the board as UINT8 (exported with
// export.py --board-uint8), so Infer sends a quarter of the bytes.
func (c *TritonClient) DetectOutputs() error {
	key := c.serverURL + "/" + c.modelName + "/" + c.modelVersion
	connMu.Lock()
	known, ok := infoByModel[key]
	connMu.Unlock()
	if ok {
		c.outputs, c.boardUint8 = known.outputs, known.boardUint8
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	md, err := (*c.client).ModelMetadata(ctx, &grpc_client.ModelMetadataRequest{
		Name: c.modelName, Version: c.modelVersion})
	if err != nil {
		return err
	}
	outputs := []string{"value"}
	for _, o := range md.Outputs {
		switch o.Name {
		case "spread", "opp_bingo_prob", "opp_score":
			outputs = append(outputs, o.Name)
		}
	}
	boardUint8 := false
	for _, in := range md.Inputs {
		if in.Name == "board" && in.Datatype == "UINT8" {
			boardUint8 = true
		}
	}
	c.outputs, c.boardUint8 = outputs, boardUint8
	connMu.Lock()
	infoByModel[key] = modelInfo{outputs, boardUint8}
	connMu.Unlock()
	return nil
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
	boardBytes := float32ToByte(boardTensorData)
	if c.boardUint8 {
		boardInput.Datatype = "UINT8"
		c.boardBuf = boardToUint8(c.boardBuf, boardTensorData)
		boardBytes = c.boardBuf
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
	inferRequest.RawInputContents = append(inferRequest.RawInputContents, boardBytes, float32ToByte(scalarTensorData))

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
		case "opp_bingo_prob":
			outputs.BingoProb = data
		case "opp_score":
			outputs.OppScore = data
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

// float32ToByte views f as its little-endian bytes without copying (the
// wire format Triton expects; every platform we build for is little-endian).
// The result aliases f: callers must not change f until the request is sent.
func float32ToByte(f []float32) []byte {
	if len(f) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), 4*len(f))
}

// boardToUint8 packs the board planes (every value 0 or 1) one byte per
// value into buf, growing it as needed.
func boardToUint8(buf []byte, f []float32) []byte {
	if cap(buf) < len(f) {
		buf = make([]byte, len(f))
	}
	buf = buf[:len(f)]
	for i, v := range f {
		buf[i] = uint8(v)
	}
	return buf
}

func byteToFloat32(b []byte) []float32 {
	f := make([]float32, len(b)/4)
	for i := range f {
		u := uint32(b[4*i+0]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24
		f[i] = math.Float32frombits(u)
	}
	return f
}
