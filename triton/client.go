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

// TritonClient is a client for the Triton Inference Server.
type TritonClient struct {
	client       *grpc_client.GRPCInferenceServiceClient
	modelName    string
	modelVersion string
}

// NewTritonClient creates a new TritonClient.
func NewTritonClient(serverURL, modelName, modelVersion string) (*TritonClient, error) {
	log.Info().Msgf("Connecting to Triton server at %s for model %s version %s", serverURL, modelName, modelVersion)
	conn, err := grpc.NewClient(serverURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("could not connect to triton server: %w", err)
	}

	client := grpc_client.NewGRPCInferenceServiceClient(conn)

	return &TritonClient{
		client:       &client,
		modelName:    modelName,
		modelVersion: modelVersion,
	}, nil
}

// Infer performs inference using the Triton server.
func (c *TritonClient) Infer(boardTensorData []float32, scalarTensorData []float32, numMoves int) ([]float32, error) {
	boardInput := &grpc_client.ModelInferRequest_InferInputTensor{
		Name:     "board",
		Datatype: "FP32",
		Shape:    []int64{int64(numMoves), 85, 15, 15},
	}
	scalarInput := &grpc_client.ModelInferRequest_InferInputTensor{
		Name:     "scalars",
		Datatype: "FP32",
		Shape:    []int64{int64(numMoves), 78},
	}

	inferRequest := &grpc_client.ModelInferRequest{
		ModelName:    c.modelName,
		ModelVersion: c.modelVersion,
		Inputs:       []*grpc_client.ModelInferRequest_InferInputTensor{boardInput, scalarInput},
	}
	inferRequest.RawInputContents = append(inferRequest.RawInputContents, float32ToByte(boardTensorData), float32ToByte(scalarTensorData))

	inferResponse, err := (*c.client).ModelInfer(context.Background(), inferRequest)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	outputData := byteToFloat32(inferResponse.RawOutputContents[0])

	return outputData, nil
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
