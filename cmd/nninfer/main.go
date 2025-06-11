package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

const (
	C        = 83
	H, W     = 15, 15
	N_PLANES = C * H * W // 18 675
	N_SCAL   = 56
	RowLen   = N_PLANES + N_SCAL
)

// readBinaryVector reads a single vector from the stream.
func readBinaryVector(r io.Reader) []float32 {
	vec := make([]float32, RowLen)
	// The binary data is expected to be in little-endian format.
	err := binary.Read(r, binary.LittleEndian, &vec)
	if err != nil {
		if err == io.EOF {
			return nil
		}
		log.Fatalf("Failed to read vector: %v", err)
	}
	return vec
}

func main() {
	// 1) Create a backend and a model
	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)

	// 2) Read the .onnx file
	b, err := os.ReadFile("../pytorch/macondo-nn.onnx")
	if err != nil {
		log.Fatalf("Failed to read onnx file: %v", err)
	}
	err = model.UnmarshalBinary(b)
	if err != nil {
		log.Fatalf("Failed to unmarshal onnx model: %v", err)
	}

	// 3) Read feature vector from stdin
	vec := readBinaryVector(os.Stdin)
	if vec == nil {
		fmt.Println("No input received.")
		return
	}

	// 4) Create input tensors
	board := tensor.New(tensor.WithShape(1, C, H, W), tensor.WithBacking(vec[:N_PLANES]))
	scal := tensor.New(tensor.WithShape(1, N_SCAL), tensor.WithBacking(vec[N_PLANES:]))

	model.SetInput(0, board)
	model.SetInput(1, scal)

	// 5) Run inference
	err = backend.Run()
	if err != nil {
		log.Fatalf("Failed to run inference: %v", err)
	}

	// 6) Get the output
	output, err := model.GetOutputTensors()
	if err != nil {
		log.Fatalf("Failed to get output tensors: %v", err)
	}

	// 7) Process result
	// The output is a tensor, so we access its data and cast it.
	delta := output[0].Data().([]float32)[0] * 300.0
	fmt.Printf("Predicted Δ-spread_k = %.2f pts\n", delta)
}
