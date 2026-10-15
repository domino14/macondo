package triton

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestFloat32ToByteIsLittleEndianFP32(t *testing.T) {
	f := []float32{0, 1, -2.5, float32(math.Pi), math.MaxFloat32, 1e-30}
	want := make([]byte, 4*len(f))
	for i, v := range f {
		binary.LittleEndian.PutUint32(want[4*i:], math.Float32bits(v))
	}
	if got := float32ToByte(f); !bytes.Equal(got, want) {
		t.Fatalf("float32ToByte = %v, want %v", got, want)
	}
	if back := byteToFloat32(float32ToByte(f)); len(back) != len(f) || back[3] != f[3] {
		t.Fatalf("round trip: %v", back)
	}
	if float32ToByte(nil) != nil {
		t.Fatal("empty input should give nil")
	}
}
