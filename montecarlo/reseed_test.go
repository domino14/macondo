package montecarlo

import (
	"testing"

	"lukechampine.com/frand"
)

// reseedRNG reaches into frand's RNG; this is the check that a reseeded
// source still gives what a new one with that seed gives.
func TestReseedRNG(t *testing.T) {
	var a, b [32]byte
	a[0], b[0] = 1, 2
	r := frand.NewCustom(a[:], 0, 0)
	// Leave it partway through its buffers.
	r.Uint64n(1000)
	var junk [3]byte
	r.Read(junk[:])
	reseedRNG(r, b)
	want := frand.NewCustom(b[:], 0, 0)
	for i := 0; i < 2000; i++ {
		if g, w := r.Uint64n(1<<40), want.Uint64n(1<<40); g != w {
			t.Fatalf("draw %d: %d from the reseeded source, %d from a new one", i, g, w)
		}
	}
	var gb, wb [37]byte
	r.Read(gb[:])
	want.Read(wb[:])
	if gb != wb {
		t.Fatal("bytes read differ")
	}
}
