// lookup_bench_test.go: microbenchmarks for the WMP hash lookup hot
// path. Used to measure the impact of low-level optimizations on
// getWordEntry without going through the full sim pipeline.
package wmp

import (
	"os"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
)

// makeBenchLookupKeys returns a deterministic set of BitRack lookup
// keys synthesized from the player's rack tiles in the BenchmarkSim
// position. We pick a mix of keys that hit and miss the WMP so the
// benchmark exercises both branches of the bucket-scan loop.
func makeBenchLookupKeys(t testing.TB, w *WMP) []BitRack {
	t.Helper()
	// Use the WMP's internal entries directly for hits — pull a few
	// real keys from each length so we exercise different bucket
	// distributions. We also synthesize a few BitRacks that won't
	// be in the table for misses.
	var keys []BitRack
	for length := 4; length <= 7; length++ {
		wfl := &w.WFLs[length]
		for i := uint32(0); i < wfl.NumWordEntries && len(keys) < 64; i += wfl.NumWordEntries/8 + 1 {
			keys = append(keys, wfl.WordMapEntries[i].ReadBitRack())
		}
	}
	// Add some misses by mutating one bit of the existing keys.
	for i := 0; i < 16 && i < len(keys); i++ {
		k := keys[i]
		k.Low ^= 0x100
		keys = append(keys, k)
	}
	return keys
}

// loadCSW24WMPForLookupBench loads the CSW24 WMP from MAGPIE's data
// directory if available, or skips the benchmark.
func loadCSW24WMPForLookupBench(t testing.TB) *WMP {
	t.Helper()
	const path = "/Users/john/sources/apr03-noprune/MAGPIE/data/lexica/CSW24.wmp"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("CSW24.wmp not present at %s", path)
	}
	w, err := LoadFromFile("CSW24", path)
	if err != nil {
		t.Fatalf("LoadFromFile: %v", err)
	}
	return w
}

// BenchmarkGetWordEntry measures the cost of the blankless hash
// lookup against a real CSW24 WMP. Each operation is one
// getWordEntry call against length-7 (most populated) followed by
// length-4 to give a mix of bucket sizes.
func BenchmarkGetWordEntry(b *testing.B) {
	w := loadCSW24WMPForLookupBench(b)
	keys := makeBenchLookupKeys(b, w)
	if len(keys) == 0 {
		b.Skip("no keys")
	}
	wfl7 := &w.WFLs[7]
	wfl4 := &w.WFLs[4]
	mask := uint32(1)
	for mask < uint32(len(keys)) {
		mask <<= 1
	}
	mask--
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k := &keys[uint32(i)&mask%uint32(len(keys))]
		_ = wfl7.getWordEntry(k)
		_ = wfl4.getWordEntry(k)
	}
}

// BenchmarkReadBitRack isolates the per-entry BitRack decode in the
// hot loop body. Reading a 16-byte BitRack from an Entry's tail half
// happens at every step of getWordEntry's bucket scan.
func BenchmarkReadBitRack(b *testing.B) {
	w := loadCSW24WMPForLookupBench(b)
	wfl := &w.WFLs[7]
	if wfl.NumWordEntries == 0 {
		b.Skip("no entries")
	}
	entries := wfl.WordMapEntries
	n := uint32(len(entries))
	b.ResetTimer()
	var sink BitRack
	for i := 0; i < b.N; i++ {
		sink = entries[uint32(i)%n].ReadBitRack()
	}
	_ = sink
}

// keep the unused tilemapping import alive in case we extend.
var _ = tilemapping.MachineLetter(0)
