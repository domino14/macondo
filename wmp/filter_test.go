package wmp

import (
	"math/rand"
	"testing"
)

// TestFilterAgreesWithTable checks the blankless lookup filter against the
// table it stands in front of, on a whole lexicon's word map: every entry
// must be found through the filter, and a lookup must give the same answer
// with the filter as without, for random racks, most of which have no
// words.
func TestFilterAgreesWithTable(t *testing.T) {
	w, err := LoadFromFile("CSW24", wmpTestFilePath(t))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	rejected, random := 0, 0
	for length := 2; length <= int(w.BoardDim); length++ {
		wfl := &w.WFLs[length]
		if len(wfl.WordMapEntries) != 0 && len(wfl.wordFilter) == 0 {
			t.Fatalf("length %d: no filter", length)
		}
		for i := range wfl.WordMapEntries {
			br := wfl.WordMapEntries[i].ReadBitRack()
			if wfl.getWordEntry(&br) != &wfl.WordMapEntries[i] {
				t.Fatalf("length %d: entry %d not found through the filter", length, i)
			}
			if !wfl.mayHaveWords(br.MixTo64()) {
				t.Fatalf("length %d: entry %d turned away by the filter", length, i)
			}
		}
		plain := *wfl
		plain.wordFilter = nil
		for n := 0; n < 20000; n++ {
			var br BitRack
			for k := 0; k < length; k++ {
				br.AddLetter(byte(1 + rng.Intn(26)))
			}
			got, want := wfl.getWordEntry(&br), plain.getWordEntry(&br)
			if got != want {
				t.Fatalf("length %d, rack %v: %p with the filter, %p without", length, br, got, want)
			}
			random++
			if !wfl.mayHaveWords(br.MixTo64()) {
				if want != nil {
					t.Fatalf("length %d, rack %v: turned away but has words", length, br)
				}
				rejected++
			}
		}
	}
	t.Logf("%d random racks, %d turned away by the filter", random, rejected)
	if rejected == 0 {
		t.Fatal("the filter turned nothing away")
	}
}
