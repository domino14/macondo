package wmp

import (
	"bytes"
	"encoding/binary"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

// wmpTestFilePath returns the path to a CSW24 WMP file for testing.
// Derives the path from $MACONDO_DATA_PATH/lexica/CSW24.wmp.
// Skips the test if MACONDO_DATA_PATH is unset or the file is absent.
func wmpTestFilePath(t *testing.T) string {
	t.Helper()
	dp := os.Getenv("MACONDO_DATA_PATH")
	if dp == "" {
		t.Skip("MACONDO_DATA_PATH not set; skipping WMP tests that need CSW24.wmp")
	}
	p := dp + "/lexica/CSW24.wmp"
	if _, err := os.Stat(p); err != nil {
		t.Skipf("CSW24.wmp not found at %s", p)
	}
	return p
}

// englishLetterDistributionCSV is a minimal English letter distribution
// in word-golib's CSV format (letter,quantity,value,vowel). The blank
// is at row 0 so it gets MachineLetter 0; A=1, B=2, …, Z=26.
const englishLetterDistributionCSV = `?,2,0,0
A,9,1,1
B,2,3,0
C,2,3,0
D,4,2,0
E,12,1,1
F,2,4,0
G,3,2,0
H,2,4,0
I,9,1,1
J,1,8,0
K,1,5,0
L,4,1,0
M,2,3,0
N,6,1,0
O,8,1,1
P,2,3,0
Q,1,10,0
R,6,1,0
S,4,1,0
T,6,1,0
U,4,1,1
V,2,4,0
W,2,4,0
X,1,8,0
Y,2,4,0
Z,1,10,0
`

// testEnglishLD builds an in-memory English LetterDistribution for tests.
func testEnglishLD(t *testing.T) *tilemapping.LetterDistribution {
	t.Helper()
	ld, err := tilemapping.ScanLetterDistribution(strings.NewReader(englishLetterDistributionCSV))
	if err != nil {
		t.Fatalf("ScanLetterDistribution failed: %v", err)
	}
	return ld
}

// stringsToMachineWords converts plain ASCII strings (uppercase A-Z) to
// MachineWords assuming the standard English mapping (A=1, B=2, ..., Z=26).
func stringsToMachineWords(words []string) []tilemapping.MachineWord {
	out := make([]tilemapping.MachineWord, len(words))
	for i, w := range words {
		mw := make(tilemapping.MachineWord, len(w))
		for j, c := range w {
			mw[j] = tilemapping.MachineLetter(c - 'A' + 1)
		}
		out[i] = mw
	}
	return out
}

// machineWordToString converts a MachineWord back to a plain ASCII string.
func machineWordToString(mw []byte) string {
	out := make([]byte, len(mw))
	for i, ml := range mw {
		out[i] = ml + 'A' - 1
	}
	return string(out)
}

// extractWordsFromBuffer parses a flat buffer of length*N bytes into N
// word strings, sorted alphabetically (for deterministic comparison).
func extractWordsFromBuffer(buf []byte, length int) []string {
	if len(buf) == 0 || length == 0 {
		return nil
	}
	n := len(buf) / length
	out := make([]string, n)
	for i := 0; i < n; i++ {
		out[i] = machineWordToString(buf[i*length : (i+1)*length])
	}
	sort.Strings(out)
	return out
}

const testBoardDim = 15

func TestBitRackBasics(t *testing.T) {
	br := BitRackFromMachineWord(stringsToMachineWords([]string{"CAT"})[0])
	if br.GetLetter(byte('C'-'A'+1)) != 1 {
		t.Errorf("expected C count 1, got %d", br.GetLetter(byte('C'-'A'+1)))
	}
	if br.GetLetter(byte('A'-'A'+1)) != 1 {
		t.Errorf("expected A count 1, got %d", br.GetLetter(byte('A'-'A'+1)))
	}
	if br.GetLetter(byte('T'-'A'+1)) != 1 {
		t.Errorf("expected T count 1, got %d", br.GetLetter(byte('T'-'A'+1)))
	}
	if br.GetLetter(byte('B'-'A'+1)) != 0 {
		t.Errorf("expected B count 0, got %d", br.GetLetter(byte('B'-'A'+1)))
	}

	// Anagram check: ACT and CAT and TAC should hash to identical BitRacks.
	br2 := BitRackFromMachineWord(stringsToMachineWords([]string{"ACT"})[0])
	br3 := BitRackFromMachineWord(stringsToMachineWords([]string{"TAC"})[0])
	if !br.Equals(&br2) || !br.Equals(&br3) {
		t.Errorf("anagrams should produce equal BitRacks")
	}
}

func TestBitRackAddTake(t *testing.T) {
	var br BitRack
	br.AddLetter(5) // E
	br.AddLetter(5)
	br.AddLetter(5)
	if br.GetLetter(5) != 3 {
		t.Errorf("expected E count 3, got %d", br.GetLetter(5))
	}
	br.TakeLetter(5)
	if br.GetLetter(5) != 2 {
		t.Errorf("expected E count 2, got %d", br.GetLetter(5))
	}
	br.SetLetterCount(5, 0)
	if br.GetLetter(5) != 0 {
		t.Errorf("expected E count 0, got %d", br.GetLetter(5))
	}
}

func TestMakeAndLookupBlankless(t *testing.T) {
	// A small word list with multiple anagram groups
	words := stringsToMachineWords([]string{
		"CAT", "ACT", "TAB", "BAT", "DOG", "GOD",
		"CARE", "RACE", "ACRE", "PEAR", "REAP",
		"AERIAL",
	})
	wmp, err := MakeFromWords(words, testEnglishLD(t), testBoardDim, 1)
	if err != nil {
		t.Fatalf("MakeFromWords failed: %v", err)
	}

	// Look up CAT (rack {C,A,T}) — should return CAT and ACT
	br := BitRackFromMachineWord(stringsToMachineWords([]string{"CAT"})[0])
	buf := make([]byte, ResultBufferSize)
	n := wmp.WriteWordsToBuffer(&br, 3, buf)
	got := extractWordsFromBuffer(buf[:n], 3)
	want := []string{"ACT", "CAT"}
	if !sliceEqual(got, want) {
		t.Errorf("CAT lookup: got %v, want %v", got, want)
	}

	// CARE rack — should return ACRE, CARE, RACE
	br = BitRackFromMachineWord(stringsToMachineWords([]string{"CARE"})[0])
	n = wmp.WriteWordsToBuffer(&br, 4, buf)
	got = extractWordsFromBuffer(buf[:n], 4)
	want = []string{"ACRE", "CARE", "RACE"}
	if !sliceEqual(got, want) {
		t.Errorf("CARE lookup: got %v, want %v", got, want)
	}

	// PEAR rack — should return PEAR, REAP
	br = BitRackFromMachineWord(stringsToMachineWords([]string{"PEAR"})[0])
	n = wmp.WriteWordsToBuffer(&br, 4, buf)
	got = extractWordsFromBuffer(buf[:n], 4)
	want = []string{"PEAR", "REAP"}
	if !sliceEqual(got, want) {
		t.Errorf("PEAR lookup: got %v, want %v", got, want)
	}

	// Long word: AERIAL (6 letters, will not be inlined since 16/6 = 2)
	br = BitRackFromMachineWord(stringsToMachineWords([]string{"AERIAL"})[0])
	n = wmp.WriteWordsToBuffer(&br, 6, buf)
	got = extractWordsFromBuffer(buf[:n], 6)
	want = []string{"AERIAL"}
	if !sliceEqual(got, want) {
		t.Errorf("AERIAL lookup: got %v, want %v", got, want)
	}

	// Negative: rack that has no anagrams
	br = BitRackFromMachineWord(stringsToMachineWords([]string{"XYZ"})[0])
	n = wmp.WriteWordsToBuffer(&br, 3, buf)
	if n != 0 {
		t.Errorf("XYZ lookup: expected 0 bytes, got %d", n)
	}
}

func TestMakeAndLookupSingleBlank(t *testing.T) {
	words := stringsToMachineWords([]string{
		"CAT", "BAT", "RAT", "EAT", "OAT",
	})
	wmp, err := MakeFromWords(words, testEnglishLD(t), testBoardDim, 1)
	if err != nil {
		t.Fatalf("MakeFromWords failed: %v", err)
	}

	// Rack: ?AT (one blank + A + T) — should match CAT, BAT, RAT, EAT, OAT
	var br BitRack
	br.AddLetter(0) // blank
	br.AddLetter(byte('A' - 'A' + 1))
	br.AddLetter(byte('T' - 'A' + 1))

	buf := make([]byte, ResultBufferSize)
	n := wmp.WriteWordsToBuffer(&br, 3, buf)
	got := extractWordsFromBuffer(buf[:n], 3)
	want := []string{"BAT", "CAT", "EAT", "OAT", "RAT"}
	if !sliceEqual(got, want) {
		t.Errorf("?AT lookup: got %v, want %v", got, want)
	}
}

func TestMakeAndLookupDoubleBlank(t *testing.T) {
	words := stringsToMachineWords([]string{
		"CAT", "BAT", "RAT", "EAT",
	})
	wmp, err := MakeFromWords(words, testEnglishLD(t), testBoardDim, 1)
	if err != nil {
		t.Fatalf("MakeFromWords failed: %v", err)
	}

	// Rack: ??T (two blanks + T) — should match all four words.
	var br BitRack
	br.AddLetter(0)
	br.AddLetter(0)
	br.AddLetter(byte('T' - 'A' + 1))

	buf := make([]byte, ResultBufferSize)
	n := wmp.WriteWordsToBuffer(&br, 3, buf)
	got := extractWordsFromBuffer(buf[:n], 3)
	want := []string{"BAT", "CAT", "EAT", "RAT"}
	if !sliceEqual(got, want) {
		t.Errorf("??T lookup: got %v, want %v", got, want)
	}
}

func TestMakeFromWordsParallel(t *testing.T) {
	// Same data, different thread counts should produce same lookups.
	words := stringsToMachineWords([]string{
		"CAT", "ACT", "DOG", "GOD", "PEAR", "REAP", "CARE", "RACE", "ACRE",
		"AERIAL", "RETINAS", "RETAINS", "RATINES", "RATINE",
	})
	w1, err := MakeFromWords(words, testEnglishLD(t), testBoardDim, 1)
	if err != nil {
		t.Fatalf("single-threaded build failed: %v", err)
	}
	w4, err := MakeFromWords(words, testEnglishLD(t), testBoardDim, 4)
	if err != nil {
		t.Fatalf("multi-threaded build failed: %v", err)
	}

	br := BitRackFromMachineWord(stringsToMachineWords([]string{"RETINAS"})[0])
	buf1 := make([]byte, ResultBufferSize)
	buf4 := make([]byte, ResultBufferSize)
	n1 := w1.WriteWordsToBuffer(&br, 7, buf1)
	n4 := w4.WriteWordsToBuffer(&br, 7, buf4)
	got1 := extractWordsFromBuffer(buf1[:n1], 7)
	got4 := extractWordsFromBuffer(buf4[:n4], 7)
	if !sliceEqual(got1, got4) {
		t.Errorf("parallel mismatch: single=%v, multi=%v", got1, got4)
	}
	want := []string{"RATINES", "RETAINS", "RETINAS"}
	if !sliceEqual(got1, want) {
		t.Errorf("RETINAS lookup: got %v, want %v", got1, want)
	}
}

func TestRoundTripBinary(t *testing.T) {
	words := stringsToMachineWords([]string{
		"CAT", "ACT", "TAB", "BAT", "DOG", "GOD",
		"CARE", "RACE", "ACRE", "PEAR", "REAP", "AERIAL",
	})
	wmp, err := MakeFromWords(words, testEnglishLD(t), testBoardDim, 1)
	if err != nil {
		t.Fatalf("MakeFromWords failed: %v", err)
	}

	// Write and read back
	var buf bytes.Buffer
	if err := wmp.Write(&buf); err != nil {
		t.Fatalf("Write failed: %v", err)
	}
	loaded, err := Load("test", &buf)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if loaded.Version != wmp.Version {
		t.Errorf("version mismatch: %d vs %d", loaded.Version, wmp.Version)
	}
	if loaded.BoardDim != wmp.BoardDim {
		t.Errorf("board_dim mismatch: %d vs %d", loaded.BoardDim, wmp.BoardDim)
	}
	if loaded.MaxWordLookupBytes != wmp.MaxWordLookupBytes {
		t.Errorf("max_word_lookup_bytes mismatch: %d vs %d",
			loaded.MaxWordLookupBytes, wmp.MaxWordLookupBytes)
	}

	// Sanity-check a lookup on the round-tripped WMP.
	br := BitRackFromMachineWord(stringsToMachineWords([]string{"CARE"})[0])
	out := make([]byte, ResultBufferSize)
	n := loaded.WriteWordsToBuffer(&br, 4, out)
	got := extractWordsFromBuffer(out[:n], 4)
	want := []string{"ACRE", "CARE", "RACE"}
	if !sliceEqual(got, want) {
		t.Errorf("after roundtrip CARE lookup: got %v, want %v", got, want)
	}
}

func TestCheckCompatibleEnglish(t *testing.T) {
	ld := testEnglishLD(t)
	if err := CheckCompatible(ld, 15); err != nil {
		t.Errorf("English/15 should be compatible, got %v", err)
	}
	if err := CheckCompatible(ld, 21); err != nil {
		t.Errorf("English/21 should be compatible, got %v", err)
	}
}

func TestCheckCompatibleTooManyBlanks(t *testing.T) {
	// English with 3 blanks instead of 2.
	csv := strings.Replace(englishLetterDistributionCSV, "?,2,0,0", "?,3,0,0", 1)
	ld, err := tilemapping.ScanLetterDistribution(strings.NewReader(csv))
	if err != nil {
		t.Fatalf("ScanLetterDistribution: %v", err)
	}
	if err := CheckCompatible(ld, 15); err == nil {
		t.Errorf("expected error for 3 blanks, got nil")
	}
}

func TestCheckCompatibleLetterCountOverflow(t *testing.T) {
	// 17 of E + 2 blanks = 19. The cap is min(maxCount, boardDim).
	// At boardDim 15 the cap pulls it to 15 (fits in 4 bits → passes).
	// At boardDim 21 the cap leaves it at 19 (> 15 → fails).
	csv := strings.Replace(englishLetterDistributionCSV, "E,12,1,1", "E,17,1,1", 1)
	ld, err := tilemapping.ScanLetterDistribution(strings.NewReader(csv))
	if err != nil {
		t.Fatalf("ScanLetterDistribution: %v", err)
	}
	if err := CheckCompatible(ld, 15); err != nil {
		t.Errorf("at boardDim 15 the cap should make this compatible, got %v", err)
	}
	if err := CheckCompatible(ld, 21); err == nil {
		t.Errorf("expected error for 17 E + 2 blanks at boardDim 21, got nil")
	}
}

func TestMakeFromWordsRejectsIncompatibleLD(t *testing.T) {
	csv := strings.Replace(englishLetterDistributionCSV, "?,2,0,0", "?,3,0,0", 1)
	ld, err := tilemapping.ScanLetterDistribution(strings.NewReader(csv))
	if err != nil {
		t.Fatalf("ScanLetterDistribution: %v", err)
	}
	words := stringsToMachineWords([]string{"CAT"})
	if _, err := MakeFromWords(words, ld, testBoardDim, 1); err == nil {
		t.Errorf("expected MakeFromWords to reject 3-blank distribution")
	}
}

// TestLoadMagpieWMP verifies binary format compatibility with MAGPIE-produced
// .wmp files. It only runs if a CSW24.wmp file is present.
func TestLoadMagpieWMP(t *testing.T) {
	path := wmpTestFilePath(t)
	wmp, err := LoadFromFile("CSW24", path)
	if err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}
	if wmp.Version != Version {
		t.Errorf("unexpected version: %d", wmp.Version)
	}
	if wmp.BoardDim != 15 {
		t.Errorf("unexpected board dim: %d", wmp.BoardDim)
	}
	if wmp.MaxWordLookupBytes == 0 {
		t.Errorf("max word lookup bytes is zero")
	}

	// Look up RETINAS — a famous 7-letter rack with multiple anagrams.
	br := BitRackFromMachineWord(stringsToMachineWords([]string{"RETINAS"})[0])
	buf := make([]byte, ResultBufferSize)
	n := wmp.WriteWordsToBuffer(&br, 7, buf)
	if n == 0 {
		t.Fatalf("RETINAS lookup returned no words")
	}
	got := extractWordsFromBuffer(buf[:n], 7)
	t.Logf("CSW24 RETINAS rack -> %d words: %v", len(got), got)
	// CSW24 should have at least RETINAS, RETAINS, NASTIER, RATINES, ANESTRI, STAINER, STEARIN, ANTSIER
	hasRetinas := false
	for _, w := range got {
		if w == "RETINAS" {
			hasRetinas = true
			break
		}
	}
	if !hasRetinas {
		t.Errorf("expected RETINAS in lookup results, got %v", got)
	}
}

// TestMagpieBinaryRoundtrip proves the on-disk format is byte-for-byte
// identical to MAGPIE's: it loads a real MAGPIE wmp file, writes it back
// out with our code, and compares the bytes.
func TestMagpieBinaryRoundtrip(t *testing.T) {
	path := wmpTestFilePath(t)
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading wmp file: %v", err)
	}
	wmp, err := Load("CSW24", bytes.NewReader(original))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	var out bytes.Buffer
	if err := wmp.Write(&out); err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	if out.Len() != len(original) {
		t.Fatalf("size mismatch: wrote %d bytes, original is %d bytes",
			out.Len(), len(original))
	}
	if !bytes.Equal(out.Bytes(), original) {
		// Find the first differing byte for a useful error message.
		for i := 0; i < len(original); i++ {
			if out.Bytes()[i] != original[i] {
				t.Fatalf("bytes differ at offset %d: got 0x%02x, want 0x%02x",
					i, out.Bytes()[i], original[i])
			}
		}
	}
}

// scanKWGFromNodes builds an in-memory KWG by serialising the given
// uint32 nodes to little-endian bytes and re-reading them through the
// only public KWG constructor.
func scanKWGFromNodes(t *testing.T, nodes []uint32) *kwg.KWG {
	t.Helper()
	buf := &bytes.Buffer{}
	for _, n := range nodes {
		if err := binary.Write(buf, binary.LittleEndian, n); err != nil {
			t.Fatalf("binary.Write: %v", err)
		}
	}
	k, err := kwg.ScanKWG(bytes.NewReader(buf.Bytes()), buf.Len())
	if err != nil {
		t.Fatalf("ScanKWG: %v", err)
	}
	return k
}

// TestExtractWordsFromKWGRejectsCorruptInput exercises the safety guards
// added after a user-reported panic: a CSW24.kwg file ended up on disk
// in a truncated state (only 227 nodes) and the recursive extractor
// dereferenced a 22-bit arc index pointing far beyond the loaded nodes,
// producing a confusing "index out of range" panic instead of a useful
// error. Now we should get a clean error in each of these scenarios.
func TestExtractWordsFromKWGRejectsCorruptInput(t *testing.T) {
	// Case 1: KWG with fewer than 2 nodes — can't even hold the
	// DAWG/GADDAG root pointer pair.
	t.Run("too small", func(t *testing.T) {
		k := scanKWGFromNodes(t, []uint32{0})
		if _, err := ExtractWordsFromKWG(k, 15); err == nil {
			t.Errorf("expected error for KWG with 1 node, got nil")
		}
	})

	// Case 2: DAWG root arc points past the end of the node array.
	// Node 0's lower 22 bits are the DAWG root arc; here we set them
	// to a value way past our 4-node KWG.
	t.Run("dawg root arc out of bounds", func(t *testing.T) {
		nodes := []uint32{
			3108706, // node 0: DAWG root arc -> beyond end
			0,       // node 1: GADDAG root (unused)
			0,       // node 2
			0,       // node 3
		}
		k := scanKWGFromNodes(t, nodes)
		_, err := ExtractWordsFromKWG(k, 15)
		if err == nil {
			t.Fatalf("expected error for out-of-bounds DAWG root arc, got nil")
		}
		if !strings.Contains(err.Error(), "out of bounds") {
			t.Errorf("expected 'out of bounds' in error, got %q", err.Error())
		}
	})

	// Case 3: A valid root, but a child arc index is out of bounds.
	// Build a single arc list at index 2 with one node that points its
	// arc at an out-of-range index.
	t.Run("child arc out of bounds", func(t *testing.T) {
		// node layout (little-endian uint32, fields packed by the KWG
		// bit-layout: bits 0-21 arc, bit 22 isEnd, bit 23 accepts,
		// bits 24-31 tile):
		//   node 0: arc=2 (DAWG root)
		//   node 1: arc=0 (GADDAG root, unused)
		//   node 2: tile=1, arc=99 (out of bounds), isEnd
		const tileShift = 24
		const isEndBit = 1 << 22
		nodes := []uint32{
			2,                              // node 0: DAWG root
			0,                              // node 1: GADDAG root
			(1 << tileShift) | isEndBit | 99, // node 2: tile=1, arc=99 OOB, isEnd
			0,
			0,
		}
		k := scanKWGFromNodes(t, nodes)
		_, err := ExtractWordsFromKWG(k, 15)
		if err == nil {
			t.Fatalf("expected error for out-of-bounds child arc, got nil")
		}
		if !strings.Contains(err.Error(), "out of bounds") {
			t.Errorf("expected 'out of bounds' in error, got %q", err.Error())
		}
	})

	// Case 4: an arc list that never sets the IsEnd bit. The recursive
	// extractor would walk past the last node forever; with the guard
	// it should report corruption.
	t.Run("missing IsEnd bit", func(t *testing.T) {
		const tileShift = 24
		// Two-node arc list at index 2 with neither node setting IsEnd.
		nodes := []uint32{
			2, // node 0: DAWG root
			0, // node 1: GADDAG root
			(1 << tileShift), // node 2: tile=1, no IsEnd
			(2 << tileShift), // node 3: tile=2, no IsEnd
		}
		k := scanKWGFromNodes(t, nodes)
		_, err := ExtractWordsFromKWG(k, 15)
		if err == nil {
			t.Fatalf("expected error for arc list without IsEnd, got nil")
		}
	})

	// Case 5: previously, the build path called MakeFromKWG which would
	// panic. Now MakeFromKWG should return the wrapped error.
	t.Run("MakeFromKWG propagates extraction error", func(t *testing.T) {
		nodes := []uint32{
			3108706, // node 0: DAWG root arc -> beyond end
			0,
		}
		k := scanKWGFromNodes(t, nodes)
		ld := testEnglishLD(t)
		_, err := MakeFromKWG(k, ld, 15, 1)
		if err == nil {
			t.Fatalf("expected error from MakeFromKWG, got nil")
		}
	})
}

func sliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
