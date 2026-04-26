// Package wmp implements WordMap, a fast anagram-lookup data structure
// for crossword move generation. Given a multiset of tiles (encoded as a
// BitRack), it can quickly retrieve all valid words that can be formed.
//
// This is a Go port of MAGPIE's WMP (https://github.com/jbradberry/MAGPIE).
// The on-disk binary format is identical to MAGPIE's, so .wmp files can be
// shared between the two implementations.
//
// The data is segregated by word length and by blank count (0, 1, or 2
// blanks). Each segment uses a hash table keyed by BitRack with bucket
// chaining for collision resolution.
package wmp

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"unsafe"
)

// Constants matching MAGPIE's wmp_defs.h.
const (
	// Version is the current WMP binary format version.
	Version = 3
	// EarliestSupportedVersion is the minimum WMP file version we accept.
	EarliestSupportedVersion = 3
	// InlineValueBytes is the number of bytes in the inline-or-metadata
	// portion of a WMPEntry. Small word groups fit entirely in here.
	InlineValueBytes = 16
	// BitRackBytes is the number of bytes used to store a BitRack key.
	BitRackBytes = 16
	// EntrySize is the on-disk size of a single WMPEntry, in bytes.
	EntrySize = InlineValueBytes + BitRackBytes
	// ResultBufferSize is the maximum number of bytes a single word lookup
	// can produce. Used to size the lookup result buffer.
	ResultBufferSize = 10000
	// minBuckets is the smallest number of hash buckets in any segment.
	minBuckets = 16
	// blankMachineLetter is the MachineLetter value for an undesignated
	// blank tile (matches macondo's tilemapping convention).
	blankMachineLetter = 0
)

// Entry is a single 32-byte WMPEntry. Bytes 0-15 hold either inlined word
// data or non-inlined metadata; bytes 16-31 hold the full 128-bit BitRack
// key (little-endian) used for collision detection.
//
// Layout of bytes 0-15 when non-inlined (byte 0 == 0):
//
//	byte 0:    nonzero_if_inlined flag (zero here)
//	bytes 1-7: padding
//	bytes 8-11: uint32 LE - word_start (blankless) or
//	            blank_letters (single blank) or
//	            first_blank_letters (double blank)
//	bytes 12-15: uint32 LE - num_words (blankless) or padding
//
// When inlined (byte 0 != 0), bytes 0-15 contain raw word data: each word
// is word_length bytes, padded with trailing zero bytes.
type Entry [EntrySize]byte

// IsInlined reports whether this entry stores its words inline.
func (e *Entry) IsInlined() bool {
	return e[0] != 0
}

// WordStart returns the offset into the WordLetters slice where this
// (non-inlined) entry's words begin.
func (e *Entry) WordStart() uint32 {
	return binary.LittleEndian.Uint32(e[8:12])
}

// NumWords returns the number of words in this (non-inlined) entry.
func (e *Entry) NumWords() uint32 {
	return binary.LittleEndian.Uint32(e[12:16])
}

// BlankLetters returns the bitmask of letters that, when substituted for
// the blank, complete a valid word. Used for single-blank entries.
func (e *Entry) BlankLetters() uint32 {
	return binary.LittleEndian.Uint32(e[8:12])
}

// FirstBlankLetters returns the bitmask of valid first-blank substitutions
// in a double-blank entry.
func (e *Entry) FirstBlankLetters() uint32 {
	return binary.LittleEndian.Uint32(e[8:12])
}

// SetWordStartAndNum writes the word_start and num_words fields.
func (e *Entry) SetWordStartAndNum(start, num uint32) {
	binary.LittleEndian.PutUint32(e[8:12], start)
	binary.LittleEndian.PutUint32(e[12:16], num)
}

// SetBlankLetters writes the blank_letters bitmask.
func (e *Entry) SetBlankLetters(mask uint32) {
	binary.LittleEndian.PutUint32(e[8:12], mask)
	binary.LittleEndian.PutUint32(e[12:16], 0)
}

// SetFirstBlankLetters writes the first_blank_letters bitmask.
func (e *Entry) SetFirstBlankLetters(mask uint32) {
	binary.LittleEndian.PutUint32(e[8:12], mask)
	binary.LittleEndian.PutUint32(e[12:16], 0)
}

// InlineData returns a slice over bytes 0-15 (the inline-or-metadata area).
func (e *Entry) InlineData() []byte {
	return e[:InlineValueBytes]
}

// ReadBitRack extracts the 128-bit BitRack key from this entry.
//
// We bypass binary.LittleEndian.Uint64's bounds-checked slice
// expression and read the two 64-bit halves directly via
// unsafe.Pointer. This is safe because Entry is a fixed-size
// [32]byte and the offsets (16, 24) are compile-time constants
// inside its bounds. On a little-endian host (every architecture
// macondo supports) the in-memory layout already matches the
// on-disk little-endian format, so no byte swap is needed.
func (e *Entry) ReadBitRack() BitRack {
	p := (*[2]uint64)(unsafe.Pointer(&e[InlineValueBytes]))
	return BitRack{Low: p[0], High: p[1]}
}

// WriteBitRack writes the BitRack key into this entry.
func (e *Entry) WriteBitRack(br *BitRack) {
	p := (*[2]uint64)(unsafe.Pointer(&e[InlineValueBytes]))
	p[0] = br.Low
	p[1] = br.High
}

// MaxInlinedWords returns the maximum number of words of the given length
// that can be stored inline in a single entry.
func MaxInlinedWords(wordLength int) int {
	return InlineValueBytes / wordLength
}

// numberOfInlinedBytes returns how many bytes of inline word data are
// actually present in this inlined entry, accounting for trailing zero
// padding when the entry isn't full.
func (e *Entry) numberOfInlinedBytes(wordLength int) int {
	numBytes := MaxInlinedWords(wordLength) * wordLength
	for numBytes > wordLength {
		if e[numBytes-1] != 0 {
			break
		}
		numBytes -= wordLength
	}
	return numBytes
}

// ForLength holds the WMP data for a single word length: blankless words,
// single-blank words, and double-blank words. Each is a separate hash table.
type ForLength struct {
	// Blankless words
	NumWordBuckets    uint32
	NumWordEntries    uint32
	NumUninlinedWords uint32
	WordBucketStarts  []uint32
	WordMapEntries    []Entry
	WordLetters       []byte

	// Single-blank words
	NumBlankBuckets   uint32
	NumBlankEntries   uint32
	BlankBucketStarts []uint32
	BlankMapEntries   []Entry

	// Double-blank words
	NumDoubleBlankBuckets   uint32
	NumDoubleBlankEntries   uint32
	DoubleBlankBucketStarts []uint32
	DoubleBlankMapEntries   []Entry
}

// WMP is a complete WordMap dictionary, containing one ForLength per
// supported word length.
type WMP struct {
	Name               string
	Version            uint8
	BoardDim           uint8
	MaxWordLookupBytes uint32
	// WFLs is indexed by word length. Indices 0 and 1 are unused; valid
	// entries occupy indices 2..BoardDim inclusive.
	WFLs []ForLength
}

// Lookup-related errors.
var (
	ErrUnsupportedVersion = fmt.Errorf("wmp: unsupported version")
	ErrIncompatibleDim    = fmt.Errorf("wmp: incompatible board dimension")
)

// LoadFromFile reads a WMP from the given filename. The lexicon name is
// stored on the returned WMP for identification.
func LoadFromFile(name, filename string) (*WMP, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Load(name, f)
}

// Load reads a WMP from a reader. The lexicon name is stored on the
// returned WMP for identification.
func Load(name string, r io.Reader) (*WMP, error) {
	wmp := &WMP{Name: name}

	// Header: 1 byte version, 1 byte board_dim, 4 bytes max_word_lookup_bytes
	var hdr [6]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, fmt.Errorf("wmp: reading header: %w", err)
	}
	wmp.Version = hdr[0]
	wmp.BoardDim = hdr[1]
	wmp.MaxWordLookupBytes = binary.LittleEndian.Uint32(hdr[2:6])

	if wmp.Version < EarliestSupportedVersion {
		return nil, fmt.Errorf("%w: detected version %d, need %d or greater",
			ErrUnsupportedVersion, wmp.Version, EarliestSupportedVersion)
	}

	wmp.WFLs = make([]ForLength, int(wmp.BoardDim)+1)
	for length := 2; length <= int(wmp.BoardDim); length++ {
		if err := readForLength(&wmp.WFLs[length], length, r); err != nil {
			return nil, fmt.Errorf("wmp: reading length %d: %w", length, err)
		}
	}
	return wmp, nil
}

// readUint32 reads a single little-endian uint32.
func readUint32(r io.Reader) (uint32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

// readUint32s reads n little-endian uint32s.
func readUint32s(r io.Reader, n uint32) ([]uint32, error) {
	if n == 0 {
		return nil, nil
	}
	buf := make([]byte, 4*n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	out := make([]uint32, n)
	for i := uint32(0); i < n; i++ {
		out[i] = binary.LittleEndian.Uint32(buf[4*i : 4*i+4])
	}
	return out, nil
}

// readEntries reads n WMPEntries (each EntrySize bytes) from the reader.
func readEntries(r io.Reader, n uint32) ([]Entry, error) {
	if n == 0 {
		return nil, nil
	}
	buf := make([]byte, EntrySize*int(n))
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	entries := make([]Entry, n)
	for i := uint32(0); i < n; i++ {
		copy(entries[i][:], buf[int(i)*EntrySize:(int(i)+1)*EntrySize])
	}
	return entries, nil
}

// validateBucketCount ensures the bucket count is a non-zero power of
// two. GetBucketIndex relies on `& (numBuckets - 1)` for hashing, which
// is only correct for power-of-two sizes; the unsafe-pointer scanBucket
// loop also depends on the masked index landing within bucketStarts.
func validateBucketCount(section string, n uint32) error {
	if n == 0 || (n&(n-1)) != 0 {
		return fmt.Errorf("wmp: %s bucket count %d is not a non-zero power of two", section, n)
	}
	return nil
}

// validateBucketStarts ensures the bucket-start offsets describe a
// monotonic, non-overflowing partition of the entries slice. Without
// these guarantees the no-bounds-check inner loop in scanBucket would
// happily walk past the end of the entries array on a corrupted file.
func validateBucketStarts(section string, starts []uint32, numEntries uint32) error {
	if len(starts) < 2 {
		return fmt.Errorf("wmp: %s bucket-starts has length %d (need >= 2)", section, len(starts))
	}
	if starts[0] != 0 {
		return fmt.Errorf("wmp: %s bucket-starts[0] is %d (must be 0)", section, starts[0])
	}
	for i := 1; i < len(starts); i++ {
		if starts[i] < starts[i-1] {
			return fmt.Errorf("wmp: %s bucket-starts not monotonic at index %d (%d < %d)", section, i, starts[i], starts[i-1])
		}
	}
	if starts[len(starts)-1] != numEntries {
		return fmt.Errorf("wmp: %s bucket-starts terminator is %d (must equal numEntries=%d)",
			section, starts[len(starts)-1], numEntries)
	}
	return nil
}

func readForLength(wfl *ForLength, length int, r io.Reader) error {
	// Blankless words section
	var err error
	wfl.NumWordBuckets, err = readUint32(r)
	if err != nil {
		return err
	}
	if err := validateBucketCount("word", wfl.NumWordBuckets); err != nil {
		return err
	}
	wfl.WordBucketStarts, err = readUint32s(r, wfl.NumWordBuckets+1)
	if err != nil {
		return err
	}
	wfl.NumWordEntries, err = readUint32(r)
	if err != nil {
		return err
	}
	if err := validateBucketStarts("word", wfl.WordBucketStarts, wfl.NumWordEntries); err != nil {
		return err
	}
	wfl.WordMapEntries, err = readEntries(r, wfl.NumWordEntries)
	if err != nil {
		return err
	}
	wfl.NumUninlinedWords, err = readUint32(r)
	if err != nil {
		return err
	}
	if wfl.NumUninlinedWords > 0 {
		wfl.WordLetters = make([]byte, int(wfl.NumUninlinedWords)*length)
		if _, err = io.ReadFull(r, wfl.WordLetters); err != nil {
			return err
		}
	}

	// Single-blank section
	wfl.NumBlankBuckets, err = readUint32(r)
	if err != nil {
		return err
	}
	if err := validateBucketCount("blank", wfl.NumBlankBuckets); err != nil {
		return err
	}
	wfl.BlankBucketStarts, err = readUint32s(r, wfl.NumBlankBuckets+1)
	if err != nil {
		return err
	}
	wfl.NumBlankEntries, err = readUint32(r)
	if err != nil {
		return err
	}
	if err := validateBucketStarts("blank", wfl.BlankBucketStarts, wfl.NumBlankEntries); err != nil {
		return err
	}
	wfl.BlankMapEntries, err = readEntries(r, wfl.NumBlankEntries)
	if err != nil {
		return err
	}

	// Double-blank section
	wfl.NumDoubleBlankBuckets, err = readUint32(r)
	if err != nil {
		return err
	}
	if err := validateBucketCount("double-blank", wfl.NumDoubleBlankBuckets); err != nil {
		return err
	}
	wfl.DoubleBlankBucketStarts, err = readUint32s(r, wfl.NumDoubleBlankBuckets+1)
	if err != nil {
		return err
	}
	wfl.NumDoubleBlankEntries, err = readUint32(r)
	if err != nil {
		return err
	}
	if err := validateBucketStarts("double-blank", wfl.DoubleBlankBucketStarts, wfl.NumDoubleBlankEntries); err != nil {
		return err
	}
	wfl.DoubleBlankMapEntries, err = readEntries(r, wfl.NumDoubleBlankEntries)
	if err != nil {
		return err
	}
	return nil
}

// WriteToFile writes the WMP to the given filename in MAGPIE-compatible
// binary format.
func (wmp *WMP) WriteToFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return wmp.Write(f)
}

// Write serializes the WMP to a writer.
func (wmp *WMP) Write(w io.Writer) error {
	// Header
	hdr := []byte{wmp.Version, wmp.BoardDim, 0, 0, 0, 0}
	binary.LittleEndian.PutUint32(hdr[2:6], wmp.MaxWordLookupBytes)
	if _, err := w.Write(hdr); err != nil {
		return err
	}
	for length := 2; length <= int(wmp.BoardDim); length++ {
		if err := writeForLength(&wmp.WFLs[length], length, w); err != nil {
			return err
		}
	}
	return nil
}

func writeUint32(w io.Writer, v uint32) error {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	_, err := w.Write(b[:])
	return err
}

func writeUint32s(w io.Writer, vs []uint32) error {
	if len(vs) == 0 {
		return nil
	}
	buf := make([]byte, 4*len(vs))
	for i, v := range vs {
		binary.LittleEndian.PutUint32(buf[4*i:4*i+4], v)
	}
	_, err := w.Write(buf)
	return err
}

func writeEntries(w io.Writer, entries []Entry) error {
	if len(entries) == 0 {
		return nil
	}
	buf := make([]byte, EntrySize*len(entries))
	for i := range entries {
		copy(buf[i*EntrySize:(i+1)*EntrySize], entries[i][:])
	}
	_, err := w.Write(buf)
	return err
}

func writeForLength(wfl *ForLength, length int, w io.Writer) error {
	// Blankless
	if err := writeUint32(w, wfl.NumWordBuckets); err != nil {
		return err
	}
	if err := writeUint32s(w, wfl.WordBucketStarts); err != nil {
		return err
	}
	if err := writeUint32(w, wfl.NumWordEntries); err != nil {
		return err
	}
	if err := writeEntries(w, wfl.WordMapEntries); err != nil {
		return err
	}
	if err := writeUint32(w, wfl.NumUninlinedWords); err != nil {
		return err
	}
	if int(wfl.NumUninlinedWords) > 0 {
		if _, err := w.Write(wfl.WordLetters[:int(wfl.NumUninlinedWords)*length]); err != nil {
			return err
		}
	}

	// Single-blank
	if err := writeUint32(w, wfl.NumBlankBuckets); err != nil {
		return err
	}
	if err := writeUint32s(w, wfl.BlankBucketStarts); err != nil {
		return err
	}
	if err := writeUint32(w, wfl.NumBlankEntries); err != nil {
		return err
	}
	if err := writeEntries(w, wfl.BlankMapEntries); err != nil {
		return err
	}

	// Double-blank
	if err := writeUint32(w, wfl.NumDoubleBlankBuckets); err != nil {
		return err
	}
	if err := writeUint32s(w, wfl.DoubleBlankBucketStarts); err != nil {
		return err
	}
	if err := writeUint32(w, wfl.NumDoubleBlankEntries); err != nil {
		return err
	}
	if err := writeEntries(w, wfl.DoubleBlankMapEntries); err != nil {
		return err
	}
	return nil
}

// ============================================================================
// Lookup operations
// ============================================================================

// scanBucket performs the bucket-scan loop common to the blankless,
// single-blank, and double-blank lookups. Given the bucket starts
// array, entries array, and bucket index, it returns a pointer to
// the matching entry or nil.
//
// The hot path (the for loop body) is hand-tuned to:
//
//   - Take the underlying entries pointer once outside the loop so
//     each iteration is a constant-stride pointer add (no slice
//     header reload, no per-iteration bounds check).
//   - Read each entry's BitRack via unsafe.Pointer cast — two
//     uint64 loads instead of two binary.LittleEndian.Uint64 calls
//     with their attendant slice expression bounds checks.
//
// Performance-critical: this is the WMP's #1 inner loop. Each
// shadow_record gating check, each wordmapGen subrack lookup, and
// each CheckNonplaythroughExistence enumeration step lands here.
//
//go:nosplit
func scanBucket(bucketStarts []uint32, entries []Entry, bucketIdx uint32, brLow, brHigh uint64) *Entry {
	// Bounds check elimination hint: a single check on the upper
	// bucket-start access tells the compiler the lower one is
	// definitely in range too.
	end := bucketStarts[bucketIdx+1]
	start := bucketStarts[bucketIdx]
	if start == end {
		return nil
	}
	// Hoist the entries base pointer out of the loop. We index by
	// uintptr stride below to keep the loop tight.
	base := unsafe.Pointer(&entries[start])
	const entrySize = unsafe.Sizeof(Entry{})
	for i := start; i < end; i++ {
		// Each Entry is 32 bytes; the BitRack key occupies the
		// trailing 16 bytes (offset InlineValueBytes). Read it as
		// two uint64s without bounds checks.
		br := (*[2]uint64)(unsafe.Add(base, uintptr(InlineValueBytes)))
		if br[0] == brLow && br[1] == brHigh {
			return (*Entry)(base)
		}
		base = unsafe.Add(base, entrySize)
	}
	return nil
}

// getWordEntry looks up the entry for a given BitRack in the
// blankless table. Returns nil if not found.
func (wfl *ForLength) getWordEntry(br *BitRack) *Entry {
	if wfl.NumWordBuckets == 0 {
		return nil
	}
	bucketIdx := br.GetBucketIndex(wfl.NumWordBuckets)
	return scanBucket(wfl.WordBucketStarts, wfl.WordMapEntries, bucketIdx, br.Low, br.High)
}

// getBlankEntry looks up the entry for a single-blank rack.
func (wfl *ForLength) getBlankEntry(br *BitRack) *Entry {
	if wfl.NumBlankBuckets == 0 {
		return nil
	}
	bucketIdx := br.GetBucketIndex(wfl.NumBlankBuckets)
	return scanBucket(wfl.BlankBucketStarts, wfl.BlankMapEntries, bucketIdx, br.Low, br.High)
}

// getDoubleBlankEntry looks up the entry for a double-blank rack.
func (wfl *ForLength) getDoubleBlankEntry(br *BitRack) *Entry {
	if wfl.NumDoubleBlankBuckets == 0 {
		return nil
	}
	bucketIdx := br.GetBucketIndex(wfl.NumDoubleBlankBuckets)
	return scanBucket(wfl.DoubleBlankBucketStarts, wfl.DoubleBlankMapEntries, bucketIdx, br.Low, br.High)
}

// GetWordEntry looks up the entry for a given BitRack and word length.
// Returns nil if not found. The lookup automatically dispatches to the
// blankless, single-blank, or double-blank table based on the blank count.
func (wmp *WMP) GetWordEntry(br *BitRack, wordLength int) *Entry {
	if wordLength < 2 || wordLength > int(wmp.BoardDim) {
		return nil
	}
	wfl := &wmp.WFLs[wordLength]
	switch br.GetLetter(blankMachineLetter) {
	case 0:
		return wfl.getWordEntry(br)
	case 1:
		return wfl.getBlankEntry(br)
	case 2:
		return wfl.getDoubleBlankEntry(br)
	}
	return nil
}

// writeBlanklessInlinedWords copies inlined word data to the buffer.
// Returns bytes written.
func writeBlanklessInlinedWords(entry *Entry, wordLength int, buffer []byte) int {
	n := entry.numberOfInlinedBytes(wordLength)
	copy(buffer, entry[:n])
	return n
}

// writeBlanklessUninlinedWords copies uninlined word data to the buffer.
// Returns bytes written.
func writeBlanklessUninlinedWords(entry *Entry, wfl *ForLength, wordLength int, buffer []byte) int {
	letters := wfl.WordLetters[entry.WordStart():]
	n := int(entry.NumWords()) * wordLength
	copy(buffer, letters[:n])
	return n
}

// writeBlanklessWords copies all words from a blankless entry to the buffer.
func writeBlanklessWords(entry *Entry, wfl *ForLength, wordLength int, buffer []byte) int {
	if entry.IsInlined() {
		return writeBlanklessInlinedWords(entry, wordLength, buffer)
	}
	return writeBlanklessUninlinedWords(entry, wfl, wordLength, buffer)
}

// writeBlankWords expands a single-blank entry by trying each valid letter
// substitution and looking up the resulting blankless rack.
func writeBlankWords(entry *Entry, wfl *ForLength, br *BitRack, wordLength int, minML byte, buffer []byte) int {
	bytesWritten := 0
	br.SetLetterCount(blankMachineLetter, 0)
	blankLetters := entry.BlankLetters()
	for ml := minML; ml < maxAlphabetSize; ml++ {
		if blankLetters&(1<<ml) == 0 {
			continue
		}
		br.AddLetter(ml)
		blanklessEntry := wfl.getWordEntry(br)
		if blanklessEntry != nil {
			bytesWritten += writeBlanklessWords(blanklessEntry, wfl, wordLength, buffer[bytesWritten:])
		}
		br.TakeLetter(ml)
	}
	br.SetLetterCount(blankMachineLetter, 1)
	return bytesWritten
}

// writeDoubleBlankWords expands a double-blank entry by trying each valid
// first-blank substitution, then recursing into single-blank expansion.
func writeDoubleBlankWords(entry *Entry, wfl *ForLength, br *BitRack, wordLength int, buffer []byte) int {
	bytesWritten := 0
	br.SetLetterCount(blankMachineLetter, 1)
	firstBlanks := entry.FirstBlankLetters()
	for ml := byte(1); ml < maxAlphabetSize; ml++ {
		if firstBlanks&(1<<ml) == 0 {
			continue
		}
		br.AddLetter(ml)
		blankEntry := wfl.getBlankEntry(br)
		if blankEntry != nil {
			bytesWritten += writeBlankWords(blankEntry, wfl, br, wordLength, ml, buffer[bytesWritten:])
		}
		br.TakeLetter(ml)
	}
	br.SetLetterCount(blankMachineLetter, 2)
	return bytesWritten
}

// WriteWordsToBuffer looks up the BitRack and writes all matching words
// (concatenated as raw MachineLetters, no separator) into the buffer.
// Returns the number of bytes written, or 0 if no words found.
//
// The buffer must be at least MaxWordLookupBytes large to be safe.
//
// Note: this may temporarily mutate the BitRack during blank expansion,
// but restores it to its original value before returning.
func (wmp *WMP) WriteWordsToBuffer(br *BitRack, wordLength int, buffer []byte) int {
	if wordLength < 2 || wordLength > int(wmp.BoardDim) {
		return 0
	}
	wfl := &wmp.WFLs[wordLength]
	entry := wmp.GetWordEntry(br, wordLength)
	if entry == nil {
		return 0
	}
	switch br.GetLetter(blankMachineLetter) {
	case 0:
		return writeBlanklessWords(entry, wfl, wordLength, buffer)
	case 1:
		return writeBlankWords(entry, wfl, br, wordLength, 1, buffer)
	case 2:
		return writeDoubleBlankWords(entry, wfl, br, wordLength, buffer)
	}
	return 0
}

// HasWord returns true if the BitRack maps to at least one word of the
// given length.
func (wmp *WMP) HasWord(br *BitRack, wordLength int) bool {
	return wmp.GetWordEntry(br, wordLength) != nil
}

// WriteWordsForEntry writes the words for a pre-found entry into buffer.
// Use this when you already obtained the entry via GetWordEntry to avoid
// a second hash lookup. The blank count in br must match the entry's
// table (0/1/2 blanks → blankless/single/double-blank entry).
//
// Mirrors MAGPIE's wmp_entry_write_words_to_buffer.
func (wmp *WMP) WriteWordsForEntry(entry *Entry, br *BitRack, wordLength int, buffer []byte) int {
	if wordLength < 2 || wordLength > int(wmp.BoardDim) {
		return 0
	}
	wfl := &wmp.WFLs[wordLength]
	switch br.GetLetter(blankMachineLetter) {
	case 0:
		return writeBlanklessWords(entry, wfl, wordLength, buffer)
	case 1:
		return writeBlankWords(entry, wfl, br, wordLength, 1, buffer)
	case 2:
		return writeDoubleBlankWords(entry, wfl, br, wordLength, buffer)
	}
	return 0
}
