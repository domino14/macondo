package wmp

import (
	"fmt"
	"math/bits"
	"runtime"
	"sync"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

// MakeFromWords builds a WMP from a list of words for the given letter
// distribution. boardDim is the maximum word length supported (typically
// 15 for standard Scrabble, 21 for SuperCrosswordGame). numThreads
// controls parallelism (0 = use all available cores).
//
// The (ld, boardDim) pair is validated via CheckCompatible: the alphabet
// must fit in 32 letters, the distribution must have ≤ 2 blanks, and
// no letter (with blanks substituted, capped by boardDim) may exceed
// the BitRack's 4-bit-per-letter limit.
func MakeFromWords(words []tilemapping.MachineWord, ld *tilemapping.LetterDistribution, boardDim, numThreads int) (*WMP, error) {
	if boardDim < 2 {
		return nil, fmt.Errorf("wmp: board dim %d too small", boardDim)
	}
	if err := CheckCompatible(ld, boardDim); err != nil {
		return nil, err
	}
	alphabetSize := len(ld.Distribution())

	if numThreads <= 0 {
		numThreads = runtime.NumCPU()
	}
	// Cap at boardDim - 1 (the number of distinct word lengths 2..boardDim)
	if numThreads > boardDim-1 {
		numThreads = boardDim - 1
	}

	// Group words by length
	wordsByLength := make([][]tilemapping.MachineWord, boardDim+1)
	for _, w := range words {
		l := len(w)
		if l < 2 || l > boardDim {
			continue
		}
		wordsByLength[l] = append(wordsByLength[l], w)
	}

	wmp := &WMP{
		Version:  Version,
		BoardDim: uint8(boardDim),
		WFLs:     make([]ForLength, boardDim+1),
	}

	radixPasses := radixPassesForAlphabet(alphabetSize)

	// Build all three phases for all lengths in parallel, capped by numThreads.
	// Phase 1 must finish before phases 2/3 since they depend on the unique
	// racks extracted in phase 1. We process lengths in work-descending order
	// so heavy work starts first.
	type lengthWork struct {
		length int
		count  int
	}
	var work []lengthWork
	for l := 2; l <= boardDim; l++ {
		work = append(work, lengthWork{length: l, count: len(wordsByLength[l])})
	}
	// Insertion sort by count descending
	for i := 1; i < len(work); i++ {
		key := work[i]
		j := i - 1
		for j >= 0 && work[j].count < key.count {
			work[j+1] = work[j]
			j--
		}
		work[j+1] = key
	}

	// uniqueRacks[length] holds the unique BitRacks extracted in phase 1
	// of that length, used as input to phases 2 and 3.
	uniqueRacks := make([][]BitRack, boardDim+1)

	// Phase 1: build word entries and extract unique racks for each length.
	var wg sync.WaitGroup
	sem := make(chan struct{}, numThreads)
	for _, lw := range work {
		l := lw.length
		wg.Add(1)
		sem <- struct{}{}
		go func(length int) {
			defer wg.Done()
			defer func() { <-sem }()
			uniqueRacks[length] = buildWordEntries(&wmp.WFLs[length], wordsByLength[length], length, radixPasses)
		}(l)
	}
	wg.Wait()

	// Phase 2: build single-blank entries.
	for _, lw := range work {
		l := lw.length
		wg.Add(1)
		sem <- struct{}{}
		go func(length int) {
			defer wg.Done()
			defer func() { <-sem }()
			buildBlankEntries(&wmp.WFLs[length], uniqueRacks[length], length, radixPasses)
		}(l)
	}
	wg.Wait()

	// Phase 3: build double-blank entries.
	for _, lw := range work {
		l := lw.length
		wg.Add(1)
		sem <- struct{}{}
		go func(length int) {
			defer wg.Done()
			defer func() { <-sem }()
			buildDoubleBlankEntries(&wmp.WFLs[length], uniqueRacks[length], length, radixPasses)
		}(l)
	}
	wg.Wait()

	// Initialize empty length slots so the binary writer doesn't choke.
	for l := 2; l <= boardDim; l++ {
		wfl := &wmp.WFLs[l]
		if wfl.NumWordBuckets == 0 {
			initEmptyForLength(wfl)
		}
	}

	wmp.MaxWordLookupBytes = calculateMaxWordLookupBytes(wmp, boardDim)
	return wmp, nil
}

// MakeFromKWG builds a WMP from the DAWG portion of a KWG. The GADDAG
// portion is not used. The letter distribution is validated against the
// WMP's BitRack constraints; see MakeFromWords for details.
func MakeFromKWG(k *kwg.KWG, ld *tilemapping.LetterDistribution, boardDim, numThreads int) (*WMP, error) {
	words, err := ExtractWordsFromKWG(k, boardDim)
	if err != nil {
		return nil, err
	}
	return MakeFromWords(words, ld, boardDim, numThreads)
}

// ExtractWordsFromKWG walks the DAWG portion of a KWG and returns all
// stored words (each as a tilemapping.MachineWord). Words longer than
// maxLength are skipped.
//
// Returns an error if the KWG is too small to contain a root node or
// if any arc index encountered during the walk points outside the node
// array (i.e. the KWG file is truncated or corrupted). Without those
// guards a malformed input would panic with an "index out of range"
// when accessing k.Tile/k.ArcIndex.
func ExtractWordsFromKWG(k *kwg.KWG, maxLength int) ([]tilemapping.MachineWord, error) {
	var words []tilemapping.MachineWord
	nodesLen := uint32(len(k.Nodes()))
	if nodesLen < 2 {
		return nil, fmt.Errorf("wmp: KWG has %d nodes; expected at least 2 (DAWG and GADDAG roots)", nodesLen)
	}
	dawgRoot := k.ArcIndex(0)
	if dawgRoot == 0 {
		return words, nil
	}
	if dawgRoot >= nodesLen {
		return nil, fmt.Errorf("wmp: KWG appears corrupted: DAWG root arc %d is out of bounds for %d-node KWG", dawgRoot, nodesLen)
	}
	prefix := make([]tilemapping.MachineLetter, 0, maxLength)
	if err := extractWordsRecursive(k, dawgRoot, prefix, maxLength, nodesLen, &words); err != nil {
		return nil, err
	}
	return words, nil
}

func extractWordsRecursive(k *kwg.KWG, nodeIdx uint32, prefix []tilemapping.MachineLetter, maxLength int, nodesLen uint32, words *[]tilemapping.MachineWord) error {
	for i := nodeIdx; ; i++ {
		if i >= nodesLen {
			return fmt.Errorf("wmp: KWG appears corrupted: arc list starting at node %d ran past end of %d-node KWG", nodeIdx, nodesLen)
		}
		tile := k.Tile(i)
		next := append(prefix, tilemapping.MachineLetter(tile))
		if k.Accepts(i) {
			word := make(tilemapping.MachineWord, len(next))
			copy(word, next)
			*words = append(*words, word)
		}
		if len(next) < maxLength {
			arc := k.ArcIndex(i)
			if arc != 0 {
				if arc >= nodesLen {
					return fmt.Errorf("wmp: KWG appears corrupted: arc %d at node %d is out of bounds for %d-node KWG", arc, i, nodesLen)
				}
				if err := extractWordsRecursive(k, arc, next, maxLength, nodesLen, words); err != nil {
					return err
				}
			}
		}
		if k.IsEnd(i) {
			return nil
		}
	}
}

// ============================================================================
// Helpers
// ============================================================================

// radixPassesForAlphabet returns the number of LSD radix-sort passes needed
// to fully sort BitRacks for an alphabet of the given size. Each letter
// uses 4 bits, so we need ceil(alphabetSize * 4 / 8) bytes.
func radixPassesForAlphabet(alphabetSize int) int {
	return (alphabetSize*bitsPerLetter + 7) / 8
}

// nextPowerOf2 rounds n up to the next power of 2 (with 0 -> 1).
func nextPowerOf2(n uint32) uint32 {
	if n == 0 {
		return 1
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	return n + 1
}

// initEmptyForLength sets up an empty ForLength with valid (zero-content)
// hash tables. Required for the binary writer.
func initEmptyForLength(wfl *ForLength) {
	wfl.NumWordBuckets = minBuckets
	wfl.WordBucketStarts = make([]uint32, minBuckets+1)
	wfl.NumBlankBuckets = minBuckets
	wfl.BlankBucketStarts = make([]uint32, minBuckets+1)
	wfl.NumDoubleBlankBuckets = minBuckets
	wfl.DoubleBlankBucketStarts = make([]uint32, minBuckets+1)
}

// ============================================================================
// Phase 1: Build word entries and extract unique racks
// ============================================================================

type wordPair struct {
	bitRack   BitRack
	wordIndex uint32
}

func buildWordEntries(wfl *ForLength, words []tilemapping.MachineWord, wordLength, radixPasses int) []BitRack {
	count := uint32(len(words))

	// Build (BitRack, wordIndex) pairs
	pairs := make([]wordPair, count)
	for i, w := range words {
		pairs[i].bitRack = BitRackFromMachineWord(w)
		pairs[i].wordIndex = uint32(i)
	}

	// Radix sort by BitRack
	temp := make([]wordPair, count)
	radixSortWordPairs(pairs, temp, radixPasses)

	// Count unique racks
	var numUnique uint32
	if count > 0 {
		numUnique = 1
		for i := uint32(1); i < count; i++ {
			if !pairs[i].bitRack.Equals(&pairs[i-1].bitRack) {
				numUnique++
			}
		}
	}

	// Extract unique racks for later phases
	uniqueRacks := make([]BitRack, 0, numUnique)
	if count > 0 {
		uniqueRacks = append(uniqueRacks, pairs[0].bitRack)
		for i := uint32(1); i < count; i++ {
			if !pairs[i].bitRack.Equals(&pairs[i-1].bitRack) {
				uniqueRacks = append(uniqueRacks, pairs[i].bitRack)
			}
		}
	}

	// Determine bucket count
	numBuckets := nextPowerOf2(numUnique)
	if numBuckets < minBuckets {
		numBuckets = minBuckets
	}
	wfl.NumWordBuckets = numBuckets

	bucketCounts := make([]uint32, numBuckets)
	maxInline := uint32(MaxInlinedWords(wordLength))
	var numUninlinedLetters uint32

	if count > 0 {
		runStart := uint32(0)
		for i := uint32(1); i <= count; i++ {
			isEnd := i == count || !pairs[i].bitRack.Equals(&pairs[i-1].bitRack)
			if isEnd {
				wordsInRun := i - runStart
				rack := pairs[runStart].bitRack
				bucketCounts[rack.GetBucketIndex(numBuckets)]++
				if wordsInRun > maxInline {
					numUninlinedLetters += wordsInRun * uint32(wordLength)
				}
				runStart = i
			}
		}
	}

	wfl.NumWordEntries = numUnique
	wfl.WordMapEntries = make([]Entry, numUnique)
	wfl.NumUninlinedWords = numUninlinedLetters / uint32(wordLength)
	wfl.WordLetters = make([]byte, numUninlinedLetters)
	wfl.WordBucketStarts = make([]uint32, numBuckets+1)

	// Compute bucket starts (prefix sum)
	var offset uint32
	for b := uint32(0); b < numBuckets; b++ {
		wfl.WordBucketStarts[b] = offset
		offset += bucketCounts[b]
	}
	wfl.WordBucketStarts[numBuckets] = offset

	// Reset bucket counts to use as insertion cursors
	for i := range bucketCounts {
		bucketCounts[i] = 0
	}

	if count > 0 {
		var letterOffset uint32
		runStart := uint32(0)
		for i := uint32(1); i <= count; i++ {
			isEnd := i == count || !pairs[i].bitRack.Equals(&pairs[i-1].bitRack)
			if !isEnd {
				continue
			}
			wordsInRun := i - runStart
			rack := pairs[runStart].bitRack
			bucketIdx := rack.GetBucketIndex(numBuckets)
			entryIdx := wfl.WordBucketStarts[bucketIdx] + bucketCounts[bucketIdx]
			bucketCounts[bucketIdx]++

			entry := &wfl.WordMapEntries[entryIdx]
			entry.WriteBitRack(&rack)

			if wordsInRun <= maxInline {
				// Inline: copy each word's bytes into the inline area.
				// Mark inlined by ensuring byte 0 is nonzero (the first
				// letter of the first word; valid letters are >= 1).
				for j := uint32(0); j < wordsInRun; j++ {
					word := words[pairs[runStart+j].wordIndex]
					off := int(j) * wordLength
					for k, ml := range word {
						entry[off+k] = byte(ml)
					}
				}
			} else {
				// Non-inline: byte 0 must remain zero, then write the
				// uint32 metadata.
				entry.SetWordStartAndNum(letterOffset, wordsInRun)
				for j := uint32(0); j < wordsInRun; j++ {
					word := words[pairs[runStart+j].wordIndex]
					for k, ml := range word {
						wfl.WordLetters[int(letterOffset)+k] = byte(ml)
					}
					letterOffset += uint32(wordLength)
				}
			}
			runStart = i
		}
	}

	return uniqueRacks
}

// ============================================================================
// Phase 2: Build single-blank entries
// ============================================================================

type blankPair struct {
	bitRack         BitRack
	blankLetterBit  uint32
}

func buildBlankEntries(wfl *ForLength, uniqueRacks []BitRack, wordLength, radixPasses int) {
	// Generate (rack_with_blank_substituted, letter_bit) pairs.
	// For each unique rack, for each non-blank letter present, substitute
	// that letter with a blank and emit a pair.
	var pairs []blankPair
	for _, r := range uniqueRacks {
		rack := r
		present := rack.GetLetterMask() &^ 1 // exclude letter 0 (blank)
		for present != 0 {
			ml := byte(bits.TrailingZeros32(present))
			present &= present - 1
			rack.TakeLetter(ml)
			rack.AddLetter(blankMachineLetter)
			pairs = append(pairs, blankPair{
				bitRack:        rack,
				blankLetterBit: 1 << ml,
			})
			rack.TakeLetter(blankMachineLetter)
			rack.AddLetter(ml)
		}
	}
	numPairs := uint32(len(pairs))

	// Radix sort
	temp := make([]blankPair, numPairs)
	radixSortBlankPairs(pairs, temp, radixPasses)

	// Count unique racks
	var numUnique uint32
	if numPairs > 0 {
		numUnique = 1
		for i := uint32(1); i < numPairs; i++ {
			if !pairs[i].bitRack.Equals(&pairs[i-1].bitRack) {
				numUnique++
			}
		}
	}

	numBuckets := nextPowerOf2(numUnique)
	if numBuckets < minBuckets {
		numBuckets = minBuckets
	}
	wfl.NumBlankBuckets = numBuckets

	bucketCounts := make([]uint32, numBuckets)
	if numPairs > 0 {
		bucketCounts[pairs[0].bitRack.GetBucketIndex(numBuckets)]++
		for i := uint32(1); i < numPairs; i++ {
			if !pairs[i].bitRack.Equals(&pairs[i-1].bitRack) {
				bucketCounts[pairs[i].bitRack.GetBucketIndex(numBuckets)]++
			}
		}
	}

	wfl.NumBlankEntries = numUnique
	wfl.BlankMapEntries = make([]Entry, numUnique)
	wfl.BlankBucketStarts = make([]uint32, numBuckets+1)

	var offset uint32
	for b := uint32(0); b < numBuckets; b++ {
		wfl.BlankBucketStarts[b] = offset
		offset += bucketCounts[b]
	}
	wfl.BlankBucketStarts[numBuckets] = offset

	// Reset for insertion
	for i := range bucketCounts {
		bucketCounts[i] = 0
	}

	// Group consecutive pairs with same BitRack and write entries.
	if numPairs > 0 {
		currentRack := pairs[0].bitRack
		currentBits := pairs[0].blankLetterBit
		for i := uint32(1); i <= numPairs; i++ {
			isEnd := i == numPairs || !pairs[i].bitRack.Equals(&currentRack)
			if !isEnd {
				currentBits |= pairs[i].blankLetterBit
				continue
			}
			bucketIdx := currentRack.GetBucketIndex(numBuckets)
			entryIdx := wfl.BlankBucketStarts[bucketIdx] + bucketCounts[bucketIdx]
			bucketCounts[bucketIdx]++
			entry := &wfl.BlankMapEntries[entryIdx]
			entry.SetBlankLetters(currentBits)
			entry.WriteBitRack(&currentRack)
			if i < numPairs {
				currentRack = pairs[i].bitRack
				currentBits = pairs[i].blankLetterBit
			}
		}
	}
}

// ============================================================================
// Phase 3: Build double-blank entries
// ============================================================================

type doubleBlankPair struct {
	bitRack    BitRack
	packedPair uint16 // ml1 | (ml2 << 8)
}

func buildDoubleBlankEntries(wfl *ForLength, uniqueRacks []BitRack, wordLength, radixPasses int) {
	// Generate (rack_with_two_blanks, packed_pair) pairs for each ordered
	// pair of distinct present letters (l1 < l2).
	var pairs []doubleBlankPair
	for _, r := range uniqueRacks {
		rack := r
		present1 := rack.GetLetterMask() &^ 1
		for present1 != 0 {
			ml1 := byte(bits.TrailingZeros32(present1))
			present1 &= present1 - 1
			rack.TakeLetter(ml1)
			rack.AddLetter(blankMachineLetter)

			present2 := rack.GetLetterMask() &^ ((uint32(1) << ml1) - 1) &^ 1
			for present2 != 0 {
				ml2 := byte(bits.TrailingZeros32(present2))
				present2 &= present2 - 1
				rack.TakeLetter(ml2)
				rack.AddLetter(blankMachineLetter)

				pairs = append(pairs, doubleBlankPair{
					bitRack:    rack,
					packedPair: uint16(ml1) | uint16(ml2)<<8,
				})

				rack.TakeLetter(blankMachineLetter)
				rack.AddLetter(ml2)
			}

			rack.TakeLetter(blankMachineLetter)
			rack.AddLetter(ml1)
		}
	}
	numPairs := uint32(len(pairs))

	// Radix sort
	temp := make([]doubleBlankPair, numPairs)
	radixSortDoubleBlankPairs(pairs, temp, radixPasses)

	// Count unique racks
	var numUnique uint32
	if numPairs > 0 {
		numUnique = 1
		for i := uint32(1); i < numPairs; i++ {
			if !pairs[i].bitRack.Equals(&pairs[i-1].bitRack) {
				numUnique++
			}
		}
	}

	numBuckets := nextPowerOf2(numUnique)
	if numBuckets < minBuckets {
		numBuckets = minBuckets
	}
	wfl.NumDoubleBlankBuckets = numBuckets

	bucketCounts := make([]uint32, numBuckets)
	if numPairs > 0 {
		bucketCounts[pairs[0].bitRack.GetBucketIndex(numBuckets)]++
		for i := uint32(1); i < numPairs; i++ {
			if !pairs[i].bitRack.Equals(&pairs[i-1].bitRack) {
				bucketCounts[pairs[i].bitRack.GetBucketIndex(numBuckets)]++
			}
		}
	}

	wfl.NumDoubleBlankEntries = numUnique
	wfl.DoubleBlankMapEntries = make([]Entry, numUnique)
	wfl.DoubleBlankBucketStarts = make([]uint32, numBuckets+1)

	var offset uint32
	for b := uint32(0); b < numBuckets; b++ {
		wfl.DoubleBlankBucketStarts[b] = offset
		offset += bucketCounts[b]
	}
	wfl.DoubleBlankBucketStarts[numBuckets] = offset

	for i := range bucketCounts {
		bucketCounts[i] = 0
	}

	if numPairs > 0 {
		runStart := uint32(0)
		for i := uint32(1); i <= numPairs; i++ {
			isEnd := i == numPairs || !pairs[i].bitRack.Equals(&pairs[i-1].bitRack)
			if !isEnd {
				continue
			}
			rack := pairs[runStart].bitRack
			var firstBlankLetters uint32
			lastPair := uint16(0xFFFF)
			for j := runStart; j < i; j++ {
				if pairs[j].packedPair != lastPair {
					firstBlankLetters |= 1 << (pairs[j].packedPair & 0xFF)
					lastPair = pairs[j].packedPair
				}
			}

			bucketIdx := rack.GetBucketIndex(numBuckets)
			entryIdx := wfl.DoubleBlankBucketStarts[bucketIdx] + bucketCounts[bucketIdx]
			bucketCounts[bucketIdx]++
			entry := &wfl.DoubleBlankMapEntries[entryIdx]
			entry.SetFirstBlankLetters(firstBlankLetters)
			entry.WriteBitRack(&rack)

			runStart = i
		}
	}
}

// ============================================================================
// Radix sort implementations (LSD, byte-at-a-time)
// ============================================================================

// extractByte returns the byte at byteIdx of a BitRack treated as a
// 16-byte little-endian value.
func extractByte(br *BitRack, byteIdx int) uint8 {
	if byteIdx < 8 {
		return uint8(br.Low >> (byteIdx * 8))
	}
	return uint8(br.High >> ((byteIdx - 8) * 8))
}

func radixPassWordPairs(src, dst []wordPair, byteIdx int) {
	var counts [256]uint32
	for i := range src {
		counts[extractByte(&src[i].bitRack, byteIdx)]++
	}
	var total uint32
	for b := 0; b < 256; b++ {
		c := counts[b]
		counts[b] = total
		total += c
	}
	for i := range src {
		b := extractByte(&src[i].bitRack, byteIdx)
		dst[counts[b]] = src[i]
		counts[b]++
	}
}

func radixSortWordPairs(pairs, temp []wordPair, numPasses int) {
	if len(pairs) <= 1 {
		return
	}
	for pass := 0; pass < numPasses; pass++ {
		if pass%2 == 0 {
			radixPassWordPairs(pairs, temp, pass)
		} else {
			radixPassWordPairs(temp, pairs, pass)
		}
	}
	if numPasses%2 == 1 {
		copy(pairs, temp)
	}
}

func radixPassBlankPairs(src, dst []blankPair, byteIdx int) {
	var counts [256]uint32
	for i := range src {
		counts[extractByte(&src[i].bitRack, byteIdx)]++
	}
	var total uint32
	for b := 0; b < 256; b++ {
		c := counts[b]
		counts[b] = total
		total += c
	}
	for i := range src {
		b := extractByte(&src[i].bitRack, byteIdx)
		dst[counts[b]] = src[i]
		counts[b]++
	}
}

func radixSortBlankPairs(pairs, temp []blankPair, numPasses int) {
	if len(pairs) <= 1 {
		return
	}
	for pass := 0; pass < numPasses; pass++ {
		if pass%2 == 0 {
			radixPassBlankPairs(pairs, temp, pass)
		} else {
			radixPassBlankPairs(temp, pairs, pass)
		}
	}
	if numPasses%2 == 1 {
		copy(pairs, temp)
	}
}

func radixPassDoubleBlankPairs(src, dst []doubleBlankPair, byteIdx int) {
	var counts [256]uint32
	for i := range src {
		var b uint8
		switch {
		case byteIdx < 2:
			b = uint8(src[i].packedPair >> (byteIdx * 8))
		default:
			b = extractByte(&src[i].bitRack, byteIdx-2)
		}
		counts[b]++
	}
	var total uint32
	for b := 0; b < 256; b++ {
		c := counts[b]
		counts[b] = total
		total += c
	}
	for i := range src {
		var b uint8
		switch {
		case byteIdx < 2:
			b = uint8(src[i].packedPair >> (byteIdx * 8))
		default:
			b = extractByte(&src[i].bitRack, byteIdx-2)
		}
		dst[counts[b]] = src[i]
		counts[b]++
	}
}

func radixSortDoubleBlankPairs(pairs, temp []doubleBlankPair, bitrackPasses int) {
	if len(pairs) <= 1 {
		return
	}
	totalPasses := 2 + bitrackPasses
	for pass := 0; pass < totalPasses; pass++ {
		if pass%2 == 0 {
			radixPassDoubleBlankPairs(pairs, temp, pass)
		} else {
			radixPassDoubleBlankPairs(temp, pairs, pass)
		}
	}
	if totalPasses%2 == 1 {
		copy(pairs, temp)
	}
}

// ============================================================================
// max_word_lookup_bytes calculation
// ============================================================================

// wflGetWordCount returns the number of words for a given (blankless)
// rack in the given length, or 0 if not present.
func wflGetWordCount(wfl *ForLength, br *BitRack, wordLength int) uint32 {
	if wfl.NumWordBuckets == 0 {
		return 0
	}
	bucketIdx := br.GetBucketIndex(wfl.NumWordBuckets)
	start := wfl.WordBucketStarts[bucketIdx]
	end := wfl.WordBucketStarts[bucketIdx+1]
	for i := start; i < end; i++ {
		entryBR := wfl.WordMapEntries[i].ReadBitRack()
		if entryBR.Equals(br) {
			e := &wfl.WordMapEntries[i]
			if e.IsInlined() {
				return uint32(e.numberOfInlinedBytes(wordLength) / wordLength)
			}
			return e.NumWords()
		}
	}
	return 0
}

// calculateMaxWordLookupBytes computes the maximum number of bytes any
// double-blank lookup might produce. Used to size result buffers.
func calculateMaxWordLookupBytes(wmp *WMP, boardDim int) uint32 {
	var maxBytes uint32
	for length := 2; length <= boardDim; length++ {
		wfl := &wmp.WFLs[length]
		for i := uint32(0); i < wfl.NumDoubleBlankEntries; i++ {
			entry := &wfl.DoubleBlankMapEntries[i]
			rack := entry.ReadBitRack()
			firstBlanks := entry.FirstBlankLetters()
			var totalWords uint32

			rack.SetLetterCount(blankMachineLetter, 1)
			for ml1 := byte(1); ml1 < maxAlphabetSize; ml1++ {
				if firstBlanks&(1<<ml1) == 0 {
					continue
				}
				rack.AddLetter(ml1)

				// Look up the corresponding single-blank entry
				blankBucket := rack.GetBucketIndex(wfl.NumBlankBuckets)
				bstart := wfl.BlankBucketStarts[blankBucket]
				bend := wfl.BlankBucketStarts[blankBucket+1]
				for bi := bstart; bi < bend; bi++ {
					blankRack := wfl.BlankMapEntries[bi].ReadBitRack()
					if !blankRack.Equals(&rack) {
						continue
					}
					secondBlanks := wfl.BlankMapEntries[bi].BlankLetters()
					for ml2 := ml1; ml2 < maxAlphabetSize; ml2++ {
						if secondBlanks&(1<<ml2) == 0 {
							continue
						}
						rack.AddLetter(ml2)
						rack.SetLetterCount(blankMachineLetter, 0)
						totalWords += wflGetWordCount(wfl, &rack, length)
						rack.SetLetterCount(blankMachineLetter, 1)
						rack.TakeLetter(ml2)
					}
					break
				}
				rack.TakeLetter(ml1)
			}
			bytes := totalWords * uint32(length)
			if bytes > maxBytes {
				maxBytes = bytes
			}
		}
	}
	return maxBytes
}
