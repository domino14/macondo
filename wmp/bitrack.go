package wmp

import (
	"fmt"

	"github.com/domino14/word-golib/tilemapping"
)

// BitRack is a 128-bit multiset representation of tiles. Each letter position
// (0-31) gets 4 bits encoding its count (0-15 occurrences). This is used as
// the lookup key in WMP hash tables.
//
// Letter 0 = blank (undesignated), letters 1+ = actual tile letters.
// Arithmetic addition of two BitRacks performs union of the multisets.
type BitRack struct {
	Low  uint64
	High uint64
}

const (
	bitsPerLetter   = 4
	maxAlphabetSize = 32 // max letters in BitRack (128 bits / 4 bits per letter)
	letterMask      = (1 << bitsPerLetter) - 1
	// maxLetterCount is the largest count any single letter can hold in a
	// BitRack (4 bits → 0..15).
	maxLetterCount = (1 << bitsPerLetter) - 1
	// maxBlanks is the largest number of blanks the WMP supports per word.
	// The WMP only stores tables for 0, 1, and 2 blanks.
	maxBlanks = 2

	// MurmurHash3-style mixing constants
	hashRotationShift = 17
	hashMixConstant1  = 0xff51afd7ed558ccd
	hashMixConstant2  = 0xc4ceb9fe1a85ec53
)

// CheckCompatible reports whether a (LetterDistribution, boardDim) pair
// can be represented by a WMP. It mirrors MAGPIE's
// bit_rack_is_compatible_with_ld plus the >2-blanks guard from
// make_wmp_from_words.
//
// The constraints are:
//
//  1. The alphabet must fit in 32 letters (BitRack uses 4 bits per letter
//     across 128 bits).
//  2. The distribution must contain at most 2 blanks. The WMP only has
//     hash tables for 0, 1, and 2 blanks.
//  3. For each letter, the maximum number of times it can appear in a
//     single word — its base count plus all blanks substituted for it,
//     capped by the board dimension — must fit in 4 bits (≤ 15). The
//     blank itself is bounded by its own count (you can't substitute it
//     with anything).
//
// Returns nil if the language is compatible, or a descriptive error.
func CheckCompatible(ld *tilemapping.LetterDistribution, boardDim int) error {
	dist := ld.Distribution()
	if len(dist) > maxAlphabetSize {
		return fmt.Errorf("wmp: alphabet size %d exceeds max %d (BitRack is 32 letters)",
			len(dist), maxAlphabetSize)
	}
	numBlanks := int(dist[blankMachineLetter])
	if numBlanks > maxBlanks {
		return fmt.Errorf("wmp: distribution has %d blanks, max supported is %d",
			numBlanks, maxBlanks)
	}
	maxCount := 0
	for ml := 0; ml < len(dist); ml++ {
		count := int(dist[ml])
		if ml != blankMachineLetter {
			// A non-blank letter can appear in a word as itself or as
			// a designated blank, so its effective max is base + blanks.
			count += numBlanks
		}
		if count > maxCount {
			maxCount = count
		}
	}
	if maxCount > boardDim {
		maxCount = boardDim
	}
	if maxCount > maxLetterCount {
		return fmt.Errorf("wmp: max letter count %d exceeds 4-bit limit %d (board dim %d)",
			maxCount, maxLetterCount, boardDim)
	}
	return nil
}

// BitRackFromWord creates a BitRack from a word (sequence of MachineLetters).
func BitRackFromWord(word []byte) BitRack {
	var br BitRack
	for _, ml := range word {
		br.AddLetter(ml)
	}
	return br
}

// BitRackFromMachineWord creates a BitRack from a tilemapping.MachineWord.
func BitRackFromMachineWord(word tilemapping.MachineWord) BitRack {
	var br BitRack
	for _, ml := range word {
		br.AddLetter(byte(ml))
	}
	return br
}

// BitRackFromRack creates a BitRack from a tilemapping.Rack. Mirrors
// MAGPIE's bit_rack_create_from_rack: iterates the rack's per-letter
// counts and sets each letter's count in the BitRack.
func BitRackFromRack(rack *tilemapping.Rack) BitRack {
	var br BitRack
	for ml, count := range rack.LetArr {
		if count > 0 {
			br.SetLetterCount(byte(ml), count)
		}
	}
	return br
}

// GetLetter returns the count of the given letter in the BitRack.
func (br *BitRack) GetLetter(ml byte) int {
	shift := int(ml) * bitsPerLetter
	if shift < 64 {
		return int((br.Low >> shift) & letterMask)
	}
	return int((br.High >> (shift - 64)) & letterMask)
}

// SetLetterCount sets the count for the given letter.
func (br *BitRack) SetLetterCount(ml byte, count int) {
	shift := int(ml) * bitsPerLetter
	mask := uint64(letterMask)
	if shift < 64 {
		br.Low &= ^(mask << shift)
		br.Low |= uint64(count) << shift
	} else {
		s := shift - 64
		br.High &= ^(mask << s)
		br.High |= uint64(count) << s
	}
}

// AddLetter increments the count for the given letter by 1.
func (br *BitRack) AddLetter(ml byte) {
	shift := int(ml) * bitsPerLetter
	if shift < 64 {
		br.Low += 1 << shift
	} else {
		br.High += 1 << (shift - 64)
	}
}

// TakeLetter decrements the count for the given letter by 1.
func (br *BitRack) TakeLetter(ml byte) {
	shift := int(ml) * bitsPerLetter
	if shift < 64 {
		br.Low -= 1 << shift
	} else {
		br.High -= 1 << (shift - 64)
	}
}

// Equals returns true if two BitRacks represent the same multiset.
func (br *BitRack) Equals(other *BitRack) bool {
	return br.Low == other.Low && br.High == other.High
}

// AddBitRack adds another BitRack to this one (multiset union).
func (br *BitRack) AddBitRack(other *BitRack) {
	br.Low += other.Low
	br.High += other.High
}

// MixTo64 mixes the 128-bit BitRack down to a 64-bit hash with good
// avalanche properties (MurmurHash3-style).
//
// I tried writing a hand-rolled ARM64+amd64 Plan 9 assembly stub
// for this finalizer (see git history of bitrack_mix_*.s) and it
// turned out ~22% SLOWER on the microbench than the pure-Go form.
// The reason is that the Go compiler already inlines MixTo64 into
// its callers (it's small enough to fit the inlining cost budget),
// so the inlined code lives directly inside the bucket-scan loop
// with no call overhead. Replacing it with an out-of-line ABI0
// assembly function added a register-shuffling bridge per call
// that the savings on a half-dozen MUL/SHR/XOR couldn't recover.
//
// Lesson: hand-written assembly only beats Go for sequences that
// (a) are too complex for the Go compiler to inline, OR (b) need
// instructions Go doesn't expose (e.g., AES-NI, SHA-NI, AVX-512).
// MixTo64 is neither — it's pure scalar arithmetic that the Go
// SSA backend already lowers to optimal code.
func (br *BitRack) MixTo64() uint64 {
	low := br.Low
	high := br.High

	// Fold high into low with rotation
	low ^= high
	low ^= (high << hashRotationShift) | (high >> (64 - hashRotationShift))

	// MurmurHash3-style finalizer
	low ^= low >> 33
	low *= hashMixConstant1
	low ^= low >> 33
	low *= hashMixConstant2
	low ^= low >> 33

	return low
}

// GetBucketIndex returns the hash bucket index for this BitRack.
// numBuckets must be a power of 2.
func (br *BitRack) GetBucketIndex(numBuckets uint32) uint32 {
	return uint32(br.MixTo64()) & (numBuckets - 1)
}

// GetLetterMask returns a bitmask of which letters (0-31) are present
// (have nonzero count) in this BitRack.
func (br *BitRack) GetLetterMask() uint32 {
	low := br.Low
	high := br.High
	var mask uint32
	for i := 0; i < 16; i++ {
		if low&0xF != 0 {
			mask |= 1 << i
		}
		low >>= 4
	}
	for i := 0; i < 16; i++ {
		if high&0xF != 0 {
			mask |= 1 << (i + 16)
		}
		high >>= 4
	}
	return mask
}
