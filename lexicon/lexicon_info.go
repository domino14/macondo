package lexicon

import (
	"github.com/domino14/macondo/gaddag"
	"math/rand"
	"time"
)

type LetterDistribution struct {
	distribution map[rune]uint8
	SortOrder    map[rune]int
	numLetters   int
}

func EnglishLetterDistribution() LetterDistribution {
	dist := map[rune]uint8{
		'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
		'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
		'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
		'Y': 2, 'Z': 1, '?': 2,
	}
	return LetterDistribution{dist, makeSortMap("ABCDEFGHIJKLMNOPQRSTUVWXYZ?"),
		100}
}

func SpanishLetterDistribution() LetterDistribution {
	dist := map[rune]uint8{
		'1': 1, '2': 1, '3': 1, // 1: CH, 2: LL, 3: RR
		'A': 12, 'B': 2, 'C': 4, 'D': 5, 'E': 12, 'F': 1, 'G': 2, 'H': 2,
		'I': 6, 'J': 1, 'L': 4, 'M': 2, 'N': 5, 'Ñ': 1, 'O': 9, 'P': 2,
		'Q': 1, 'R': 5, 'S': 6, 'T': 4, 'U': 5, 'V': 1, 'X': 1, 'Y': 1,
		'Z': 1, '?': 2,
	}
	return LetterDistribution{dist, makeSortMap("ABC1DEFGHIJL2MNÑOPQR3STUVXYZ?"),
		100}
}

// Bag returns a shuffled bag of tiles.
func (ld LetterDistribution) MakeBag() Bag {
	rand.Seed(time.Now().UnixNano())
	bag := make([]rune, ld.numLetters)
	idx := 0
	for rn, val := range ld.distribution {
		for j := uint8(0); j < val; j++ {
			bag[idx] = rn
			idx += 1
		}
	}
	// shuffle
	for i := range bag {
		j := rand.Intn(i + 1)
		bag[i], bag[j] = bag[j], bag[i]
	}
	b := Bag{tiles: bag, idx: 0}
	return b
}

func makeSortMap(order string) map[rune]int {
	sortMap := make(map[rune]int)
	for idx, letter := range order {
		sortMap[letter] = idx
	}
	return sortMap
}

type LexiconInfo struct {
	LexiconName        string
	LexiconFilename    string
	LexiconIndex       uint8
	DescriptiveName    string
	LetterDistribution LetterDistribution
	subChooseCombos    [][]uint64
	Gaddag             gaddag.SimpleGaddag
}

// Initialize the LexiconInfo data structure for a new lexicon,
// pre-calculating combinations as necessary.
func (l *LexiconInfo) Initialize() {
	// Adapted from GPL Zyzzyva's calculation code.
	maxFrequency := uint8(0)
	totalLetters := uint8(0)
	r := uint8(1)
	for _, value := range l.LetterDistribution.distribution {
		freq := value
		totalLetters += freq
		if freq > maxFrequency {
			maxFrequency = freq
		}
	}
	// Precalculate M choose N combinations
	l.subChooseCombos = make([][]uint64, maxFrequency+1)
	for i := uint8(0); i <= maxFrequency; i, r = i+1, r+1 {
		subList := make([]uint64, maxFrequency+1)
		for j := uint8(0); j <= maxFrequency; j++ {
			if (i == j) || (j == 0) {
				subList[j] = 1.0
			} else if i == 0 {
				subList[j] = 0.0
			} else {
				subList[j] = l.subChooseCombos[i-1][j-1] +
					l.subChooseCombos[i-1][j]
			}
		}
		l.subChooseCombos[i] = subList
	}
}

// Calculate the number of combinations for an alphagram.
func (l *LexiconInfo) Combinations(alphagram string) uint64 {
	// Adapted from GPL Zyzzyva's calculation code.
	letters := make([]rune, 0)
	counts := make([]uint8, 0)
	combos := make([][]uint64, 0)
	for _, letter := range alphagram {
		foundLetter := false
		for j, char := range letters {
			if char == letter {
				counts[j]++
				foundLetter = true
				break
			}
		}
		if !foundLetter {
			letters = append(letters, letter)
			counts = append(counts, 1)
			combos = append(combos,
				l.subChooseCombos[l.LetterDistribution.distribution[letter]])

		}
	}
	totalCombos := uint64(0)
	numLetters := len(letters)
	// Calculate combinations with no blanks
	thisCombo := uint64(1)
	for i := 0; i < numLetters; i++ {
		thisCombo *= combos[i][counts[i]]
	}
	totalCombos += thisCombo
	// Calculate combinations with one blank
	for i := 0; i < numLetters; i++ {
		counts[i]--
		thisCombo = l.subChooseCombos[l.LetterDistribution.distribution['?']][1]
		for j := 0; j < numLetters; j++ {
			thisCombo *= combos[j][counts[j]]
		}
		totalCombos += thisCombo
		counts[i]++
	}
	// Calculate combinations with two blanks
	for i := 0; i < numLetters; i++ {
		counts[i]--
		for j := i; j < numLetters; j++ {
			if counts[j] == 0 {
				continue
			}
			counts[j]--
			thisCombo = l.subChooseCombos[l.LetterDistribution.distribution['?']][2]

			for k := 0; k < numLetters; k++ {
				thisCombo *= combos[k][counts[k]]
			}
			totalCombos += thisCombo
			counts[j]++
		}
		counts[i]++
	}
	return totalCombos
}

// func (l *lexiconInfo) blankCombinations(numBlanks uint8, counts []uint8,
//  combos [][]uint64) uint64 {
//  for i := 0; i < numLetters; i++ {

//  }
// }
