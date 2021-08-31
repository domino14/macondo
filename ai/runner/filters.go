package runner

import (
	"math"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"lukechampine.com/frand"
)

var BotTypeMoveFilterMap = map[pb.BotRequest_BotCode]func(*config.Config, []string, []uint64, pb.BotRequest_BotCode) (bool, error){
	pb.BotRequest_HASTY_BOT:            noFilter,
	pb.BotRequest_LEVEL1_CEL_BOT:       celFilter,
	pb.BotRequest_LEVEL2_CEL_BOT:       celFilter,
	pb.BotRequest_LEVEL3_CEL_BOT:       celFilter,
	pb.BotRequest_LEVEL1_PROBABILISTIC: findabilityFilter,
	pb.BotRequest_LEVEL2_PROBABILISTIC: findabilityFilter,
	pb.BotRequest_LEVEL3_PROBABILISTIC: findabilityFilter,
	pb.BotRequest_LEVEL4_PROBABILISTIC: findabilityFilter,
	pb.BotRequest_LEVEL5_PROBABILISTIC: findabilityFilter,
}

// Note: because of the nature of this algorithm, the lower these numbers, the
// more time the bot will take to find its move.
var BotFindabilities = map[pb.BotRequest_BotCode]float64{
	pb.BotRequest_LEVEL1_CEL_BOT:       0.2,
	pb.BotRequest_LEVEL2_CEL_BOT:       0.5,
	pb.BotRequest_LEVEL3_CEL_BOT:       1,
	pb.BotRequest_LEVEL1_PROBABILISTIC: 0.07,
	pb.BotRequest_LEVEL2_PROBABILISTIC: 0.15,
	pb.BotRequest_LEVEL3_PROBABILISTIC: 0.35,
	pb.BotRequest_LEVEL4_PROBABILISTIC: 0.6,
	pb.BotRequest_LEVEL5_PROBABILISTIC: 0.85, // Unlikely to be used for now; this should just be hasty
}

var BotParallelFindabilities = map[pb.BotRequest_BotCode]float64{
	pb.BotRequest_LEVEL1_CEL_BOT:       0.25,
	pb.BotRequest_LEVEL2_CEL_BOT:       0.5,
	pb.BotRequest_LEVEL3_CEL_BOT:       1,
	pb.BotRequest_LEVEL1_PROBABILISTIC: 0.1,
	pb.BotRequest_LEVEL2_PROBABILISTIC: 0.2,
	pb.BotRequest_LEVEL3_PROBABILISTIC: 0.45,
	pb.BotRequest_LEVEL4_PROBABILISTIC: 0.7,
	pb.BotRequest_LEVEL5_PROBABILISTIC: 0.85,
}

func noFilter(cfg *config.Config, words []string, combos []uint64, findability pb.BotRequest_BotCode) (bool, error) {
	return true, nil
}

func celFilter(cfg *config.Config, words []string, combos []uint64, findability pb.BotRequest_BotCode) (bool, error) {
	gd, err := gaddag.GetDawg(cfg, "ECWL")
	if err != nil {
		return false, err
	}
	for _, word := range words {
		isPhony, err := isPhony(gd, word)
		if err != nil {
			return false, nil
		}
		if isPhony {
			return false, nil
		}
	}
	return findabilityFilter(cfg, words, combos, findability)
}

func findabilityFilter(cfg *config.Config, words []string, combos []uint64, findability pb.BotRequest_BotCode) (bool, error) {
	finalProbableFindability := 1.0
	finalParallelFindability := 1.0
	for i, word := range words {
		wordLength := len(word)
		if i == 0 && (wordLength >= 7) {
			finalProbableFindability = probableFindability(word, combos[i])
		} else if i > 0 {
			finalParallelFindability = finalParallelFindability * BotParallelFindabilities[findability]
		}
	}
	finalFindability := finalProbableFindability * finalParallelFindability * BotFindabilities[findability]
	return frand.Float64() < finalFindability, nil
}

func probableFindability(words string, combos uint64) float64 {
	return math.Min(math.Log10(float64(combos))/float64(len(words)-1), 1.0)
}

func createSubCombos(dist *alphabet.LetterDistribution) [][]uint64 {
	// Adapted from GPL Zyzzyva's calculation code.
	maxFrequency := uint8(0)
	totalLetters := uint8(0)
	r := uint8(1)
	for _, value := range dist.Distribution {
		freq := value
		totalLetters += freq
		if freq > maxFrequency {
			maxFrequency = freq
		}
	}
	// Precalculate M choose N combinations
	subChooseCombos := make([][]uint64, maxFrequency+1)
	for i := uint8(0); i <= maxFrequency; i, r = i+1, r+1 {
		subList := make([]uint64, maxFrequency+1)
		for j := uint8(0); j <= maxFrequency; j++ {
			if (i == j) || (j == 0) {
				subList[j] = 1.0
			} else if i == 0 {
				subList[j] = 0.0
			} else {
				subList[j] = subChooseCombos[i-1][j-1] +
					subChooseCombos[i-1][j]
			}
		}
		subChooseCombos[i] = subList
	}
	return subChooseCombos
}

func combinations(dist *alphabet.LetterDistribution, subChooseCombos [][]uint64, alphagram string, withBlanks bool) uint64 {
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
				subChooseCombos[dist.Distribution[letter]])

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
	if !withBlanks {
		return totalCombos
	}
	// Calculate combinations with one blank
	for i := 0; i < numLetters; i++ {
		counts[i]--
		thisCombo = subChooseCombos[dist.Distribution['?']][1]
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
			thisCombo = subChooseCombos[dist.Distribution['?']][2]

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

func isPhony(gd gaddag.GenericDawg, word string) (bool, error) {
	lex := gaddag.Lexicon{GenericDawg: gd}
	machineWord, err := alphabet.ToMachineWord(word, lex.GetAlphabet())
	if err != nil {
		return false, err
	}
	return !lex.HasWord(machineWord), nil
}
