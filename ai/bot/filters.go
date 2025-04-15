package bot

import (
	"math"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

// Note: because of the nature of this algorithm, the lower these numbers, the
// more time the bot will take to find its move.
var BotConfigs = map[pb.BotRequest_BotCode]struct {
	baseFindability     float64
	longWordFindability float64
	parallelFindability float64
	isCommonWord        bool
}{
	pb.BotRequest_LEVEL1_COMMON_WORD_BOT: {baseFindability: 0.3, longWordFindability: 0.1, parallelFindability: 0.3, isCommonWord: true},
	pb.BotRequest_LEVEL2_COMMON_WORD_BOT: {baseFindability: 0.7, longWordFindability: 0.4, parallelFindability: 0.5, isCommonWord: true},
	pb.BotRequest_LEVEL3_COMMON_WORD_BOT: {baseFindability: 0.8, longWordFindability: 0.5, parallelFindability: 0.75, isCommonWord: true},
	pb.BotRequest_LEVEL4_COMMON_WORD_BOT: {baseFindability: 1.0, longWordFindability: 1.0, parallelFindability: 1.0, isCommonWord: true},

	pb.BotRequest_LEVEL1_PROBABILISTIC: {baseFindability: 0.2, longWordFindability: 0.07, parallelFindability: 0.15, isCommonWord: false},
	pb.BotRequest_LEVEL2_PROBABILISTIC: {baseFindability: 0.4, longWordFindability: 0.2, parallelFindability: 0.3, isCommonWord: false},
	pb.BotRequest_LEVEL3_PROBABILISTIC: {baseFindability: 0.55, longWordFindability: 0.35, parallelFindability: 0.45, isCommonWord: false},
	pb.BotRequest_LEVEL4_PROBABILISTIC: {baseFindability: 0.85, longWordFindability: 0.45, parallelFindability: 0.85, isCommonWord: false},
	pb.BotRequest_LEVEL5_PROBABILISTIC: {baseFindability: 0.9, longWordFindability: 0.8, parallelFindability: 0.85, isCommonWord: false},
}

func filter(cfg *config.Config, g *game.Game, rack *tilemapping.Rack, plays []*move.Move, botType pb.BotRequest_BotCode,
	lexName string) *move.Move {
	passMove := move.NewPassMove(rack.TilesOn(), g.Alphabet())
	botConfig, botConfigExists := BotConfigs[botType]
	if !botConfigExists {
		if len(plays) > 0 {
			return plays[0]
		}
		return passMove
	}

	filterFunction := func([]tilemapping.MachineWord, float64) (bool, error) { return true, nil }
	if botConfig.isCommonWord {
		pld, err := tilemapping.ProbableLetterDistributionName(lexName)
		if err != nil {
			log.Err(err).Str("lexicon", lexName).Msg("could-not-load-probable-letter-distribution-name")
			return passMove
		}
		var commonWordLexicon string
		switch pld {
		case "english":
			commonWordLexicon = "ECWL"
		case "german":
			commonWordLexicon = "CGL"
		default:
			log.Error().Str("ld", pld).Msg("no common word lexicon for this letter distribution")
			return passMove
		}

		gd, err := kwg.Get(cfg.WGLConfig(), commonWordLexicon)
		if err != nil {
			log.Err(err).Str("commonWordLexicon", commonWordLexicon).Msg("could-not-load-cwl")
			filterFunction = func([]tilemapping.MachineWord, float64) (bool, error) { return false, err }
		} else {
			lex := kwg.Lexicon{KWG: *gd}
			filterFunction = func(mws []tilemapping.MachineWord, r float64) (bool, error) {
				err = g.ValidateWords(lex, mws)
				if err != nil {
					// validation error means at least one word is phony.
					return false, nil
				}
				return true, nil
			}
		}
	}

	// LEVEL4_COMMON_WORD_BOT is an unfiltered common-word bot. Only filter if we're
	// not selecting this particular bot.
	if botType != pb.BotRequest_LEVEL4_COMMON_WORD_BOT {
		dist := g.Bag().LetterDistribution()
		// XXX: This should be cached
		subChooseCombos := createSubCombos(dist)
		filterFunctionPrev := filterFunction
		filterFunction = func(mws []tilemapping.MachineWord, r float64) (bool, error) {
			allowed, err := filterFunctionPrev(mws, r)
			if !allowed || err != nil {
				return allowed, err
			}
			ans := botConfig.baseFindability * math.Pow(botConfig.parallelFindability, float64(len(mws)-1))

			mw := mws[0] // assume len > 0
			// Check for long words (7 or more letters)
			if len(mw) >= g.ExchangeLimit() {
				ans *= probableFindability(len(mw), combinations(dist, subChooseCombos, mw, true)) * botConfig.longWordFindability
			}
			log.Debug().Float64("ans", ans).Float64("r", r).Msg("checking-answer")
			return r < ans, nil
		}
	}

	var mws []tilemapping.MachineWord
	for _, play := range plays {
		var err error
		allowed := true
		r := frand.Float64()

		if play.Action() == move.MoveTypePlay {
			mws, err = g.Board().FormedWords(play)
			if err != nil {
				log.Err(err).Msg("formed-words-filter-error")
				break
			}
			allowed, err = filterFunction(mws, r)
			if err != nil {
				log.Err(err).Msg("bot-type-move-filter-internal-error")
				break
			}
		} else if play.Action() == move.MoveTypeExchange {
			if r >= botConfig.baseFindability {
				allowed = false
			}
		}
		if allowed && err == nil {
			return play
		}
	}

	return passMove
}

func probableFindability(wordLen int, combos uint64) float64 {
	// This assumes the following preconditions:
	//   len(word) >= 2
	//   combos >= 1
	return math.Min(math.Log10(float64(combos))/float64(wordLen-1), 1.0)
}

func createSubCombos(dist *tilemapping.LetterDistribution) [][]uint64 {
	// Adapted from GPL Zyzzyva's calculation code.
	maxFrequency := uint8(0)
	totalLetters := uint8(0)
	for _, value := range dist.Distribution() {
		freq := value
		totalLetters += freq
		if freq > maxFrequency {
			maxFrequency = freq
		}
	}
	// Precalculate M choose N combinations
	r := uint8(1)
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

func combinations(dist *tilemapping.LetterDistribution, subChooseCombos [][]uint64,
	alphagram tilemapping.MachineWord, withBlanks bool) uint64 {
	// Adapted from GPL Zyzzyva's calculation code.
	letters := make([]tilemapping.MachineLetter, 0)
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
				subChooseCombos[dist.Distribution()[letter]])

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
		thisCombo = subChooseCombos[dist.Distribution()[0]][1]
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
			thisCombo = subChooseCombos[dist.Distribution()[0]][2]

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
