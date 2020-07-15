package alphabet

import (
	"encoding/csv"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/domino14/macondo/config"
	"github.com/rs/zerolog/log"
)

// LetterDistribution encodes the tile distribution for the relevant game.
type LetterDistribution struct {
	alph             *Alphabet
	Distribution     map[rune]uint8
	PointValues      map[rune]uint8
	SortOrder        map[rune]int
	Vowels           []rune
	numUniqueLetters int
	numLetters       int
	scores           []int
}

// EnglishLetterDistribution returns the English letter distribution.
func EnglishLetterDistribution(cfg *config.Config) (*LetterDistribution, error) {
	return NamedLetterDistribution(cfg, "english")
}

func SpanishLetterDistribution(cfg *config.Config) (*LetterDistribution, error) {
	return NamedLetterDistribution(cfg, "spanish")
}

// NamedLetterDistribution loads a letter distribution by name.
func NamedLetterDistribution(cfg *config.Config, name string) (*LetterDistribution, error) {
	name = strings.ToLower(name)
	filename := filepath.Join(cfg.LetterDistributionPath, name+".csv")

	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	r := csv.NewReader(file)
	dist := map[rune]uint8{}
	ptValues := map[rune]uint8{}
	sortOrder := []rune{}
	vowels := []rune{}
	alph := &Alphabet{}
	alph.Init()
	// letter,quantity,value,vowel
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		letter := rune(record[0][0])
		sortOrder = append(sortOrder, letter)
		n, err := strconv.Atoi(record[1])
		if err != nil {
			return nil, err
		}
		p, err := strconv.Atoi(record[2])
		if err != nil {
			return nil, err
		}
		v, err := strconv.Atoi(record[3])
		if err != nil {
			return nil, err
		}
		if v == 1 {
			vowels = append(vowels, letter)
		}
		dist[letter] = uint8(n)
		ptValues[letter] = uint8(p)
		if letter != BlankToken {
			// The Blank should not be part of the alphabet, only the letter dist.
			alph.Update(string(letter))
		}
	}
	alph.Reconcile()

	return newLetterDistribution(alph, dist, ptValues, makeSortMap(sortOrder), vowels), nil
}

func newLetterDistribution(alph *Alphabet, dist map[rune]uint8,
	ptValues map[rune]uint8, sortOrder map[rune]int, vowels []rune) *LetterDistribution {

	numTotalLetters := 0
	numUniqueLetters := len(dist)
	for _, v := range dist {
		numTotalLetters += int(v)
	}
	// Make an array of scores in alphabet order. This will allow for
	// fast lookups in move generators, etc, vs looking up a map.
	scores := make([]int, numUniqueLetters)
	for rn, ptVal := range ptValues {
		ml, err := alph.Val(rn)
		if err != nil {
			panic("Wrongly initialized")
		}
		if ml == BlankMachineLetter {
			ml = MachineLetter(numUniqueLetters - 1)
		}
		scores[ml] = int(ptVal)
	}

	return &LetterDistribution{
		alph:             alph,
		Distribution:     dist,
		PointValues:      ptValues,
		SortOrder:        sortOrder,
		Vowels:           vowels,
		numUniqueLetters: numUniqueLetters,
		numLetters:       numTotalLetters,
		scores:           scores,
	}

}

func (ld *LetterDistribution) Alphabet() *Alphabet {
	return ld.alph
}

func (ld *LetterDistribution) NumTotalTiles() int {
	return ld.numLetters
}

// MakeBag returns a bag of tiles.
func (ld *LetterDistribution) MakeBag(randSource *rand.Rand) *Bag {

	b := NewBag(ld, ld.alph, randSource)
	b.Shuffle()

	return b
}

// Score gives the score of the given machine letter. This is used by the
// move generator to score plays more rapidly than looking up a map.
func (ld *LetterDistribution) Score(ml MachineLetter) int {
	if ml >= BlankOffset || ml == BlankMachineLetter {
		return ld.scores[ld.numUniqueLetters-1]
	}
	return ld.scores[ml]
}

func makeSortMap(order []rune) map[rune]int {
	sortMap := make(map[rune]int)
	for idx, letter := range order {
		sortMap[letter] = idx
	}
	log.Debug().Interface("sortMap", sortMap).Interface("order", order).Msg("sort map")
	return sortMap
}
