package tilemapping

import (
	"encoding/csv"
	"io"
	"strconv"

	"github.com/domino14/macondo/config"
)

// LetterDistribution encodes the tile distribution for the relevant game.
type LetterDistribution struct {
	tilemapping      *TileMapping
	Vowels           []MachineLetter
	distribution     []uint8
	scores           []int
	numUniqueLetters uint
	numLetters       uint
	Name             string
}

func ScanLetterDistribution(data io.Reader) (*LetterDistribution, error) {
	r := csv.NewReader(data)
	dist := []uint8{}
	ptValues := []int{}
	vowels := []MachineLetter{}
	alph := &TileMapping{}
	alph.Init()
	// letter,quantity,value,vowel
	idx := 0
	letters := []string{}
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		letter := record[0]
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
			vowels = append(vowels, MachineLetter(idx))
		}
		dist = append(dist, uint8(n))
		ptValues = append(ptValues, p)
		letters = append(letters, letter)
		idx++
	}
	alph.Reconcile(letters)
	return newLetterDistribution(alph, dist, ptValues, vowels), nil
}

func newLetterDistribution(alph *TileMapping, dist []uint8,
	ptValues []int, vowels []MachineLetter) *LetterDistribution {

	numTotalLetters := uint(0)
	numUniqueLetters := uint(len(dist))
	for _, v := range dist {
		numTotalLetters += uint(v)
	}
	// Note: numUniqueLetters/numTotalLetters includes the blank.

	return &LetterDistribution{
		tilemapping:      alph,
		distribution:     dist,
		scores:           ptValues,
		Vowels:           vowels,
		numUniqueLetters: numUniqueLetters,
		numLetters:       numTotalLetters,
	}

}

func makeSortMap(order []string) map[string]int {
	sortMap := make(map[string]int)
	for idx, letter := range order {
		sortMap[letter] = idx
	}
	return sortMap
}

// Score gives the score of the given machine letter. This is used by the
// move generator to score plays more rapidly than looking up a map.
func (ld *LetterDistribution) Score(ml MachineLetter) int {
	if ml.IsBlanked() {
		return ld.scores[0] // the blank
	}
	return ld.scores[ml]
}

func (d *LetterDistribution) TileMapping() *TileMapping {
	return d.tilemapping
}

// Score returns the score of this word given the ld.
func (ld *LetterDistribution) WordScore(mw MachineWord) int {
	score := 0
	for _, c := range mw {
		score += ld.Score(c)
	}
	return score
}

func (ld *LetterDistribution) Distribution() []uint8 {
	return ld.distribution
}

// EnglishLetterDistribution returns the English letter distribution.
func EnglishLetterDistribution(cfg *config.Config) (*LetterDistribution, error) {
	return NamedLetterDistribution(cfg, "english")
}

// MakeBag returns a bag of tiles.
func (ld *LetterDistribution) MakeBag() *Bag {

	b := NewBag(ld, ld.tilemapping)
	b.Shuffle()

	return b
}
