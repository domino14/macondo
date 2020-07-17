package alphabet

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

var LexiconDir = os.Getenv("LEXICON_PATH")

func TestScoreOn(t *testing.T) {
	ld, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	type racktest struct {
		rack string
		pts  int
	}
	testCases := []racktest{
		{"ABCDEFG", 16},
		{"XYZ", 22},
		{"??", 0},
		{"?QWERTY", 21},
		{"RETINAO", 7},
	}
	for _, tc := range testCases {
		r := RackFromString(tc.rack, ld.Alphabet())
		score := r.ScoreOn(ld)
		if score != tc.pts {
			t.Errorf("For %v, expected %v, got %v", tc.rack, tc.pts, score)
		}
	}
}

func TestRackFromString(t *testing.T) {
	alph := EnglishAlphabet()
	rack := RackFromString("AENPPSW", alph)

	expected := make([]int, MaxAlphabetSize+1)
	expected[0] = 1
	expected[4] = 1
	expected[13] = 1
	expected[15] = 2
	expected[18] = 1
	expected[22] = 1

	assert.Equal(t, expected, rack.LetArr)

}

func TestRackTake(t *testing.T) {
	alph := EnglishAlphabet()
	rack := RackFromString("AENPPSW", alph)
	rack.Take(MachineLetter(15))
	expected := make([]int, MaxAlphabetSize+1)
	expected[0] = 1
	expected[4] = 1
	expected[13] = 1
	expected[15] = 1
	expected[18] = 1
	expected[22] = 1

	assert.Equal(t, expected, rack.LetArr)

	rack.Take(MachineLetter(15))
	expected[15] = 0
	assert.Equal(t, expected, rack.LetArr)
}

func TestRackTakeAll(t *testing.T) {
	alph := EnglishAlphabet()
	rack := RackFromString("AENPPSW", alph)

	rack.Take(MachineLetter(15))
	rack.Take(MachineLetter(15))
	rack.Take(MachineLetter(0))
	rack.Take(MachineLetter(4))
	rack.Take(MachineLetter(13))
	rack.Take(MachineLetter(18))
	rack.Take(MachineLetter(22))
	expected := make([]int, MaxAlphabetSize+1)

	assert.Equal(t, expected, rack.LetArr)
}

func TestRackTakeAndAdd(t *testing.T) {
	alph := EnglishAlphabet()
	rack := RackFromString("AENPPSW", alph)

	rack.Take(MachineLetter(15))
	rack.Take(MachineLetter(15))
	rack.Take(MachineLetter(0))
	rack.Add(MachineLetter(0))

	expected := make([]int, MaxAlphabetSize+1)
	expected[0] = 1
	expected[4] = 1
	expected[13] = 1
	expected[18] = 1
	expected[22] = 1

	assert.Equal(t, expected, rack.LetArr)

}
