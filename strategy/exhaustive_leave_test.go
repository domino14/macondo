package strategy

import (
	"os"
	"sort"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/movegen"
	"github.com/stretchr/testify/assert"
)

func TestLeaveMPH(t *testing.T) {
	els := ExhaustiveLeaveStrategy{}
	alph := alphabet.EnglishAlphabet()
	err := els.Init("NWL18", alph, os.Getenv("STRATEGY_PARAMS_PATH"))
	assert.Nil(t, err)

	type testcase struct {
		leave string
		ev    float64
	}

	for _, tc := range []testcase{
		{"?", 16.895568129846936},
		{"Q", -3.9469371326190554},
		{"?I", 18.447465459354476},
		{"I?", 18.447465459354476},
		{"?DLQSV", -3.9815582627814337},
		{"HMRRSS", -11.46401069407107},
		{"AEINST", 32.356318409092445},
		{"SATINE", 32.356318409092445},
	} {
		leave, _ := alphabet.ToMachineLetters(tc.leave, alph)
		assert.InEpsilon(t, tc.ev, els.LeaveValue(leave), 0.00001)
	}
}

func TestEndgameTiming(t *testing.T) {
	els := ExhaustiveLeaveStrategy{}
	gd, err := GaddagFromLexicon("NWL18")
	assert.Nil(t, err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld := alphabet.EnglishLetterDistribution(gd.GetAlphabet())
	generator := movegen.NewGordonGenerator(gd, bd, ld)
	tilesInPlay := bd.SetToGame(gd.GetAlphabet(), board.MavenVsMacondo)
	bd.GenAllCrossSets(gd, ld)
	generator.GenAll(alphabet.RackFromString("AEEORS?", alph), false)

	err = els.Init("NWL18", alph, os.Getenv("STRATEGY_PARAMS_PATH"))
	assert.Nil(t, err)

	oppRack := alphabet.NewRack(alph)
	oppRack.Set(tilesInPlay.Rack1)
	assert.Equal(t, oppRack.NumTiles(), uint8(2))

	bag := alphabet.NewBag(ld, alph)
	bag.Draw(100)

	plays := generator.Plays()

	for _, m := range plays {
		m.SetEquity(els.Equity(m, bd, bag, oppRack))
	}

	sort.Slice(plays, func(i, j int) bool {
		return plays[j].Equity() < plays[i].Equity()
	})

	assert.Equal(t, plays[0].Equity(), float64(22))
	// Use your blank
	assert.Equal(t, plays[0].ShortDescription(), "M6 RE.EArS")
	assert.Equal(t, plays[1].ShortDescription(), "L1 S..s")
	assert.Equal(t, plays[2].ShortDescription(), "M6 RE.EAmS")
}
