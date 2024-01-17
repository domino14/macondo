package equity_test

import (
	"sort"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/stretchr/testify/assert"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/testhelpers"
)

var DefaultConfig = config.DefaultConfig()

func GaddagFromLexicon(lex string) (gaddag.WordGraph, error) {
	return kwg.Get(DefaultConfig.AllSettings(), lex)
}

func TestLeaveValues(t *testing.T) {
	alph := testhelpers.EnglishAlphabet()

	els, err := equity.NewExhaustiveLeaveCalculator("NWL20", &DefaultConfig, "")
	assert.Nil(t, err)

	type testcase struct {
		leave string
		ev    float64
	}

	for _, tc := range []testcase{
		{"?", 24.504596710205078},
		{"Q", -6.793293476104736},
		{"?I", 25.864593505859375},
		{"I?", 25.864593505859375},
		{"?DLQSV", 10.06197452545166},
		{"HMRRSS", -2.2670013904571533},
		{"AEINST", 30.4634952545166},
		{"SATINE", 30.4634952545166},
	} {
		leave, _ := tilemapping.ToMachineLetters(tc.leave, alph)
		assert.Equal(t, tc.ev, els.LeaveValue(leave))
	}
}

func TestOtherLeaveValues(t *testing.T) {
	alph := testhelpers.EnglishAlphabet()

	els, err := equity.NewExhaustiveLeaveCalculator("NWL18", &DefaultConfig, "")
	assert.Nil(t, err)

	type testcase struct {
		leave string
		ev    float64
	}

	for _, tc := range []testcase{
		{"", 0},
		{"ABCDEFG", 0},
	} {
		leave, _ := tilemapping.ToMachineLetters(tc.leave, alph)
		assert.Equal(t, tc.ev, els.LeaveValue(leave))
	}
}

func BenchmarkLeaveValue(b *testing.B) {
	alph := testhelpers.EnglishAlphabet()

	els, _ := equity.NewExhaustiveLeaveCalculator("NWL18", &DefaultConfig, "leaves.klv2")
	leave, _ := tilemapping.ToMachineLetters("AENST", alph)

	for i := 0; i <= b.N; i++ {
		els.LeaveValue(leave)
	}
}

func TestEndgameTiming(t *testing.T) {
	gd, err := GaddagFromLexicon("NWL18")
	assert.Nil(t, err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	assert.Nil(t, err)
	generator := movegen.NewGordonGenerator(gd, bd, ld)
	tilesInPlay := bd.SetToGame(gd.GetAlphabet(), board.MavenVsMacondo)
	cross_set.GenAllCrossSets(bd, gd, ld)
	generator.GenAll(tilemapping.RackFromString("AEEORS?", alph), false)

	els, err := equity.NewExhaustiveLeaveCalculator("NWL18", &DefaultConfig, "")
	assert.Nil(t, err)

	eac := &equity.EndgameAdjustmentCalculator{}

	oppRack := tilemapping.NewRack(alph)
	oppRack.Set(tilesInPlay.Rack1)
	assert.Equal(t, oppRack.NumTiles(), uint8(2))

	bag := tilemapping.NewBag(ld, alph)
	ml := make([]tilemapping.MachineLetter, 100)
	err = bag.Draw(100, ml)
	assert.Nil(t, err)
	plays := generator.Plays()

	for _, m := range plays {
		m.SetEquity(els.Equity(m, bd, bag, oppRack) + eac.Equity(m, bd, bag, oppRack))
	}

	sort.Slice(plays, func(i, j int) bool {
		if plays[j].Equity() == plays[i].Equity() {
			return plays[i].ShortDescription() < plays[j].ShortDescription()
		}
		return plays[j].Equity() < plays[i].Equity()
	})
	assert.Equal(t, plays[0].Equity(), float64(22))
	// Use your blank
	assert.Equal(t, plays[0].ShortDescription(), " L1 S..s")
	assert.Equal(t, plays[1].ShortDescription(), " M6 RE.EAmS")
	assert.Equal(t, plays[2].ShortDescription(), " M6 RE.EArS")
}

func TestPreendgameTiming(t *testing.T) {
	gd, err := GaddagFromLexicon("NWL20")
	assert.Nil(t, err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	assert.Nil(t, err)
	generator := movegen.NewGordonGenerator(gd, bd, ld)
	tilesInPlay := bd.SetToGame(gd.GetAlphabet(), board.VsOxy)
	cross_set.GenAllCrossSets(bd, gd, ld)
	generator.GenAll(tilemapping.RackFromString("OXPBAZE", alph), false)

	els, err := equity.NewExhaustiveLeaveCalculator("NWL20", &DefaultConfig, "")
	assert.Nil(t, err)
	pac, err := equity.NewPreEndgameAdjustmentCalculator(&DefaultConfig, "NWL20", "quackle_preendgame.json")
	assert.Nil(t, err)
	bag := tilemapping.NewBag(ld, alph)
	bag.RemoveTiles(tilesInPlay.OnBoard)
	bag.RemoveTiles(tilesInPlay.Rack1)
	bag.RemoveTiles(tilesInPlay.Rack2)

	plays := generator.Plays()

	for _, m := range plays {
		// OppRack can be nil because that branch of code that checks it
		// will never be called.
		m.SetEquity(els.Equity(m, bd, bag, nil) + pac.Equity(m, bd, bag, nil))
	}

	sort.Slice(plays, func(i, j int) bool {
		return plays[j].Equity() < plays[i].Equity()
	})

	// There are 5 tiles in the bag. 5 - 7 (used tiles) + 7 = 5.
	// This should add a penalty of -3.5 (see quackle_preendgame.json)

	assert.Equal(t, plays[0].Equity(), float64(1780-3.5))
}

func TestOpeningPlayHeuristic(t *testing.T) {
	gd, err := GaddagFromLexicon("NWL20")
	assert.Nil(t, err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.AllSettings())
	assert.Nil(t, err)
	generator := movegen.NewGordonGenerator(gd, bd, ld)
	cross_set.GenAllCrossSets(bd, gd, ld)
	generator.GenAll(tilemapping.RackFromString("AEFLR", alph), false)
	els, err := equity.NewCombinedStaticCalculator(
		"NWL20", &DefaultConfig, "", "")
	assert.Nil(t, err)
	plays := generator.Plays()
	bag := tilemapping.NewBag(ld, alph)
	for _, m := range plays {
		// OppRack can be nil because that branch of code that checks it
		// will never be called.
		m.SetEquity(els.Equity(m, bd, bag, nil))
	}
	sort.Slice(plays, func(i, j int) bool {
		return plays[j].Equity() < plays[i].Equity()
	})
	// first two plays have 24 equity: FLARE and FARLE, because they
	// do not expose any vowels next to 2LS
	// third play FERAL even though it scores 24 should have a penalty.
	assert.Equal(t, plays[2].ShortDescription(), " 8D FERAL")
	assert.Equal(t, plays[2].Equity(), 23.3)
}
