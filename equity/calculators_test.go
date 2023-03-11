package equity_test

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/testcommon"
	"github.com/domino14/macondo/tilemapping"
	"github.com/stretchr/testify/assert"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	testcommon.CreateGaddags(DefaultConfig, []string{"NWL18"})
	os.Exit(m.Run())
}

func GaddagFromLexicon(lex string) (gaddag.WordGraph, error) {
	return kwg.LoadKWG(&DefaultConfig, filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".kwg"))
}

func TestLeaveMPH(t *testing.T) {
	alph := tilemapping.EnglishAlphabet()

	els, err := equity.NewExhaustiveLeaveCalculator("NWL18", &DefaultConfig, "")
	assert.Nil(t, err)

	type testcase struct {
		leave string
		ev    float64
	}

	for _, tc := range []testcase{
		{"?", 25.19870376586914},
		{"Q", -7.26110315322876},
		{"?I", 26.448156356811523},
		{"I?", 26.448156356811523},
		{"?DLQSV", -1.2257566452026367},
		{"HMRRSS", -7.6917290687561035},
		{"AEINST", 30.734148025512695},
		{"SATINE", 30.734148025512695},
	} {
		leave, _ := tilemapping.ToMachineLetters(tc.leave, alph)
		assert.InEpsilon(t, tc.ev, els.LeaveValue(leave), 0.00001)
	}
}

func TestEndgameTiming(t *testing.T) {
	gd, err := GaddagFromLexicon("NWL18")
	assert.Nil(t, err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
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
	gd, err := GaddagFromLexicon("NWL18")
	assert.Nil(t, err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	assert.Nil(t, err)
	generator := movegen.NewGordonGenerator(gd, bd, ld)
	tilesInPlay := bd.SetToGame(gd.GetAlphabet(), board.VsOxy)
	cross_set.GenAllCrossSets(bd, gd, ld)
	generator.GenAll(tilemapping.RackFromString("OXPBAZE", alph), false)

	els, err := equity.NewExhaustiveLeaveCalculator("NWL18", &DefaultConfig, "")
	assert.Nil(t, err)
	pac, err := equity.NewPreEndgameAdjustmentCalculator(&DefaultConfig, "NWL18", "quackle_preendgame.json")
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
