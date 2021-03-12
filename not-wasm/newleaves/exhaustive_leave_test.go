package newleaves

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
	"github.com/stretchr/testify/assert"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	for _, lex := range []string{"NWL18"} {
		gdgPath := filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag")
		if _, err := os.Stat(gdgPath); os.IsNotExist(err) {
			gaddagmaker.GenerateGaddag(filepath.Join(DefaultConfig.LexiconPath, lex+".txt"), true, true)
			err = os.Rename("out.gaddag", gdgPath)
			if err != nil {
				panic(err)
			}
		}
	}
	os.Exit(m.Run())
}

func GaddagFromLexicon(lex string) (*gaddag.SimpleGaddag, error) {
	return gaddag.LoadGaddag(filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag"))
}

func TestLeaveMPH(t *testing.T) {
	alph := alphabet.EnglishAlphabet()

	els, err := strategy.NewExhaustiveLeaveStrategy("NWL18", alph, &DefaultConfig, "", "")
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
		leave, _ := alphabet.ToMachineLetters(tc.leave, alph)
		assert.InEpsilon(t, tc.ev, els.LeaveValue(leave), 0.00001)
	}
}

func TestEndgameTiming(t *testing.T) {
	gd, err := GaddagFromLexicon("NWL18")
	assert.Nil(t, err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	assert.Nil(t, err)
	generator := movegen.NewGordonGenerator(gd, bd, ld)
	tilesInPlay := bd.SetToGame(gd.GetAlphabet(), board.MavenVsMacondo)
	cross_set.GenAllCrossSets(bd, gd, ld)
	generator.GenAll(alphabet.RackFromString("AEEORS?", alph), false)

	els, err := strategy.NewExhaustiveLeaveStrategy("NWL18", alph, &DefaultConfig, "", "")

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

func TestPreendgameTiming(t *testing.T) {
	gd, err := GaddagFromLexicon("NWL18")
	assert.Nil(t, err)
	alph := gd.GetAlphabet()
	bd := board.MakeBoard(board.CrosswordGameBoard)
	ld, err := alphabet.EnglishLetterDistribution(&DefaultConfig)
	assert.Nil(t, err)
	generator := movegen.NewGordonGenerator(gd, bd, ld)
	tilesInPlay := bd.SetToGame(gd.GetAlphabet(), board.VsOxy)
	cross_set.GenAllCrossSets(bd, gd, ld)
	generator.GenAll(alphabet.RackFromString("OXPBAZE", alph), false)

	els, err := strategy.NewExhaustiveLeaveStrategy("NWL18", alph, &DefaultConfig, "", "quackle_preendgame.json")
	assert.Nil(t, err)

	bag := alphabet.NewBag(ld, alph)
	bag.RemoveTiles(tilesInPlay.OnBoard)
	bag.RemoveTiles(tilesInPlay.Rack1)
	bag.RemoveTiles(tilesInPlay.Rack2)

	plays := generator.Plays()

	for _, m := range plays {
		// OppRack can be nil because that branch of code that checks it
		// will never be called.
		m.SetEquity(els.Equity(m, bd, bag, nil))
	}

	sort.Slice(plays, func(i, j int) bool {
		return plays[j].Equity() < plays[i].Equity()
	})

	// There are 5 tiles in the bag. 5 - 7 (used tiles) + 7 = 5.
	// This should add a penalty of -3.5 (see quackle_preendgame.json)

	assert.Equal(t, plays[0].Equity(), float64(1780-3.5))
}

func BenchmarkOldThing(b *testing.B) {
	alph := alphabet.EnglishAlphabet()

	els, err := strategy.NewExhaustiveLeaveStrategy("NWL18", alph, &DefaultConfig, "", "")
	_ = err

	b.ResetTimer()

	v := 0.0

	type testcase struct {
		leave string
		ev    float64
	}

	for i := 0; i < b.N; i++ {
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
			leave, _ := alphabet.ToMachineLetters(tc.leave, alph)
			v += els.LeaveValue(leave)
			if i == 0 {
				//fmt.Println(tc.leave, tc.ev, "els", els.LeaveValue(leave))
			}
		}
	}

	fmt.Println("els", b.N, v, "els")
}

func BenchmarkImprovedOldThing1(b *testing.B) {
	olv, err := (func() (*OldLeaves, error) {
		f, err := os.Open("data.olv")
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return ReadOldLeaves(f)
	})()
	if err != nil {
		panic(err)
	}

	alph := alphabet.EnglishAlphabet()

	b.ResetTimer()

	v := 0.0

	type testcase struct {
		leave string
		ev    float64
	}

	for i := 0; i < b.N; i++ {
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
			leave, _ := alphabet.ToMachineLetters(tc.leave, alph)
			v += olv.LeaveValue1(leave)
			if i == 0 {
				//fmt.Println(tc.leave, tc.ev, "olv1", olv.LeaveValue1(leave))
			}
		}
	}

	fmt.Println("olv1", b.N, v, "olv1")
}

func BenchmarkImprovedOldThing2(b *testing.B) {
	olv, err := (func() (*OldLeaves, error) {
		f, err := os.Open("data.olv")
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return ReadOldLeaves(f)
	})()
	if err != nil {
		panic(err)
	}

	alph := alphabet.EnglishAlphabet()

	b.ResetTimer()

	v := 0.0

	type testcase struct {
		leave string
		ev    float64
	}

	for i := 0; i < b.N; i++ {
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
			leave, _ := alphabet.ToMachineLetters(tc.leave, alph)
			v += olv.LeaveValue2(leave)
			if i == 0 {
				//fmt.Println(tc.leave, tc.ev, "olv2", olv.LeaveValue2(leave))
			}
		}
	}

	fmt.Println("olv2", b.N, v, "olv2")
}

func BenchmarkNewThing1(b *testing.B) {
	nlv, err := (func() (*NewLeaves, error) {
		f, err := os.Open("data.nlv")
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return ReadNewLeaves(f)
	})()
	if err != nil {
		panic(err)
	}

	alph := alphabet.EnglishAlphabet()

	b.ResetTimer()

	v := 0.0

	type testcase struct {
		leave string
		ev    float64
	}

	for i := 0; i < b.N; i++ {
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
			leave, _ := alphabet.ToMachineLetters(tc.leave, alph)
			v += nlv.LeaveValue1(leave)
			if i == 0 {
				//fmt.Println(tc.leave, tc.ev, "nlv1", nlv.LeaveValue1(leave))
			}
		}
	}

	fmt.Println("nlv1", b.N, v, "nlv1")
}

func BenchmarkNewThing2(b *testing.B) {
	nlv, err := (func() (*NewLeaves, error) {
		f, err := os.Open("data.nlv")
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return ReadNewLeaves(f)
	})()
	if err != nil {
		panic(err)
	}

	alph := alphabet.EnglishAlphabet()

	b.ResetTimer()

	v := 0.0

	type testcase struct {
		leave string
		ev    float64
	}

	for i := 0; i < b.N; i++ {
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
			leave, _ := alphabet.ToMachineLetters(tc.leave, alph)
			v += nlv.LeaveValue2(leave)
			if i == 0 {
				//fmt.Println(tc.leave, tc.ev, "nlv2", nlv.LeaveValue2(leave))
			}
		}
	}

	fmt.Println("nlv2", b.N, v, "nlv2")
}

func BenchmarkNewThing3(b *testing.B) {
	nlv, err := (func() (*NewLeavesAlt, error) {
		f, err := os.Open("data.nlv")
		if err != nil {
			return nil, err
		}
		defer f.Close()
		o, err := ReadNewLeaves(f)
		if err != nil {
			return nil, err
		}
		return MakeNewLeavesAlt(o), nil
	})()
	if err != nil {
		panic(err)
	}

	alph := alphabet.EnglishAlphabet()

	b.ResetTimer()

	v := 0.0

	type testcase struct {
		leave string
		ev    float64
	}

	for i := 0; i < b.N; i++ {
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
			leave, _ := alphabet.ToMachineLetters(tc.leave, alph)
			v += nlv.LeaveValue3(leave)
			if i == 0 {
				//fmt.Println(tc.leave, tc.ev, "nlv3", nlv.LeaveValue3(leave))
			}
		}
	}

	fmt.Println("nlv3", b.N, v, "nlv3")
}

func BenchmarkNewThing4(b *testing.B) {
	nlv, err := (func() (*NewLeavesAlt, error) {
		f, err := os.Open("data.nlv")
		if err != nil {
			return nil, err
		}
		defer f.Close()
		o, err := ReadNewLeaves(f)
		if err != nil {
			return nil, err
		}
		return MakeNewLeavesAlt(o), nil
	})()
	if err != nil {
		panic(err)
	}

	alph := alphabet.EnglishAlphabet()

	b.ResetTimer()

	v := 0.0

	type testcase struct {
		leave string
		ev    float64
	}

	for i := 0; i < b.N; i++ {
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
			leave, _ := alphabet.ToMachineLetters(tc.leave, alph)
			v += nlv.LeaveValue4(leave)
			if i == 0 {
				//fmt.Println(tc.leave, tc.ev, "nlv4", nlv.LeaveValue4(leave))
			}
		}
	}

	fmt.Println("nlv4", b.N, v, "nlv4")
}
