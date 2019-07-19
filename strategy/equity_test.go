package strategy

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/move"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/stretchr/testify/assert"
)

var LexiconDir = os.Getenv("LEXICON_PATH")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/gen_nwl18.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "NWL18.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/gen_nwl18.gaddag")
	}
	os.Exit(m.Run())
}

func TestCreateLeaveMap(t *testing.T) {
	gd, _ := gaddag.LoadGaddag("/tmp/gen_nwl18.gaddag")

	sss := SimpleSynergyStrategy{}
	err := sss.Init("NWL18", gd.GetAlphabet(), "leave_values_010919_v4.csv")
	assert.Nil(t, err)

	type testcase struct {
		leave   string
		synergy float64
		ev      float64
	}

	for _, tc := range []testcase{
		{"Q", 0, -4.049634765297462},
		{"FO", 0.6130525209743425, -1.9677355130602692},
		{"??", -9.455016582289886, 22.01544689734004},
	} {
		leave, _ := alphabet.ToMachineOnlyString(tc.leave, gd.GetAlphabet())
		assert.Equal(t, SynergyAndEV{
			synergy: tc.synergy,
			ev:      tc.ev,
		}, sss.leaveMap[leave])

	}
}

func TestSimpleSynergyLookup(t *testing.T) {
	gd, _ := gaddag.LoadGaddag("/tmp/gen_nwl18.gaddag")

	sss := SimpleSynergyStrategy{}
	sss.Init("NWL18", gd.GetAlphabet(), "leave_values_010919_v4.csv")

	type testcase struct {
		leave string
		ev    float64
	}

	for _, tc := range []testcase{
		{"", 0},
		{"Q", -4.049634765297462},
		{"RE", 4.790123765017576},
		{"ENARS", 15.304076461857065},
		{"AATA", -12.145087426808537},
	} {
		leave, _ := alphabet.ToMachineWord(tc.leave, gd.GetAlphabet())
		assert.Equal(t, tc.ev, sss.lookup(leave))
	}
}

func TestPlacementAdjustment(t *testing.T) {
	gd, _ := gaddag.LoadGaddag("/tmp/gen_nwl18.gaddag")
	alph := gd.GetAlphabet()

	vowelPenalty := -0.7

	var cases = []struct {
		pos     string
		word    string
		penalty float64
	}{
		{"H8", "TUX", vowelPenalty},
		{"H7", "TUX", 0.0},
		{"H6", "TUX", vowelPenalty},
		{"H5", "EUOI", vowelPenalty},
		{"H6", "EUOI", vowelPenalty * 2},
		{"H7", "EUOI", vowelPenalty * 2},
		{"H8", "EUOI", vowelPenalty},
		{"8D", "MAYBE", 0.0},
		{"8H", "MAYBE", vowelPenalty},
		{"8D", "MaYBe", 0.0},
		{"8H", "MaYBe", vowelPenalty},
	}

	for _, tc := range cases {
		move := move.NewScoringMoveSimple(42, /* score doesn't matter */
			tc.pos, tc.word, "", alph)
		adj := placementAdjustment(move)
		assert.Equal(t, tc.penalty, adj)
	}
}

func TestShouldExchange(t *testing.T) {
	gd, _ := gaddag.LoadGaddag("/tmp/gen_nwl18.gaddag")

	sss := SimpleSynergyStrategy{}
	sss.Init("NWL18", gd.GetAlphabet(), "leave_values_010919_v4.csv")

	//	rack := "COTTTV?"
	// leave1 assumes we exchange TTV
	leave1, _ := alphabet.ToMachineWord("COT?", gd.GetAlphabet())
	// leave2 assumes we play TOT
	leave2, _ := alphabet.ToMachineWord("CTV?", gd.GetAlphabet())
	// The equity of leave1 is greater than the equity of leave2 by at
	// least 7 pts.
	assert.True(t, sss.lookup(leave1) > (sss.lookup(leave2)+6+1))

}
