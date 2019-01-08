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

var LexiconDir = os.Getenv("LEXICON_DIR")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/gen_america.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "America.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/gen_america.gaddag")
	}
	os.Exit(m.Run())
}

func TestCreateLeaveMap(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")

	sss := SimpleSynergyStrategy{}
	err := sss.Init("America", gd.GetAlphabet())
	assert.Nil(t, err)

	type testcase struct {
		leave   string
		synergy float64
		ev      float64
	}

	for _, tc := range []testcase{
		// the Q is worth -1.4250594492829194 (lol)
		{"Q", 0, -1.4250594492829194},
		{"FO", 0.49861204019961036, 0.2284590808105662},
		{"??", 0.16499348055292984, 38.98224629850245},
	} {
		leave, _ := alphabet.ToMachineOnlyString(tc.leave, gd.GetAlphabet())
		assert.Equal(t, SynergyAndEV{
			synergy: tc.synergy,
			ev:      tc.ev,
		}, sss.leaveMap[leave])

	}
}

func TestSimpleSynergyLookup(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")

	sss := SimpleSynergyStrategy{}
	sss.Init("America", gd.GetAlphabet())

	type testcase struct {
		leave string
		ev    float64
	}

	for _, tc := range []testcase{
		{"", 0},
		{"Q", -1.4250594492829194},
		{"RE", 4.75999566616953},
		{"ENARS", 18.136184658534305},
		{"AATA", -8.205913901558304},
	} {
		leave, _ := alphabet.ToMachineWord(tc.leave, gd.GetAlphabet())
		assert.Equal(t, tc.ev, sss.lookup(leave))
	}
}

func TestPlacementAdjustment(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
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
	gd := gaddag.LoadGaddag("/tmp/gen_america.gaddag")

	sss := SimpleSynergyStrategy{}
	sss.Init("America", gd.GetAlphabet())

	//	rack := "COTTTV?"
	// leave1 assumes we exchange TTV
	leave1, _ := alphabet.ToMachineWord("COT?", gd.GetAlphabet())
	// leave2 assumes we play TOT
	leave2, _ := alphabet.ToMachineWord("CTV?", gd.GetAlphabet())
	// The equity of leave1 is greater than the equity of leave2 by at
	// least 7 pts.
	assert.True(t, sss.lookup(leave1) > (sss.lookup(leave2)+6+1))

}
