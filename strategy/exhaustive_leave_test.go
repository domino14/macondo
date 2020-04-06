package strategy

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/stretchr/testify/assert"
)

func TestLeaveMPH(t *testing.T) {
	t.Skip()
	els := ExhaustiveLeaveStrategy{}
	alph := alphabet.EnglishAlphabet()
	err := els.Init("NWL18", alph, "leave_values_112719.idx.gz")
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
