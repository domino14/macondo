package alphabet

import (
	"testing"
)

type alphagramtestpair struct {
	word      string
	dist      *LetterDistribution
	alphagram string
}

func TestAlphagram(t *testing.T) {

	englishLD, err := EnglishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	spanishLD, err := SpanishLetterDistribution(&DefaultConfig)
	if err != nil {
		t.Error(err)
	}
	var utilsTests = []alphagramtestpair{
		{"FIREFANG", englishLD, "AEFFGINR"},
		{"QAJAQ", englishLD, "AAJQQ"},
		{"EROTICA", englishLD, "ACEIORT"},
		{"MUUMUUS", englishLD, "MMSUUUU"},
		{"PRIVATDOZENT", englishLD, "ADEINOPRTTVZ"},
		{"DEUTERANOMALIES", englishLD, "AADEEEILMNORSTU"},
		{"1ARMAQUITO", spanishLD, "AA1IMOQRTU"},
		{"ÑOÑE3IN1AS", spanishLD, "A1EINÑÑO3S"},
	}

	for _, pair := range utilsTests {
		word := Word{Word: pair.word, Dist: pair.dist}
		alphagram := word.MakeAlphagram()
		if alphagram != pair.alphagram {
			t.Error("For", pair.word, "expected", pair.alphagram,
				"got", alphagram)
		}

	}
}
