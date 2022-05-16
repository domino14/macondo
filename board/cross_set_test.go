package board

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
)

func TestCrossSet(t *testing.T) {
	cs := CrossSet(0)
	cs.Set(13)

	if uint64(cs) != 8192 /* 1<<13 */ {
		t.Errorf("Expected cross-set to be %v, got %v", 8192, cs)
	}
	cs.Set(0)
	if uint64(cs) != 8193 {
		t.Errorf("Expected cross-set to be %v, got %v", 8193, cs)
	}
}

type testpair struct {
	l       alphabet.MachineLetter
	allowed bool
}

func TestCrossSetAllowed(t *testing.T) {
	cs := CrossSet(8193)

	var allowedTests = []testpair{
		{alphabet.MachineLetter(1), false},
		{alphabet.MachineLetter(0), true},
		{alphabet.MachineLetter(14), false},
		{alphabet.MachineLetter(13), true},
		{alphabet.MachineLetter(12), false},
	}

	for _, pair := range allowedTests {
		allowed := cs.Allowed(pair.l)
		if allowed != pair.allowed {
			t.Errorf("For %v, expected %v, got %v", pair.l, pair.allowed,
				allowed)
		}
	}
}
