package playability

import "testing"

func TestCombosNoBlanks(t *testing.T) {
	// Without blanks, combos is a product of binomials: C(9,1)*C(1,1) for "AQ"
	// style alphagram with A=9, Q=1.
	dist := []uint8{0, 9, 1}
	if c := Combos(dist, []byte{1, 2}); c != 9 {
		t.Fatalf("got %v", c)
	}
	// With two blanks: A?:  C(9,1)*C(1,1) + C(2,1)*(C(1,1)+C(9,1)) + C(2,2)
	dist = []uint8{2, 9, 1}
	if c := Combos(dist, []byte{1, 2}); c != 9+2*10+1 {
		t.Fatalf("got %v", c)
	}
}
