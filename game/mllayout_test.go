package game

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
)

func layoutFor(t *testing.T, dist string) *MLLayout {
	t.Helper()
	ld, err := tilemapping.GetDistribution(DefaultConfig.WGLConfig(), dist)
	if err != nil {
		t.Fatal(err)
	}
	l, err := MLLayoutFor(ld)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func letters(t *testing.T, dist string, mls []tilemapping.MachineLetter) []string {
	t.Helper()
	ld, err := tilemapping.GetDistribution(DefaultConfig.WGLConfig(), dist)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, ml := range mls {
		out = append(out, ld.TileMapping().Letter(ml))
	}
	return out
}

// Every shipped model reads this layout: it must not move.
func TestEnglishMLLayout(t *testing.T) {
	is := is.New(t)
	l := layoutFor(t, "english")
	is.Equal(l.Letters, 26)
	is.Equal(l.NumPlanes, 85)
	is.Equal(l.NumScalars, 72)
	is.Equal(l.PlaneLen, 85*15*15)
	is.Equal(l.RowLen, 85*15*15+72)
	is.Equal([]int{l.blankPlane, l.horCCPlanes, l.verCCPlanes, l.bonusPlanes, l.historyPlanes},
		[]int{26 * 225, 27 * 225, 53 * 225, 79 * 225, 83 * 225})
	is.Equal([]int{l.historyScal, l.powerScal, l.vcBagScal, l.vcRackScal, l.addlScal},
		[]int{54, 57, 63, 65, 67})
	is.Equal(letters(t, "english", l.power[:]), []string{"?", "J", "Q", "S", "X", "Z"})
	is.Equal(l.powerCount, [6]float32{2, 1, 1, 4, 1, 1}) // blank/2, S/4
	is.Equal(letters(t, "english", l.vowels), []string{"A", "E", "I", "O", "U"})
}

func TestFrenchMLLayout(t *testing.T) {
	is := is.New(t)
	l := layoutFor(t, "french")
	is.Equal(l.NumPlanes, 85) // the English shape, but not the English features
	is.Equal(l.NumScalars, 72)
	is.Equal(letters(t, "french", l.power[1:]), []string{"K", "W", "X", "Y", "Z"}) // the 10-pointers
	is.Equal(letters(t, "french", l.vowels), []string{"A", "E", "I", "O", "U", "Y"})
}

func TestGermanMLLayout(t *testing.T) {
	is := is.New(t)
	l := layoutFor(t, "german")
	is.Equal(l.Letters, 29) // A-Z and Ä Ö Ü
	is.Equal(l.NumPlanes, 94)
	is.Equal(l.NumScalars, 78)
	is.Equal(l.historyPlanes+2*mlPlaneSize, l.PlaneLen)
	is.Equal(l.addlScal+5, l.NumScalars)
	// Q, Y (10), Ö, X (8), then the first 6-pointer, Ä.
	is.Equal(letters(t, "german", l.power[1:]), []string{"Ä", "Ö", "Q", "X", "Y"})
	is.Equal(letters(t, "german", l.vowels), []string{"A", "Ä", "E", "I", "O", "Ö", "U", "Ü"})
}
