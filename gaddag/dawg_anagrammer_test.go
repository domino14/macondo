package gaddag

import (
	"testing"

	"github.com/domino14/macondo/alphabet"
)

func BenchmarkAnagramBlanks(b *testing.B) {
	// ~1.78 ms on 12thgen-monolith

	d, err := loadDawg("CSW21", false)
	if err != nil {
		b.Error("loading America dawg")
		return
	}
	alph := d.GetAlphabet()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var anags []string
		da := DawgAnagrammer{}
		if err = da.InitForString(d, "RETINA??"); err != nil {
			b.Error(err)
		} else if err = da.Anagram(d, func(word alphabet.MachineWord) error {
			anags = append(anags, word.UserVisible(alph))
			return nil
		}); err != nil {
			b.Error(err)
		}
	}
}
