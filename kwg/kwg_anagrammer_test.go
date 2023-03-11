package kwg

import (
	"testing"

	"github.com/domino14/macondo/tilemapping"
	"github.com/matryer/is"
)

func BenchmarkAnagramBlanks(b *testing.B) {
	// ~1.78 ms on 12thgen-monolith
	is := is.New(b)
	kwg, err := Get(&DefaultConfig, "CSW21")
	is.NoErr(err)
	alph := kwg.GetAlphabet()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var anags []string
		da := KWGAnagrammer{}
		if err = da.InitForString(kwg, "RETINA??"); err != nil {
			b.Error(err)
		} else if err = da.Anagram(kwg, func(word tilemapping.MachineWord) error {
			anags = append(anags, word.UserVisible(alph))
			return nil
		}); err != nil {
			b.Error(err)
		}
	}
}
