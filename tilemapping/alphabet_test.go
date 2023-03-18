package tilemapping

import (
	"testing"

	"github.com/matryer/is"
)

func TestToMachineLetters(t *testing.T) {
	is := is.New(t)
	cat, err := GetDistribution(&DefaultConfig, "catalan")
	is.NoErr(err)
	// AL·LOQUIMIQUES is only 10 tiles despite being 14 codepoints long.
	// A L·L O QU I M I QU E S
	mls, err := ToMachineLetters("AL·LOQUIMIQUES", cat.TileMapping())
	is.NoErr(err)
	is.Equal(mls, []MachineLetter{
		1, 13, 17, 19, 10, 14, 10, 19, 6, 21,
	})
	mls, err = ToMachineLetters("Al·lOQUIMIquES", cat.TileMapping())
	is.NoErr(err)
	is.Equal(mls, []MachineLetter{
		1, 13 | 0x80, 17, 19, 10, 14, 10, 19 | 0x80, 6, 21,
	})
	mls, err = ToMachineLetters("ARQUEGESSIU", cat.TileMapping())
	is.NoErr(err)
	is.Equal(mls, []MachineLetter{
		1, 20, 19, 6, 8, 6, 21, 21, 10, 23,
	})
}

func TestUV(t *testing.T) {
	is := is.New(t)
	cat, err := GetDistribution(&DefaultConfig, "catalan")
	is.NoErr(err)

	uv := MachineWord([]MachineLetter{
		1, 13, 17, 19, 10, 14, 10, 19, 6, 21,
	}).UserVisible(cat.TileMapping())
	is.Equal(uv, "AL·LOQUIMIQUES")

	uv = MachineWord([]MachineLetter{
		1, 13 | 0x80, 17, 19, 10, 14, 10, 19 | 0x80, 6, 21,
	}).UserVisible(cat.TileMapping())
	is.Equal(uv, "Al·lOQUIMIquES")
}
