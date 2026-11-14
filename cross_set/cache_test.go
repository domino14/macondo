package cross_set

import (
	"math/rand"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

// sameCrosses fails the test if the two boards differ in any square's
// cross-set or cross-score, in either direction.
func sameCrosses(t *testing.T, what string, got, want *board.GameBoard) {
	t.Helper()
	for row := 0; row < want.Dim(); row++ {
		for col := 0; col < want.Dim(); col++ {
			for _, dir := range []board.BoardDirection{Horizontal, Vertical} {
				if got.GetCrossSet(row, col, dir) != want.GetCrossSet(row, col, dir) {
					t.Fatalf("%s: cross-set at row %d col %d dir %v is %x, want %x", what,
						row, col, dir, got.GetCrossSet(row, col, dir), want.GetCrossSet(row, col, dir))
				}
				if got.GetCrossScore(row, col, dir) != want.GetCrossScore(row, col, dir) {
					t.Fatalf("%s: cross-score at row %d col %d dir %v is %d, want %d", what,
						row, col, dir, got.GetCrossScore(row, col, dir), want.GetCrossScore(row, col, dir))
				}
			}
		}
	}
}

// The cached generator must write exactly what the plain one writes: on the
// test boards, on a second pass that is answered from the cache, and after
// moves; with a cache so small that entries are evicted all the time, and
// with a roomy one.
func TestCachedGeneratorMatchesPlain(t *testing.T) {
	is := is.New(t)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.WGLConfig())
	is.NoErr(err)
	alph := dist.TileMapping()
	plain := &GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}

	moves := []updateCrossesForMoveTestCase{
		{VsMatt, move.NewScoringMoveSimple(38, "K9", "TAEL", "ABD", alph), "TAEL"},
		{VsMatt2, move.NewScoringMoveSimple(77, "O8", "TENsILE", "", alph), "TENsILE"},
		{VsOxy, move.NewScoringMoveSimple(1780, "A1", "OX.P...B..AZ..E", "", alph),
			"OXYPHENBUTAZONE"},
		{VsJeremy, move.NewScoringMoveSimple(14, "1G", "S.oWED", "D?", alph), "SNoWED"},
		{VsJeremy, move.NewScoringMoveSimple(11, "15F", "F..ER", "", alph), "FOYER"},
		{VsEd, move.NewScoringMoveSimple(5, "N1", "AB", "", alph), "AB"},
	}
	for _, bits := range []int{2, 6, 14} {
		cached := NewCachedGenerator(dist, gd, bits)
		for _, tc := range moves {
			want := board.MakeBoard(board.CrosswordGameBoard)
			want.SetToGame(alph, tc.testGame)
			want.SetSkipExtensionSets(true)
			got := want.Copy()
			got.SetSkipExtensionSets(true)

			plain.GenerateAll(want)
			cached.GenerateAll(got)
			sameCrosses(t, "first pass", got, want)
			cached.GenerateAll(got)
			sameCrosses(t, "second pass", got, want)

			want.PlayMove(tc.m)
			plain.UpdateForMove(want, tc.m)
			got.PlayMove(tc.m)
			cached.UpdateForMove(got, tc.m)
			sameCrosses(t, "after the move", got, want)
			// And the whole board again, mostly from the cache.
			cached.GenerateAll(got)
			sameCrosses(t, "after the move, regenerated", got, want)
		}
		is.True(cached.Hits > 0)
		is.True(cached.Misses > 0)
	}
}

// Random boards -- any tiles anywhere, blanks included, words or not -- all
// answered by one cache, so that an entry written for one board is read for
// another. Verify is on as well: each hit is recomputed and compared.
func TestCachedGeneratorRandomBoards(t *testing.T) {
	is := is.New(t)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.WGLConfig())
	is.NoErr(err)
	plain := &GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
	rng := rand.New(rand.NewSource(7))
	// Few distinct letters, so that the same words come up again and again.
	letters := []tilemapping.MachineLetter{1, 5, 9, 14, 18, 19, 20, 1 | 0x80, 19 | 0x80}

	for _, verify := range []bool{false, true} {
		cached := NewCachedGenerator(dist, gd, 8)
		cached.Verify = verify
		for iter := 0; iter < 300; iter++ {
			want := board.MakeBoard(board.CrosswordGameBoard)
			want.SetSkipExtensionSets(true)
			// From sparse boards to nearly full lanes.
			fill := rng.Intn(180) + 5
			for i := 0; i < fill; i++ {
				want.SetLetter(rng.Intn(15), rng.Intn(15), letters[rng.Intn(len(letters))])
			}
			if iter%10 == 0 {
				// A lane with a single gap: the longest key there is.
				row := rng.Intn(15)
				gap := rng.Intn(15)
				for col := 0; col < 15; col++ {
					if col != gap {
						want.SetLetter(row, col, letters[rng.Intn(len(letters))])
					}
				}
			}
			got := want.Copy()
			got.SetSkipExtensionSets(true)
			plain.GenerateAll(want)
			cached.GenerateAll(got)
			sameCrosses(t, "random board", got, want)
		}
		is.True(cached.Hits > cached.Misses)
	}
}

// Two lanes that differ only in a blank, or only in where the gap is, have
// different cross-scores or cross-sets; the cache must not confuse them.
func TestCachedGeneratorTellsLanesApart(t *testing.T) {
	is := is.New(t)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.WGLConfig())
	is.NoErr(err)
	alph := dist.TileMapping()
	cached := NewCachedGenerator(dist, gd, 10)

	set := func(tiles string, row, col int) *board.GameBoard {
		b := board.MakeBoard(board.CrosswordGameBoard)
		b.SetSkipExtensionSets(true)
		mls, err := tilemapping.ToMachineLetters(tiles, alph)
		is.NoErr(err)
		for i, ml := range mls {
			if ml != 0 {
				b.SetLetter(row, col+i, ml)
			}
		}
		return b
	}
	// CAT then a gap, against CaT (blank A) then a gap: same hooks, the
	// score differs by the A.
	b := set("CAT", 7, 3)
	cached.Generate(b, 7, 6, Horizontal)
	is.Equal(b.GetCrossScore(7, 6, Horizontal), 5)
	is.Equal(b.GetCrossSet(7, 6, Horizontal), board.CrossSetFromString("ES", alph))
	b = set("CaT", 7, 3)
	cached.Generate(b, 7, 6, Horizontal)
	is.Equal(b.GetCrossScore(7, 6, Horizontal), 4)
	is.Equal(b.GetCrossSet(7, 6, Horizontal), board.CrossSetFromString("ES", alph))
	// The same three tiles with the gap in front: the front hooks.
	b = set("CAT", 7, 3)
	cached.Generate(b, 7, 2, Horizontal)
	is.Equal(b.GetCrossSet(7, 2, Horizontal), board.CrossSetFromString("S", alph))
	b = set("ARE", 7, 3)
	cached.Generate(b, 7, 2, Horizontal)
	front := b.GetCrossSet(7, 2, Horizontal)
	cached.Generate(b, 7, 6, Horizontal)
	back := b.GetCrossSet(7, 6, Horizontal)
	is.True(front != back)
	// A.E with the gap in the middle against AE then a gap.
	b = set("A", 7, 3)
	b.SetLetter(7, 5, 5)
	cached.Generate(b, 7, 4, Horizontal)
	middle := b.GetCrossSet(7, 4, Horizontal)
	b = set("AE", 7, 3)
	cached.Generate(b, 7, 5, Horizontal)
	is.True(middle != b.GetCrossSet(7, 5, Horizontal))
	is.Equal(cached.Hits, uint64(0))
}

// Reset must forget the old lexicon's answers.
func TestCachedGeneratorReset(t *testing.T) {
	is := is.New(t)
	nwl, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	csw, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.WGLConfig())
	is.NoErr(err)
	alph := dist.TileMapping()
	cached := NewCachedGenerator(dist, nwl, 12)
	for i, gd := range []*kwg.KWG{nwl, csw, nwl, csw} {
		if i > 0 {
			cached.Reset(dist, gd)
		}
		plain := &GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
		for _, g := range []board.VsWho{VsEd, VsJeremy, VsMatt, VsMatt2, VsOxy} {
			want := board.MakeBoard(board.CrosswordGameBoard)
			want.SetToGame(alph, g)
			want.SetSkipExtensionSets(true)
			got := want.Copy()
			got.SetSkipExtensionSets(true)
			plain.GenerateAll(want)
			cached.GenerateAll(got)
			sameCrosses(t, "after a reset", got, want)
		}
	}
}

// With extension sets on, a square's update writes more than the cache
// holds, so the generator must not cache at all, and must leave the board
// exactly as the plain generator does.
func TestCachedGeneratorOffWithExtensionSets(t *testing.T) {
	is := is.New(t)
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	dist, err := tilemapping.EnglishLetterDistribution(DefaultConfig.WGLConfig())
	is.NoErr(err)
	alph := dist.TileMapping()
	plain := &GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
	cached := NewCachedGenerator(dist, gd, 10)
	for _, g := range []board.VsWho{VsEd, VsJeremy, VsMatt, VsMatt2, VsOxy} {
		want := board.MakeBoard(board.CrosswordGameBoard)
		want.SetToGame(alph, g)
		got := want.Copy()
		plain.GenerateAll(want)
		cached.GenerateAll(got)
		cached.GenerateAll(got)
		is.True(got.Equals(want))
	}
	is.Equal(cached.Hits+cached.Misses, uint64(0))
}
