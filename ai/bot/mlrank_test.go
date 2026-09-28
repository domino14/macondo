package bot

import (
	"math"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
)

func englishAlphabet(t *testing.T) *tilemapping.TileMapping {
	t.Helper()
	ld, err := tilemapping.EnglishLetterDistribution(config.DefaultConfig().WGLConfig())
	is.New(t).NoErr(err)
	return ld.TileMapping()
}

// play makes a horizontal play of `word` scoring `score` (tiles played = len(word)).
func play(t *testing.T, alph *tilemapping.TileMapping, word string, score int) *move.Move {
	t.Helper()
	mw, err := tilemapping.ToMachineWord(word, alph)
	is.New(t).NoErr(err)
	m := move.NewScoringMove(score, mw, nil, false, len(word), alph, 7, 0)
	return m
}

func norm(spread float64) float32 { return game.NormalizeSpreadForML(float32(spread)) }

func TestRankMLMovesContestedUsesValue(t *testing.T) {
	is := is.New(t)
	alph := englishAlphabet(t)
	moves := []*move.Move{play(t, alph, "ZA", 40), play(t, alph, "RETAINS", 70), play(t, alph, "AT", 8)}
	values := []float32{0.31, 0.28, 0.35}
	spreads := []float32{norm(40), norm(200), norm(-5)}
	got := rankMLMoves(moves, values, spreads, -10, mlDecidedValue)
	is.Equal(got[0], moves[2]) // highest value wins; the spread head is not consulted
	is.Equal(got[1], moves[0])
	is.Equal(got[2], moves[1])
}

func TestRankMLMovesDecidedLossPrefersSpread(t *testing.T) {
	is := is.New(t)
	alph := englishAlphabet(t)
	// Down 256 with the bag nearly empty: every candidate is a certain loss to
	// the value head; the 176-point bingo is the best final spread by far.
	dump, bingo, other := play(t, alph, "V", 7), play(t, alph, "VORTICS", 176), play(t, alph, "VI", 14)
	moves := []*move.Move{dump, bingo, other}
	values := []float32{-0.9982, -0.9990, -0.9985}
	spreads := []float32{norm(17), norm(-11), norm(10)}
	got := rankMLMoves(moves, values, spreads, -256, mlDecidedValue)
	is.Equal(got[0], bingo)
	// With the rule off, the (noisy) value order stands.
	got = rankMLMoves(moves, values, spreads, -256, 0)
	is.Equal(got[0], dump)
}

func TestRankMLMovesDecidedWinKeepsClearlyBetterValue(t *testing.T) {
	is := is.New(t)
	alph := englishAlphabet(t)
	// Won position, but one candidate is distinctly less certain than the
	// others: the tie window keeps it out of the spread ranking.
	safe, bingo, risky := play(t, alph, "AT", 20), play(t, alph, "AIRLIKE", 158), play(t, alph, "QI", 60)
	moves := []*move.Move{safe, bingo, risky}
	values := []float32{0.9990, 0.9996, 0.9800}
	spreads := []float32{norm(7), norm(-16), norm(60)}
	got := rankMLMoves(moves, values, spreads, 81, mlDecidedValue)
	is.Equal(got[0], bingo) // among the near-ties (safe, bingo) the bingo's final spread is higher
	is.Equal(got[2], risky) // 0.98 is outside the 0.01 window below 0.9996
}

func TestExpectedFinalSpreadUndoesTheNormalization(t *testing.T) {
	is := is.New(t)
	alph := englishAlphabet(t)
	m := play(t, alph, "AT", 30)
	got := expectedFinalSpread(m, -50, norm(25))
	is.True(math.Abs(got-(-50+30+25)) < 0.01)
	ex := move.NewExchangeMove(nil, nil, alph)
	is.True(math.Abs(expectedFinalSpread(ex, -50, norm(25))-(-25)) < 0.01)
}
