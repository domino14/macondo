package movegen

import (
	"os"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
)

const emptyWordSmogCGP = "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex CSW21; var wordsmog;"

// wordSmogStep is one position from magpie's wordsmog_test
// (test/move_gen_test.c). tiles is the play as magpie writes it: parenthesized
// runs are played through, lowercase letters are designated blanks.
type wordSmogStep struct {
	rack   string
	coords string
	tiles  string
	score  int
}

var magpieWordSmogSequence = []wordSmogStep{
	{"FEEZEEE", "8D", "ZEEFE", 54},
	{"HOWDYES", "7D", "ODHYW", 55},
	{"SIXATAS", "6D", "SIXAT", 88},
	{"OSSTTUU", "E2", "OSST(IDE)TUU", 94},
	{"AEGNSUV", "C8", "SVAEGU", 46},
	{"AEHLLOY", "B10", "HLOYL", 61},
	{"BEENROW", "A11", "WBEON", 62},
	{"ADEEIMN", "5G", "DAEEIMN", 88},
	{"PINLIK?", "F2", "KIIL(XHE)sPN", 136}, // SPHINXLIKE
	{"AEINNR?", "D3", "ANI(SOZ)ERcN", 182}, // CANONIZERS
}

// macondoTiles rewrites magpie's play notation into macondo's: magpie
// parenthesizes played-through runs, macondo uses one '.' per tile.
func macondoTiles(tiles string) string {
	var sb strings.Builder
	inPlaythrough := false
	for _, r := range tiles {
		switch {
		case r == '(':
			inPlaythrough = true
		case r == ')':
			inPlaythrough = false
		case inPlaythrough:
			sb.WriteRune(tilemapping.ASCIIPlayedThrough)
		default:
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

func wordSmogGame(t *testing.T) *game.Game {
	t.Helper()
	if os.Getenv("MACONDO_DATA_PATH") == "" {
		t.Skip("MACONDO_DATA_PATH not set")
	}
	g, err := cgp.ParseCGP(DefaultConfig, emptyWordSmogCGP)
	if err != nil {
		t.Skip("could not set up a WordSmog game (is CSW21.kad present?): " + err.Error())
	}
	return g.Game
}

func newWordSmogGenerator(t *testing.T, g *game.Game) *GordonGenerator {
	t.Helper()
	gd, err := GaddagFromLexicon(g.LexiconName())
	if err != nil {
		t.Fatal(err)
	}
	gen := NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
	gen.SetWordSmog(g.Rules().AlphaDawg())
	gen.SetGame(g)
	return gen
}

// TestWordSmogPlaysMagpieSequence walks magpie's wordsmog_test sequence. At
// each position it checks that (a) macondo scores magpie's play identically,
// (b) the play's words are legal WordSmog words, and (c) the alpha generator
// actually finds that play.
func TestWordSmogPlaysMagpieSequence(t *testing.T) {
	is := is.New(t)
	g := wordSmogGame(t)
	alph := g.Alphabet()
	gen := newWordSmogGenerator(t, g)

	for _, step := range magpieWordSmogSequence {
		m, err := g.CreateAndScorePlacementMove(step.coords, macondoTiles(step.tiles), step.rack, false)
		is.NoErr(err)
		if m.Score() != step.score {
			t.Fatalf("%s %s: expected score %d, got %d", step.coords, step.tiles,
				step.score, m.Score())
		}

		// The formed words must all be legal in WordSmog.
		words, err := g.Board().FormedWords(m)
		is.NoErr(err)
		is.NoErr(g.ValidateWords(g.Lexicon(), words))

		// The generator must find it.
		rack := tilemapping.RackFromString(step.rack, alph)
		is.NoErr(g.SetRackFor(g.PlayerOnTurn(), rack))
		gen.SetPlayRecorder(AllPlaysRecorder)
		plays := gen.GenAll(rack, false)
		if !containsPlay(plays, m) {
			t.Fatalf("%s %s (%d): alpha generator did not find this play among %d plays",
				step.coords, step.tiles, step.score, len(plays))
		}

		is.NoErr(g.PlayMove(m, true, 0))
	}
}

// TestWordSmogShadowAgreement checks that best-first generation with shadow
// finds the same top-scoring play as exhaustive generation, at every position
// of magpie's sequence. This is the test that catches a wrong bingo gate or a
// zeroed extension set: both make shadow silently drop anchors.
func TestWordSmogShadowAgreement(t *testing.T) {
	is := is.New(t)
	g := wordSmogGame(t)
	alph := g.Alphabet()

	genAll := newWordSmogGenerator(t, g)
	genShadow := newWordSmogGenerator(t, g)
	genShadow.SetPlayRecorderTopPlay()

	for _, step := range magpieWordSmogSequence {
		rack := tilemapping.RackFromString(step.rack, alph)
		is.NoErr(g.SetRackFor(g.PlayerOnTurn(), rack))

		all := genAll.GenAll(rack, false)
		is.True(len(all) > 0)
		sort.Slice(all, func(i, j int) bool { return all[i].Score() > all[j].Score() })
		best := all[0]

		shadowPlays := genShadow.GenAll(rack, false)
		is.Equal(len(shadowPlays), 1)
		if shadowPlays[0].Score() != best.Score() {
			t.Fatalf("at %s: shadow found %s (%d), exhaustive found %s (%d)",
				step.coords, shadowPlays[0].ShortDescription(), shadowPlays[0].Score(),
				best.ShortDescription(), best.Score())
		}

		m, err := g.CreateAndScorePlacementMove(step.coords, macondoTiles(step.tiles), step.rack, false)
		is.NoErr(err)
		is.NoErr(g.PlayMove(m, true, 0))
	}
}

// TestWordSmogGeneratesAnagrams checks the defining property of the variant:
// on an empty board, plays whose letters are only an anagram of a word are
// generated, and plays whose letters spell nothing in any order are not.
func TestWordSmogGeneratesAnagrams(t *testing.T) {
	is := is.New(t)
	g := wordSmogGame(t)
	alph := g.Alphabet()
	gen := newWordSmogGenerator(t, g)

	rack := tilemapping.RackFromString("AEINRST", alph)
	is.NoErr(g.SetRackFor(g.PlayerOnTurn(), rack))
	plays := gen.GenAll(rack, false)

	var sevens, jumbledSevens int
	for _, p := range plays {
		if p.TilesPlayed() != 7 {
			continue
		}
		sevens++
		// RETAINS is a word; NASTIER, RATINES, RETSINA, STAINER... are all
		// anagrams of each other. A permutation that isn't itself a word (say
		// TIRANES) is legal here and would never come out of the classic
		// generator.
		if !g.Lexicon().HasAnagram(p.Tiles()) {
			t.Fatal("every generated play must be a legal WordSmog word")
		}
		mw := slices.Clone(p.Tiles())
		slices.Sort(mw)
		if string(mw) != string(p.Tiles()) {
			jumbledSevens++
		}
	}
	is.True(sevens > 0)
	// The whole point: most bingos found are jumbles, not dictionary spellings.
	is.True(jumbledSevens > 0)

	// A rack whose letters spell nothing in any order produces no 7-tile play.
	rack2 := tilemapping.RackFromString("TRONGLE", alph)
	is.NoErr(g.SetRackFor(g.PlayerOnTurn(), rack2))
	for _, p := range gen.GenAll(rack2, false) {
		is.True(p.TilesPlayed() != 7)
	}
}

func containsPlay(plays []*move.Move, m *move.Move) bool {
	mr, mc, mv := m.CoordsAndVertical()
	for _, p := range plays {
		r, c, v := p.CoordsAndVertical()
		if r != mr || c != mc || v != mv || p.Score() != m.Score() {
			continue
		}
		if len(p.Tiles()) != len(m.Tiles()) {
			continue
		}
		same := true
		for i := range p.Tiles() {
			if p.Tiles()[i] != m.Tiles()[i] {
				same = false
				break
			}
		}
		if same {
			return true
		}
	}
	return false
}
