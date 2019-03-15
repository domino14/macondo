package endgame

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"

	"github.com/domino14/macondo/move"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/stretchr/testify/assert"
)

var LexiconDir = os.Getenv("LEXICON_DIR")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/gen_csw15.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "CSW15.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/gen_csw15.gaddag")
	}
	os.Exit(m.Run())
}

func TestAddEntryToTable(t *testing.T) {

	pt := make(playTable)
	pt.addEntry("foo", 0, -15, nil)

	assert.Equal(t, 1, len(pt))
	assert.Equal(t, 1, len(pt["foo"]))
	assert.Nil(t, pt["foo"][0].move)
	assert.Equal(t, -15, pt["foo"][0].value)
}

func TestAddEntryToTableAlreadyExisting(t *testing.T) {
	pt := make(playTable)
	pt.addEntry("foo", 3, 15, nil)
	pt.addEntry("foo", 3, 40, nil)

	assert.Equal(t, 1, len(pt))
	assert.Equal(t, 1, len(pt["foo"]))
	assert.Nil(t, pt["foo"][3].move)
	assert.Equal(t, 40, pt["foo"][3].value)
}

func TestGetEntryFromTable(t *testing.T) {

	pt := make(playTable)
	pt.addEntry("foo", 0, -15, nil)

	entry, ok := pt.get("bar", 0)
	assert.False(t, ok)
	assert.Nil(t, entry)

	entry2, ok := pt.get("foo", 1)
	assert.False(t, ok)
	assert.Nil(t, entry2)

	entry3, ok := pt.get("foo", 0)
	assert.True(t, ok)
	assert.Nil(t, entry3.move)
	assert.Equal(t, -15, entry3.value)
}

func TestCanBePlayedWith(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_csw15.gaddag")
	alph := gd.GetAlphabet()

	type pwtest struct {
		play     string
		rack     string
		playable bool
		leave    string
	}
	playedWithTests := []pwtest{
		{".....ULARY", "ABLRUY", true, "B"},
		{".....ULARY", "ALRUY", true, ""},
		{".....ULARY", "ALRU", false, ""},
		{".....ULARY", "AALLRUY", true, "AL"},
		{".....ULARY", "VOCAB", false, ""},
		{".....ULARY", "ALRU", false, ""},
		{"OX.P...B..AZ..E", "ABEOPXZ", true, ""},
		{"OX.P...B..AZ..E", "ABEOPXY", false, ""},
		// blanks
		{".....ULArY", "ALRUY", false, ""},
		{".....ULArY", "ALUY?", true, ""},
		{".....ULArY", "ALUY??", true, "?"},
		{".....ULarY", "LUY??", true, ""},
		{".....ULarY", "ALRUY??", true, "AR"},
		{"COoKIES", "CEIKS??", false, ""},
		{"COoKIES", "CEIKOS?", true, ""},
	}

	for _, tt := range playedWithTests {
		mv := move.NewScoringMoveSimple(30, "B3", tt.play, "", alph)
		rack, err := alphabet.ToMachineWord(tt.rack, alph)

		assert.Nil(t, err)

		playable, leave := canBePlayedWith(mv.Tiles(), rack)
		assert.Equal(t, tt.playable, playable,
			"for %v expected %v got %v", tt, tt.playable, playable)
		assert.Equal(t, tt.leave, leave.UserVisible(alph))
	}

}

func TestTilesUsedForWord(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_csw15.gaddag")
	alph := gd.GetAlphabet()

	type tutest struct {
		play  string
		tiles string
	}
	tilesUsedTests := []tutest{
		{".....ULARY", "ALRUY"},
		{"OX.P...B..AZ..E", "ABEOPXZ"},
		// blanks
		{".....ULArY", "ALUY?"},
		{".....ULarY", "LUY??"},
		{"COoKIES", "CEIKOS?"},
		{"COOKIES", "CEIKOOS"},
	}

	for _, tt := range tilesUsedTests {
		mv := move.NewScoringMoveSimple(30, "B3", tt.play, "", alph)
		tilesUsed := tilesUsedForWord(mv.Tiles())
		assert.Equal(t, tt.tiles, tilesUsed.UserVisible(alph))
	}
}

func TestGeneratePlayTables(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_csw15.gaddag")
	b := board.MakeBoard(board.CrosswordGameBoard)
	alph := gd.GetAlphabet()
	dist := alphabet.EnglishLetterDistribution()
	bag := dist.MakeBag(alph)

	generator := movegen.NewGordonGenerator(gd, bag, b, &strategy.NoLeaveStrategy{})
	generator.SetBoardToGame(alph, board.JDvsNB)

	rack := alphabet.RackFromString("RR", alph)

	generator.GenAll(rack)

	s := new(DynamicProgSolver)
	s.Init(b, generator, bag, gd)
	s.generatePlayTables(rack, generator.Plays(), OurPlayTable)
	fmt.Println(s.playTableOurs.Printable(alph))
	assert.False(t, true)
}
